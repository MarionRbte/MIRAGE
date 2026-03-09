# %% [markdown]
# # Phase 3 : Reranking avec VLM

# %%
import os
import json
import numpy as np
import torch
from datasets import load_from_disk
from collections import defaultdict
from tqdm.auto import tqdm
import re
import config

from utils_indexation import SearchIndex
from rerankers import Qwen2VLReranker
from utils_evaluation import evaluate_from_indices

# %% [markdown]
# ## Configuration

# %%
TOP_K_I2T = 5
TOP_K_T2I = 5
MODE_TEST = False
NB_QUERIES_TEST = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Initialisation du Reranking sur {device} (Mode Test: {MODE_TEST})")

vlm = Qwen2VLReranker(model_id="Qwen/Qwen2-VL-7B-Instruct")

# %% [markdown]
# ## Chargement des données

# %%
print("Chargement du Dataset, des Poids et des Matrices...")
test_dataset = load_from_disk(f"{config.RAW_DATA_DIR}/test")

with open(config.BEST_WEIGHTS_FILE, 'r') as f:
    BEST_WEIGHTS = json.load(f)

model_names = list(set(f[5:-13] for f in os.listdir(config.INDEX_DIR) if f.startswith("test_") and f.endswith("_img_vecs.npy")))
txt_vecs, img_vecs = {}, {}
prefix = "test_"

for name in model_names:
    img_vecs[name] = np.load(f"{config.INDEX_DIR}/{prefix}{name}_img_vecs.npy")
    txt_vecs[name] = np.load(f"{config.INDEX_DIR}/{prefix}{name}_txt_vecs.npy")

idx_img = SearchIndex(0); idx_img.load_from_disk(f"{config.INDEX_DIR}/{prefix}{model_names[0]}_img")
test_img_ids = idx_img.image_ids

idx_txt = SearchIndex(0); idx_txt.load_from_disk(f"{config.INDEX_DIR}/{prefix}{model_names[0]}_txt")
test_txt_to_img_id = idx_txt.image_ids

test_img_id_to_idx = {img_id: idx for idx, img_id in enumerate(test_img_ids)}

targets_t2i_test = np.array([test_img_id_to_idx[iid] for iid in test_txt_to_img_id])
test_img_to_txt_indices = defaultdict(list)
for txt_idx, iid in enumerate(test_txt_to_img_id):
    test_img_to_txt_indices[test_img_id_to_idx[iid]].append(txt_idx)
targets_i2t_test = [test_img_to_txt_indices[i] for i in range(len(test_img_id_to_idx))]

targets_t2i_gpu = torch.tensor(targets_t2i_test.reshape(-1, 1), device=device)
max_len_i2t = max(len(t) for t in targets_i2t_test)
targets_i2t_padded = [t + [-1] * (max_len_i2t - len(t)) for t in targets_i2t_test]
targets_i2t_gpu = torch.tensor(targets_i2t_padded, device=device)
def get_fused_matrix(weights_dict):
    return sum(
        w * (torch.tensor(txt_vecs[name], device=device) @ torch.tensor(img_vecs[name], device=device).T)
        for name, w in weights_dict.items() if w > 0
    )

S_t2i_stage1 = get_fused_matrix(BEST_WEIGHTS['t2i']['R@1'])
S_i2t_stage1 = get_fused_matrix(BEST_WEIGHTS['i2t']['R@1']).t()

# N'oubliez pas de générer la liste de tous les textes pour pouvoir y accéder via leur ID !
all_test_texts = [str(c) for item in test_dataset for c in (item['caption'] if isinstance(item['caption'], list) else [item['caption']])]

# %% [markdown]
# ## Exécution de Reranking 

# %%
sorted_idx_i2t = torch.argsort(S_i2t_stage1, dim=1, descending=True)

# Chargement du cache s'il existe
if os.path.exists(config.CACHE_I2T_FILE):
    with open(config.CACHE_I2T_FILE, 'r') as f:
        cache_i2t = json.load(f)
    print(f"✅ Cache I2T chargé : {len(cache_i2t)} requêtes déjà calculées.")
else:
    cache_i2t = {}

limit_i2t = NB_QUERIES_TEST if MODE_TEST else len(targets_i2t_test)
print(f"\nDémarrage Reranking I2T Rapide avec VLM ({limit_i2t} requêtes)...")

# 1. On clone la matrice de résultats initiaux pour pouvoir la modifier
sorted_idx_i2t_reranked = sorted_idx_i2t.clone()
targets_i2t_eval = targets_i2t_gpu[:limit_i2t]

try:
    if hasattr(vlm.model, "disable_adapters") and getattr(vlm.model, "_hf_peft_config_loaded", False):
        print("Désactivation des adaptateurs LoRA pour l'I2T...")
        vlm.model.disable_adapters()
    else:
        print("Modèle de base détecté pour l'I2T (aucun adaptateur à désactiver).")
except ValueError:
    pass

# Votre boucle I2T classique ici...
for q_idx in tqdm(range(limit_i2t)):
    if str(q_idx) in cache_i2t:
        # On restaure l'ordre depuis le cache et on ignore l'appel au VLM
        sorted_idx_i2t_reranked[q_idx, :TOP_K_I2T] = torch.tensor(cache_i2t[str(q_idx)], device=device)
        continue
    image_query = test_dataset[q_idx]['image']
    # On récupère les TOP_K candidats
    top_k_idx = [int(idx) for idx in sorted_idx_i2t[q_idx, :TOP_K_I2T].tolist()]

    textes_candidats = "\n".join([
        f"ID {idx}: {all_test_texts[idx]}" 
        for idx in top_k_idx
    ])


    # Prompt optimisé pour la tâche I2T (Listwise Ranking)
    prompt = f"""You are an expert image-to-text alignment evaluator.
    Here is a query image and {TOP_K_I2T} candidate descriptions with their unique IDs:
    
    {textes_candidats}

    Task: Rank ALL {TOP_K_I2T} IDs from the most accurate description to the least accurate.
    
    CRUCIAL RULES:
    1. Multiple descriptions can be correct! You MUST group ALL factually correct descriptions at the very top of your ranking.
    2. Penalize hallucination: If a text mentions something NOT present in the image (e.g., wrong color, wrong action, wrong object), it must be ranked lower.
    3. Reward factual accuracy: Even short or simple descriptions must be ranked highly if they are 100% visually correct.
    
    First, think step-by-step. Analyze the visual elements of the image and explicitly compare them to the claims made in each candidate description. Point out any hallucinations or exact matches.
    
    After your analysis, you MUST conclude with a single line containing exactly {TOP_K_I2T} IDs in the new order.
    
    Format:
    Final Ranking: [ID1, ID2, ID3, ID4, ID5] (Replace with the actual integer IDs from the candidates above)"""
    response_text = vlm.generate_response(prompt, [image_query])
    
    # --- PARSING DIRECT DES VRAIS IDs ---
    match = re.search(r'Final Ranking:\s*(?:\[)?(.*?)(?:\])?$', response_text, re.IGNORECASE | re.MULTILINE)
    new_order = []
    
    if match:
        # On extrait tous les nombres (les IDs)
        predicted_ids = [int(n) for n in re.findall(r'\d+', match.group(1))]
        
        # Sécurité : on ne garde que les IDs qui font bien partie de nos candidats, sans doublons
        for idx in predicted_ids:
            if idx in top_k_idx and idx not in new_order:
                new_order.append(idx)
        
    # Fallback de sécurité : s'il manque des IDs, on complète avec l'ordre initial du Stage 1
    for idx in top_k_idx:
        if idx not in new_order:
            new_order.append(idx)
            
    # 2. Mise à jour du tenseur avec les nouveaux IDs
    sorted_idx_i2t_reranked[q_idx, :TOP_K_I2T] = torch.tensor(new_order, device=device)
            
    cache_i2t[str(q_idx)] = new_order
    # Sauvegarde sur le disque toutes les 5 requêtes pour ne pas perdre la progression
    if q_idx % 5 == 0: 
        with open(config.CACHE_I2T_FILE, 'w') as f:
            json.dump(cache_i2t, f)
with open(config.CACHE_I2T_FILE, 'w') as f:
    json.dump(cache_i2t, f)

# %%
import pandas as pd
import torch

limit_i2t = NB_QUERIES_TEST if MODE_TEST else len(targets_i2t_test)
targets_i2t_eval = targets_i2t_gpu[:limit_i2t]

# =====================================================================
# CALCUL ET AFFICHAGE DES MÉTRIQUES FINALES
# =====================================================================
print(f"\nÉvaluation des méthodes sur {limit_i2t} requêtes...")

# 1. Évaluation des prédictions initiales (Stage 1)
m_i2t_base = evaluate_from_indices(sorted_idx_i2t[:limit_i2t], targets_i2t_eval)

# 2. Évaluation après le passage du VLM (Stage 2)
# (sorted_idx_i2t_reranked a déjà été rempli lors de ta boucle d'inférence)
m_i2t_vlm_only = evaluate_from_indices(sorted_idx_i2t_reranked[:limit_i2t], targets_i2t_eval)

print("\n" + "="*80)
print("COMPARAISON FINALE : LATE FUSION vs VLM PUR (I2T)")
print("="*80)

df_metrics = pd.DataFrame(
    [m_i2t_base, m_i2t_vlm_only], 
    index=["1. Avant (Stage 1 Fusion)", "2. Après (VLM Pur)"]
)
print(df_metrics.to_string(float_format=lambda x: f"{x:.4f}"))

# %%
# %% [NOUVELLE CELLULE : RERANKING EN CASCADE (THRESHOLD PRE-VLM)]
import os
import json
import re
import torch
from peft import PeftModel 
from tqdm.auto import tqdm

TOP_K_GLOBAL = 10 # Le nombre final de candidats
TOP_K_VLM = 5     # Strictement 5 pour correspondre au fine-tuning LoRA
SEUIL_CONFIANCE = 0.015 # Le "Golden Threshold" que tu as trouvé

sorted_idx_t2i = torch.argsort(S_t2i_stage1, dim=1, descending=True)
limit_t2i = NB_QUERIES_TEST if MODE_TEST else len(targets_t2i_test)

# ⚙️ CHARGEMENT DU MODÈLE ELITE
DOSSIER_LORA_T2I = f"{config.FINETUNING_DIR}/qwen2vl_t2i_lora_elite_cot"

if os.path.exists(DOSSIER_LORA_T2I):
    print(f"⚙️ Injection des poids LoRA ELITE CoT...")
    if not (getattr(vlm.model, "_hf_peft_config_loaded", False) or isinstance(vlm.model, PeftModel)):
        old_no_split = getattr(vlm.model, "_no_split_modules", None)
        vlm.model._no_split_modules = None
        vlm.model = PeftModel.from_pretrained(vlm.model, DOSSIER_LORA_T2I)
        vlm.model._no_split_modules = old_no_split
    vlm.model.eval()
    print("✅ Modèle prêt pour le raisonnement visuel !")

# 📁 CACHE POUR LE RERANKING SÉLECTIF
if os.path.exists(config.CACHE_T2I_FILE):
    with open(config.CACHE_T2I_FILE, 'r') as f: 
        cache_t2i = json.load(f)
else: cache_t2i = {}

sorted_idx_t2i_final = sorted_idx_t2i.clone()
targets_t2i_eval = targets_t2i_gpu[:limit_t2i]
compteur_expert = 0

print(f"\n🚀 Démarrage de l'inférence en cascade (Seuil: {SEUIL_CONFIANCE})...")

# 🔄 BOUCLE DE DÉCISION ET RERANKING
for q_idx in tqdm(range(limit_t2i)):
    # 1. On récupère les 10 meilleurs candidats initiaux
    top_k_global_idx = [int(idx) for idx in sorted_idx_t2i[q_idx, :TOP_K_GLOBAL].tolist()]
    
    # 2. On calcule la confiance de la Phase 1
    score_1 = S_t2i_stage1[q_idx, top_k_global_idx[0]].item()
    score_2 = S_t2i_stage1[q_idx, top_k_global_idx[1]].item()
    ecart = score_1 - score_2
    
    # 3. DÉCISION : Faut-il appeler le VLM ?
    if ecart >= SEUIL_CONFIANCE:
        # La Phase 1 est sûre d'elle, on garde les 10 images dans l'ordre initial
        sorted_idx_t2i_final[q_idx, :TOP_K_GLOBAL] = torch.tensor(top_k_global_idx, device=device)
        continue
        
    # Si on arrive ici, la Phase 1 doute. On appelle le VLM sur le Top 5.
    compteur_expert += 1
    top_5_ambigus = top_k_global_idx[:TOP_K_VLM]
    les_5_suivants = top_k_global_idx[TOP_K_VLM:] # Ceux-ci ne seront pas réévalués
    
    if str(q_idx) in cache_t2i:
        new_top_5 = cache_t2i[str(q_idx)]
    else:
        requete_texte = all_test_texts[q_idx]
        images_pil = [test_dataset[idx]['image'] for idx in top_5_ambigus]
        pos_to_id = {i+1: top_5_ambigus[i] for i in range(TOP_K_VLM)}
        
        prompt = (
            f"You are an expert visual verifier. These {TOP_K_VLM} images are pre-ranked by an AI for the query: '{requete_texte}'. "
            f"Image 1 is mathematically the most probable match. Your task is to verify this. "
            f"Do NOT demote Image 1 unless another image clearly matches the subtle details better. "
            f"Think step by step, penalize visual hallucinations, and output the Final Ranking."
        )
        response_text = vlm.generate_response(prompt, images_pil) 
        
        match = re.search(r'Final Ranking:\s*(?:\[)?(.*?)(?:\])?$', response_text, re.IGNORECASE | re.MULTILINE)
        new_top_5 = []
        
        if match:
            predicted_positions = [int(n) for n in re.findall(r'\d+', match.group(1))]
            for pos in predicted_positions:
                if pos in pos_to_id and pos_to_id[pos] not in new_top_5:
                    new_top_5.append(pos_to_id[pos])
                    
        # Fallback de sécurité
        for idx in top_5_ambigus:
            if idx not in new_top_5: new_top_5.append(idx)
                
        cache_t2i[str(q_idx)] = new_top_5
        if compteur_expert % 5 == 0:
            with open(config.CACHE_T2I_FILE, 'w') as f: json.dump(cache_t2i, f)
            
    # 4. On fusionne l'ordre expert (Top 5) avec le reste (Top 6-10)
    ordre_final = new_top_5 + les_5_suivants
    sorted_idx_t2i_final[q_idx, :TOP_K_GLOBAL] = torch.tensor(ordre_final, device=device)

# Sauvegarde finale du cache
with open(config.CACHE_T2I_FILE, 'w') as f: json.dump(cache_t2i, f)

print(f"✅ Reranking terminé. Le VLM est intervenu sur {compteur_expert} requêtes ambiguës sur {limit_t2i}.")

# %%
# =====================================================================
# ÉVALUATION FINALE T2I
# =====================================================================
print("\n" + "="*80)
print("COMPARAISON FINALE : LATE FUSION vs CASCADE RETRIEVAL (T2I)")
print("="*80)

# On évalue bien sur la tranche [ : TOP_K_GLOBAL ]
m_t2i_base = evaluate_from_indices(sorted_idx_t2i[:limit_t2i], targets_t2i_eval)
m_t2i_cascade = evaluate_from_indices(sorted_idx_t2i_final[:limit_t2i], targets_t2i_eval)

import pandas as pd
df_metrics_t2i = pd.DataFrame(
    [m_t2i_base, m_t2i_cascade], 
    index=["1. Avant (Stage 1 Fusion)", "2. Après (Cascade Threshold)"]
)
print(df_metrics_t2i.to_string(float_format=lambda x: f"{x:.4f}"))

# %%
# =====================================================================
# ANALYSE DÉTAILLÉE T2I (CASCADE VLM)
# =====================================================================
print("\n" + "="*80)
print("🔍 AUTOPSIE DES REQUÊTES AMBIGUËS (T2I)")
print("="*80)

total_ambigus_t2i = 0
sauvetages_t2i = 0
sabotages_t2i = 0
doubles_echecs_t2i = 0
confirmations_t2i = 0

for q_idx in range(limit_t2i):
    top_k_global_idx = [int(idx) for idx in sorted_idx_t2i[q_idx, :TOP_K_GLOBAL].tolist()]
    target_id = int(targets_t2i_eval[q_idx].item())
    
    score_1 = S_t2i_stage1[q_idx, top_k_global_idx[0]].item()
    score_2 = S_t2i_stage1[q_idx, top_k_global_idx[1]].item()
    ecart = score_1 - score_2
    
    # On ne regarde QUE les requêtes où la Phase 1 a douté (envoyées au VLM)
    if ecart < SEUIL_CONFIANCE:
        total_ambigus_t2i += 1
        
        top1_stage1 = top_k_global_idx[0]
        top1_final = int(sorted_idx_t2i_final[q_idx, 0].item())
        
        if top1_stage1 != target_id:
            if top1_final == target_id:
                sauvetages_t2i += 1
            else:
                doubles_echecs_t2i += 1
        else:
            if top1_final != target_id:
                sabotages_t2i += 1
            else:
                confirmations_t2i += 1

print("\n--- BILAN COMPTABLE T2I (Sur les requêtes ambiguës) ---")
print(f"Total de requêtes envoyées au VLM : {total_ambigus_t2i}")
print(f"✅ Sauvetages (Stage 1 faux -> VLM vrai) : {sauvetages_t2i}")
print(f"⚠️ Sabotages (Stage 1 vrai -> VLM faux) : {sabotages_t2i}")
print(f"❌ Doubles Échecs (Faux -> Faux)        : {doubles_echecs_t2i}")
print(f"🆗 Confirmations (Vrai -> Vrai)         : {confirmations_t2i}")

gain_net_t2i = sauvetages_t2i - sabotages_t2i
print(f"\n📈 GAIN NET DE R@1 (T2I) : {gain_net_t2i} requêtes (soit +{gain_net_t2i/limit_t2i*100:.2f}%)")

# %%
# =====================================================================
# ANALYSE DÉTAILLÉE I2T (ZERO-SHOT)
# =====================================================================
print("\n" + "="*80)
print("🔍 AUTOPSIE DES REQUÊTES I2T")
print("="*80)

total_i2t = limit_i2t
sauvetages_i2t = 0
sabotages_i2t = 0
doubles_echecs_i2t = 0
confirmations_i2t = 0

for q_idx in range(limit_i2t):
    # En I2T, il y a plusieurs cibles valides (les -1 sont du padding)
    target_ids = [idx for idx in targets_i2t_eval[q_idx].tolist() if idx != -1]
    
    # Top 1 du Stage 1 (Avant VLM)
    top1_stage1 = int(sorted_idx_i2t[q_idx, 0].item())
    # Top 1 du Stage 2 (Après VLM)
    top1_final = int(sorted_idx_i2t_reranked[q_idx, 0].item())
    
    stage1_etait_correct = top1_stage1 in target_ids
    vlm_est_correct = top1_final in target_ids
    
    if not stage1_etait_correct:
        if vlm_est_correct:
            sauvetages_i2t += 1
        else:
            doubles_echecs_i2t += 1
    else:
        if not vlm_est_correct:
            sabotages_i2t += 1
        else:
            confirmations_i2t += 1

print("\n--- BILAN COMPTABLE I2T (Sur l'ensemble des requêtes) ---")
print(f"Total de requêtes envoyées au VLM : {total_i2t}")
print(f"✅ Sauvetages (Stage 1 faux -> VLM vrai) : {sauvetages_i2t}")
print(f"⚠️ Sabotages (Stage 1 vrai -> VLM faux) : {sabotages_i2t}")
print(f"❌ Doubles Échecs (Faux -> Faux)        : {doubles_echecs_i2t}")
print(f"🆗 Confirmations (Vrai -> Vrai)         : {confirmations_i2t}")

gain_net_i2t = sauvetages_i2t - sabotages_i2t
print(f"\n📈 GAIN NET DE R@1 (I2T) : {gain_net_i2t} requêtes (soit +{gain_net_i2t/limit_i2t*100:.2f}%)")

# %%
# %% [NOUVELLE CELLULE : VISUALISATION DES ERREURS T2I]
import matplotlib.pyplot as plt

print("\n" + "="*80)
print("👀 VISUALISATION DES ERREURS DU VLM (SABOTAGES & DOUBLES ÉCHECS)")
print("="*80)

# On limite à 5 ou 10 affichages pour ne pas faire exploser la taille du notebook
MAX_AFFICHAGES = 10 
erreurs_affichees = 0

for q_idx in range(limit_t2i):
    if erreurs_affichees >= MAX_AFFICHAGES:
        break
        
    top_k_global_idx = [int(idx) for idx in sorted_idx_t2i[q_idx, :TOP_K_GLOBAL].tolist()]
    target_id = int(targets_t2i_eval[q_idx].item())
    
    score_1 = S_t2i_stage1[q_idx, top_k_global_idx[0]].item()
    score_2 = S_t2i_stage1[q_idx, top_k_global_idx[1]].item()
    ecart = score_1 - score_2
    
    # On ne regarde QUE les requêtes traitées par le VLM (celles qui étaient ambiguës)
    if ecart < SEUIL_CONFIANCE:
        top1_stage1 = top_k_global_idx[0]
        top1_final = int(sorted_idx_t2i_final[q_idx, 0].item())
        
        # Condition d'erreur : le Top 1 Final n'est pas la target
        if top1_final != target_id:
            
            # Détermination du type d'erreur
            if top1_stage1 == target_id:
                type_erreur = "⚠️ SABOTAGE (La Phase 1 avait juste, le VLM a changé l'ordre)"
            else:
                type_erreur = "❌ DOUBLE ÉCHEC (La Phase 1 ET le VLM se sont trompés)"
                
            requete_texte = all_test_texts[q_idx]
            img_target = test_dataset[target_id]['image']
            img_predite = test_dataset[top1_final]['image']
            
            # --- AFFICHAGE MATPLOTLIB ---
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Titre global avec la requête
            fig.suptitle(f"Requête [{q_idx}] : '{requete_texte}'\n{type_erreur}", fontsize=12, fontweight='bold')
            
            # Image 1 : La Target (Vérité terrain)
            axes[0].imshow(img_target)
            axes[0].set_title(f"Target Attendue (ID {target_id})", color='green', fontweight='bold')
            axes[0].axis('off')
            
            # Image 2 : La Prédiction du VLM
            axes[1].imshow(img_predite)
            axes[1].set_title(f"Choix du VLM (ID {top1_final})", color='red', fontweight='bold')
            axes[1].axis('off')
            
            plt.tight_layout()
            plt.show()
            
            erreurs_affichees += 1

print(f"✅ Affichage terminé ({erreurs_affichees} erreurs montrées).")

# %%
# %% [NOUVELLE CELLULE : VISUALISATION DES ERREURS I2T]
import matplotlib.pyplot as plt

print("\n" + "="*80)
print("👀 ANALYSE DES ERREURS I2T : IMAGE UNIQUE VS TEXTES CONFLICTUELS")
print("="*80)

MAX_AFFICHAGES_I2T = 26 
erreurs_affichees_i2t = 0

for q_idx in range(limit_i2t):
    if erreurs_affichees_i2t >= MAX_AFFICHAGES_I2T:
        break
        
    # Identification des IDs
    # targets_i2t_eval contient une liste d'IDs valides pour cette image (souvent 5 pour Flickr)
    target_ids = [idx for idx in targets_i2t_eval[q_idx].tolist() if idx != -1]
    
    # Top 1 selon le Stage 1
    top1_stage1 = int(sorted_idx_i2t[q_idx, 0].item())
    # Top 1 final (après passage du VLM)
    top1_vlm = int(sorted_idx_i2t_reranked[q_idx, 0].item())
    
    # On n'affiche que si le VLM s'est trompé (le Top 1 n'est pas dans les targets)
    if top1_vlm not in target_ids:
        erreurs_affichees_i2t += 1
        
        # Détermination du type d'échec
        if top1_stage1 in target_ids:
            type_erreur = "⚠️ SABOTAGE (Le Stage 1 avait juste, le VLM a dégradé la réponse)"
        else:
            type_erreur = "❌ DOUBLE ÉCHEC (Le Stage 1 et le VLM ont tous deux échoué)"
            
        img_query = test_dataset[q_idx]['image']
        texte_attendu = all_test_texts[target_ids[0]] # On prend la première légende valide comme référence
        texte_predit = all_test_texts[top1_vlm]
        
        # --- AFFICHAGE MATPLOTLIB ---
        plt.figure(figsize=(10, 6))
        plt.imshow(img_query)
        plt.axis('off')
        
        # Titre avec les textes pour comparaison directe
        titre = (f"Requête Image [{q_idx}]\n{type_erreur}\n\n"
                 f"✅ ATTENDU : '{texte_attendu}'\n"
                 f"❌ PRÉDIT : '{texte_predit}'")
        
        plt.title(titre, fontsize=10, loc='left', fontweight='bold', pad=20)
        plt.tight_layout()
        plt.show()

print(f"✅ Analyse terminée ({erreurs_affichees_i2t} erreurs I2T visualisées).")


