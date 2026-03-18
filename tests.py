# %% [markdown]
# # Tests

# %%
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import config
from tqdm.auto import tqdm
from peft import PeftModel
import time
import json
import re
import json
import numpy as np
import pandas as pd
from IPython.display import display
from qwen_vl_utils import process_vision_info


# Import de tes modules personnalisés
from utils_data import load_reranking_data
from utils_reranking import RerankingCache, parse_vlm_ranking, calibrate_confidence_threshold
from utils_analysis import compute_autopsy, evaluate_and_save_results
from rerankers import Qwen2VLReranker

# ==========================================
# CONFIGURATION
# ==========================================
TOP_K_I2T = 10
TOP_K_T2I = 10
MODE_TEST = False 
NB_QUERIES_TEST = 1000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Chargement du VLM (Modèle de base)
vlm = Qwen2VLReranker(model_id="Qwen/Qwen2-VL-7B-Instruct", device=device)
# ==========================================
# CHARGEMENT DES DONNÉES
# ==========================================
# 1. On charge la validation pour trouver le seuil optimal
val_dataset, val_texts , S_t2i_val_stage1, S_i2t_val_stage1, targets_t2i_val_gpu, targets_i2t_val_gpu = load_reranking_data("val", device)

# 2. On charge le set de test pour l'évaluation finale
dataset, all_texts, S_t2i_stage1, S_i2t_stage1, targets_t2i_gpu, targets_i2t_gpu = load_reranking_data("test", device)

limit_i2t = NB_QUERIES_TEST if MODE_TEST else len(targets_i2t_gpu)
limit_t2i = NB_QUERIES_TEST if MODE_TEST else len(targets_t2i_gpu)

# %% [markdown]
# ## I2T

# %%
'''
# ## Reranking I2T (SELF-CRITIQUE SOTA - 2ème Passe)

# %%
print("\n" + "="*50)
print("🚀 DÉMARRAGE I2T (SELF-CRITIQUE - 2ème Passe)")
print("="*50)

# 1. Chargement de tes prédictions initiales du VLM (Le fameux 0.971)
if not os.path.exists(config.CACHE_I2T_FILE):
    raise FileNotFoundError("Le cache initial est introuvable ! Lancez la première passe d'abord.")
    
with open(config.CACHE_I2T_FILE, 'r') as f:
    initial_vlm_predictions = json.load(f)

# 2. Création d'un NOUVEAU cache pour la critique (pour ne rien écraser)
CACHE_CRITIQUE_FILE = config.CACHE_I2T_FILE.replace('.json', '_critique.json')
cache_critique = RerankingCache(CACHE_CRITIQUE_FILE)

sorted_idx_i2t_base = torch.argsort(S_i2t_stage1, dim=1, descending=True)
sorted_idx_i2t_final = sorted_idx_i2t_base.clone()

# Sécurité LoRA
if hasattr(vlm.model, "disable_adapters") and getattr(vlm.model, "_hf_peft_config_loaded", False):
    vlm.model.disable_adapters()

temps_vlm_reel = 0.0
appels_vlm_reels = 0

for q_idx in tqdm(range(limit_i2t), desc="Inférence I2T Self-Critique"):
    top_k_idx = sorted_idx_i2t_base[q_idx, :TOP_K_I2T].tolist()
    
    # On vérifie si la critique a déjà été faite
    cached = cache_critique.get(q_idx)
    if cached:
        sorted_idx_i2t_final[q_idx, :TOP_K_I2T] = torch.tensor(cached, device=device)
        continue
        
    # On récupère l'ordre initial que le VLM avait choisi lors de la passe 1
    ordre_initial = initial_vlm_predictions[str(q_idx)]
    choix_top_1 = ordre_initial[0]
    
    image_query = dataset[q_idx]['image']
    textes_candidats = "\n".join([f"ID {idx}: {all_texts[idx]}" for idx in top_k_idx])
    
    # 🕵️‍♂️ LE PROMPT DE SELF-CRITIQUE
    prompt = f"""You are an expert image-to-text alignment evaluator.
    Here is a query image and {TOP_K_I2T} candidate descriptions with their unique IDs:
    
    {textes_candidats}

    Previously, you analyzed this image and ranked ID {choix_top_1} as the absolute best match.
    
    Your task: CRITICALLY RE-EVALUATE your choice.
    Look at the image very closely. Does the description in ID {choix_top_1} contain ANY subtle hallucination? (e.g., mentioning a color, an action, an object, or a background detail that is NOT visually present).
    
    - If ID {choix_top_1} is 100% flawless, you MUST keep it at the top.
    - If ID {choix_top_1} contains even a tiny hallucination, you MUST demote it and promote the true perfect match to the top.
    
    Think step-by-step to justify if your previous choice was correct or flawed.
    After your analysis, output the final ranking.
    
    Format:
    Final Ranking: [ID1, ID2, ID3, ID4, ID5]"""
    
    t0_call = time.time()
    
    response_text = vlm.generate_response(prompt, [image_query])
    
    t1_call = time.time()
    temps_vlm_reel += (t1_call - t0_call)
    appels_vlm_reels += 1

    # Parsing
    new_order = parse_vlm_ranking(response_text, top_k_idx)
    
    # Fallback si le parsing échoue ou renvoie une liste vide, on garde le 1er choix du VLM
    if len(new_order) == 0:
        new_order = ordre_initial
        
    sorted_idx_i2t_final[q_idx, :TOP_K_I2T] = torch.tensor(new_order, device=device)
    cache_critique.set(q_idx, new_order)
    
    if appels_vlm_reels % 5 == 0: 
        cache_critique.save()

cache_critique.save()

temps_total_vlm_i2t = temps_vlm_reel if appels_vlm_reels > 0 else 0.0

# --- Bilan I2T ---
evaluate_and_save_results(
    sorted_idx_i2t_base[:limit_i2t], 
    sorted_idx_i2t_final[:limit_i2t], 
    targets_i2t_gpu[:limit_i2t], 
    is_i2t=True, 
    csv_path=config.metriques_i2t_csv.replace('.csv', '_critique.csv'), 
    md_path=config.metriques_i2t_md.replace('.md', '_critique.md'),
    temps_vlm=temps_total_vlm_i2t
)

_ = compute_autopsy(sorted_idx_i2t_base, sorted_idx_i2t_final, targets_i2t_gpu[:limit_i2t], is_i2t=True, mask_vlm_called=[True]*limit_i2t)
'''

# %%
'''
print("\n" + "="*50)
print("🚀 DÉMARRAGE I2T (FULL LISTWISE ZERO-SHOT - SANS CASCADE)")
print("="*50)

#cache_i2t = RerankingCache(config.CACHE_I2T_FILE.replace('.json', '_512.json'))
cache_i2t = RerankingCache(config.CACHE_I2T_FILE)
sorted_idx_i2t_base = torch.argsort(S_i2t_stage1, dim=1, descending=True)
sorted_idx_i2t_final = sorted_idx_i2t_base.clone()

# Sécurité : on s'assure que le VLM n'utilise pas LoRA pour la tâche Zero-Shot
if hasattr(vlm.model, "disable_adapters") and getattr(vlm.model, "_hf_peft_config_loaded", False):
    print("⚙️ Désactivation des adaptateurs LoRA pour test Listwise Zero-Shot...")
    vlm.model.disable_adapters()

temps_vlm_reel = 0.0
appels_vlm_reels = 0

# Pas de masque conditionnel ici car on envoie TOUTES les requêtes (Full Reranking)
mask_vlm_called_i2t = [True] * limit_i2t

for q_idx in tqdm(range(limit_i2t), desc="Inférence I2T Full Listwise"):
    top_k_idx = sorted_idx_i2t_base[q_idx, :TOP_K_I2T].tolist()
    
    # Vérification du cache
    cached = cache_i2t.get(q_idx)
    if cached:
        sorted_idx_i2t_final[q_idx, :TOP_K_I2T] = torch.tensor(cached, device=device)
        continue
        
    image_query = dataset[q_idx]['image']
    textes_candidats = "\n".join([f"ID {idx}: {all_texts[idx]}" for idx in top_k_idx])
    
    # 🥇 TON ANCIEN PROMPT OPTIMISÉ (Le secret du 0.975)
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
    Final Ranking: [ID1, ID2, ID3, ID4, ID5]"""
    
    t0_call = time.time()
    
    # Génération
    response_text = vlm.generate_response(prompt, [image_query])
    
    t1_call = time.time()
    temps_vlm_reel += (t1_call - t0_call)
    appels_vlm_reels += 1

    # --- PARSING PROPRE ---
    new_order = parse_vlm_ranking(response_text, top_k_idx)
    
    sorted_idx_i2t_final[q_idx, :TOP_K_I2T] = torch.tensor(new_order, device=device)
    cache_i2t.set(q_idx, new_order)
    
    if appels_vlm_reels % 5 == 0: 
        cache_i2t.save()

cache_i2t.save()

# --- Calcul du temps projeté ---
temps_total_vlm_i2t = temps_vlm_reel if appels_vlm_reels > 0 else 0.0

# --- Bilan I2T (Métriques et Sauvegarde) ---
evaluate_and_save_results(
    sorted_idx_i2t_base[:limit_i2t], 
    sorted_idx_i2t_final[:limit_i2t], 
    targets_i2t_gpu[:limit_i2t], 
    is_i2t=True, 
    csv_path=config.metriques_i2t_csv, 
    md_path=config.metriques_i2t_md,
    temps_vlm=temps_total_vlm_i2t
)

# --- Autopsie I2T (Sauvetages/Sabotages) ---
_ = compute_autopsy(sorted_idx_i2t_base, sorted_idx_i2t_final, targets_i2t_gpu[:limit_i2t], is_i2t=True, mask_vlm_called=mask_vlm_called_i2t)
'''

# %%
# 2. CONFIGURATION DES DEUX PASSES
passes_extraction_i2t = [
    {
        "nom": "Validation",
        "limite": len(targets_i2t_val_gpu),
        "S_base": S_i2t_val_stage1,
        "dataset": val_dataset,
        "textes": val_texts,
        "fichier_out": os.path.join(config.RESULTS_DIR, "scores_bruts_i2t_val.json")
    },
    {
        "nom": "Test",
        "limite": limit_i2t,
        "S_base": S_i2t_stage1,
        "dataset": dataset,
        "textes": all_texts,
        "fichier_out": os.path.join(config.RESULTS_DIR, "scores_bruts_i2t_test.json") 
    }
]

# 3. BOUCLE D'EXTRACTION EXHAUSTIVE I2T
for passe in passes_extraction_i2t:
    print(f"\n" + "="*50)
    print(f"🚀 DÉMARRAGE EXTRACTION I2T EXHAUSTIVE : Set de {passe['nom']}")
    print("="*50)
    
    fichier_out = passe['fichier_out']
    S_base = passe['S_base']
    limite = passe['limite']
    data_images = passe['dataset']
    data_textes = passe['textes']
    
    # Recalcul du Top K de base pour le set en cours
    sorted_idx_base = torch.argsort(S_base, dim=1, descending=True)
    scores_all_queries_i2t = {}

    # Chargement du cache existant
    if os.path.exists(fichier_out):
        with open(fichier_out, 'r') as f:
            scores_all_queries_i2t = json.load(f)

    for q_idx in tqdm(range(limite), desc=f"Extraction I2T Listwise ({passe['nom']})"):
        str_qidx = str(q_idx)
        top_k_idx = sorted_idx_base[q_idx, :TOP_K_I2T].tolist()
        
        if str_qidx in scores_all_queries_i2t:
            continue

        # Préparation du prompt Listwise
        image_query = data_images[q_idx]['image']
        textes_candidats = "\n".join([f"ID {idx}: {data_textes[idx]}" for idx in top_k_idx])
        
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
        Final Ranking: [ID_A, ID_B, ID_C, ...]"""
        
        # Inférence VLM
        response_text = vlm.generate_response(prompt, [image_query])
        
        # Parsing
        new_order = parse_vlm_ranking(response_text, top_k_idx)
        
        # 🎯 Conversion du classement en Pseudo-Scores
        scores_vlm = []
        for idx in top_k_idx:
            if idx in new_order:
                rank = new_order.index(idx)
                # Formule linéaire : 1er=1.0, 2e=0.9, ..., 10e=0.1
                score = 1.0 - (rank / len(top_k_idx)) 
            else:
                score = 0.0 # Pénalité si le VLM a oublié cet ID
            scores_vlm.append(score)

        # Récupération des scores MIRAGE bruts
        scores_mirage = [S_base[q_idx, idx].item() for idx in top_k_idx]
        
        scores_all_queries_i2t[str_qidx] = {
            "candidate_ids": top_k_idx,
            "vlm_scores": scores_vlm,
            "mirage_scores": scores_mirage
        }
        
        with open(fichier_out, 'w') as f:
            json.dump(scores_all_queries_i2t, f)

    # Sauvegarde finale pour ce set
    with open(fichier_out, 'w') as f:
        json.dump(scores_all_queries_i2t, f)

    print(f"✅ Extraction I2T totale terminée et sauvegardée dans : {fichier_out}")

# %%
def evaluate_fusion_i2t(scores_dict, targets_gpu, method="additive", alpha=0.5, rrf_k=60, top_k=10, cascade_threshold=None):
    """Calcule le R@1 en simulant dynamiquement le Top-K et la Cascade pour l'I2T."""
    correct_count = 0
    total = len(scores_dict)
    reranked_results = {}
    mask_vlm_called = {} 
    
    for q_idx_str, data in scores_dict.items():
        q_idx = int(q_idx_str)
        
        # 🎯 GESTION DES MULTIPLES TARGETS POUR L'I2T
        target_tensor = targets_gpu[q_idx]
        if target_tensor.dim() == 0:
            valid_targets = [target_tensor.item()]
        else:
            valid_targets = [int(t) for t in target_tensor.tolist() if t != -1]
        
        ids = data["candidate_ids"][:top_k]
        vlm = np.array(data["vlm_scores"][:top_k])
        mirage = np.array(data["mirage_scores"][:top_k])
        
        # 🛡️ --- LOGIQUE DE CASCADE --- 🛡️
        ecart_confiance = mirage[0] - mirage[1] if len(mirage) > 1 else 0
        
        if cascade_threshold is not None and ecart_confiance > cascade_threshold:
            new_top_k = ids
            mask_vlm_called[q_idx] = False
        else:
            mask_vlm_called[q_idx] = True
            mirage_norm = (mirage - mirage.min()) / (mirage.max() - mirage.min() + 1e-8)
            
            if method == "additive":
                final_scores = (alpha * vlm) + ((1 - alpha) * mirage_norm)
            elif method == "multiplicative":
                final_scores = vlm * mirage_norm
            elif method == "maximum":
                final_scores = np.maximum(vlm, mirage_norm)
            elif method == "concatenation":
                final_scores = (vlm * 1000.0) + mirage_norm
            elif method == "rrf":
                rank_vlm = np.argsort(np.argsort(-vlm)) + 1
                rank_mirage = np.argsort(np.argsort(-mirage_norm)) + 1
                final_scores = (1.0 / (rrf_k + rank_vlm)) + (1.0 / (rrf_k + rank_mirage))
            elif method == "harmonique":
                final_scores = 2 * (vlm * mirage_norm) / (vlm + mirage_norm + 1e-8)
            elif method == "geometrique":
                final_scores = np.sqrt(vlm * mirage_norm)
            elif method == "rank_penalty":
                rank_mirage = np.argsort(np.argsort(-mirage)) # 0 pour le 1er, 1 pour le 2ème...
                # On retire 5% du score VLM pour chaque place de retard dans MIRAGE
                final_scores = vlm - (0.05 * rank_mirage) 
            elif method == "vlm_seul": # N'oublie pas celle-ci pour prouver l'utilité de ta fusion !
                final_scores = vlm
                
            best_indices = np.argsort(-final_scores) 
            new_top_k = [ids[i] for i in best_indices]
            
        reranked_results[q_idx] = new_top_k
        
        # 🎯 VÉRIFICATION I2T : Le top 1 est-il dans la liste des descriptions valides ?
        if new_top_k[0] in valid_targets:
            correct_count += 1
            
    return correct_count / total, reranked_results, mask_vlm_called


# ==========================================
# 0. PRÉPARATION DES DONNÉES (VAL ET TEST)
# ==========================================
RAW_SCORES_I2T_VAL_FILE = os.path.join(config.RESULTS_DIR, "scores_bruts_i2t_val_bis.json")
RAW_SCORES_I2T_TEST_FILE = os.path.join(config.RESULTS_DIR, "scores_bruts_i2t_test_bis.json")

with open(RAW_SCORES_I2T_VAL_FILE, 'r') as f:
    scores_bruts_val = json.load(f)
with open(RAW_SCORES_I2T_TEST_FILE, 'r') as f:
    scores_bruts_test = json.load(f)

# On s'assure d'avoir la matrice de base I2T pour les deux
sorted_idx_i2t_val_base = torch.argsort(S_i2t_val_stage1, dim=1, descending=True)
sorted_idx_i2t_test_base = torch.argsort(S_i2t_stage1, dim=1, descending=True)

# ==========================================
# 1. PHASE 1 : GRID SEARCH SUR VALIDATION
# ==========================================
print("\n" + "="*70)
print("🧪 PHASE 1 : RECHERCHE DES HYPERPARAMÈTRES I2T (SET DE VALIDATION)")
print("="*70)

# Attention: comme ton VLM a été prompté pour classer le Top 5 ou 10, 
options_top_k = [5, 10] 
methodes_simples = ["vlm_seul", "multiplicative", "maximum", "concatenation", "rrf", "harmonique", "geometrique", "rank_penalty"]
print("🎯 Calibrage des seuils de cascade sur le set de validation I2T...")
options_cascade = {"Sans Cascade": None}
recalls_a_tester = [0.70, 0.80, 0.85, 0.90, 0.95, 0.99, 1.00]

for tr in recalls_a_tester:
    seuil = calibrate_confidence_threshold(S_i2t_val_stage1, targets_i2t_val_gpu, is_i2t=True, target_recall=tr)
    options_cascade[f"Avec (Recall={tr:.2f})"] = seuil

resultats_val = []
grand_gagnant_val = {"r1": 0, "nom": "", "params": {}}

# Exécution de toutes les combinaisons sur le VAL
for top_k in options_top_k:
    for nom_cascade, val_cascade in options_cascade.items():
        
        # 1. Tests des méthodes simples
        for methode in methodes_simples:
            r1, _, mask = evaluate_fusion_i2t(scores_bruts_val, targets_i2t_val_gpu, method=methode, top_k=top_k, cascade_threshold=val_cascade)
            req_sauvees = sum(not v for v in mask.values())
            
            resultats_val.append({
                "Profondeur": f"Top {top_k}",
                "Cascade": nom_cascade.split(" ")[0], 
                "Stratégie de Fusion": methode.capitalize(),
                "R@1 Val (%)": r1 * 100,
                "VLM Évités": req_sauvees
            })
            
            if r1 > grand_gagnant_val["r1"]:
                grand_gagnant_val = {"r1": r1, "nom": f"{methode.capitalize()}_Top{top_k}_{nom_cascade.split(' ')[0]}", 
                                     "params": {"method": methode, "alpha": 0.5, "top_k": top_k, "cascade_threshold": val_cascade}}

        # 2. Test Additif avec Grid Search Intégré
        best_add_r1, best_alpha = 0, 0
        best_add_mask = {}
        for a in np.linspace(0, 1, 21):
            r1, _, mask = evaluate_fusion_i2t(scores_bruts_val, targets_i2t_val_gpu, method="additive", alpha=a, top_k=top_k, cascade_threshold=val_cascade)
            if r1 > best_add_r1:
                best_add_r1 = r1
                best_alpha = a
                best_add_mask = mask
                
        req_sauvees_add = sum(not v for v in best_add_mask.values())
        resultats_val.append({
            "Profondeur": f"Top {top_k}",
            "Cascade": nom_cascade.split(" ")[0],
            "Stratégie de Fusion": f"Grid Search (Alpha={best_alpha:.2f})",
            "R@1 Val (%)": best_add_r1 * 100,
            "VLM Évités": req_sauvees_add
        })
        
        if best_add_r1 > grand_gagnant_val["r1"]:
            grand_gagnant_val = {"r1": best_add_r1, "nom": f"Grid_Search_A{best_alpha:.2f}_Top{top_k}_{nom_cascade.split(' ')[0]}", 
                                 "params": {"method": "additive", "alpha": best_alpha, "top_k": top_k, "cascade_threshold": val_cascade}}


# Affichage du classement Validation
df_val = pd.DataFrame(resultats_val).sort_values(by=["R@1 Val (%)", "VLM Évités"], ascending=[False, False]).reset_index(drop=True)
display(df_val) # On n'affiche que le Top 10 pour ne pas surcharger

print("\n" + "⭐ "*15)
print(f"LA MEILLEURE CONFIGURATION I2T (VAL) EST : {grand_gagnant_val['nom']}")
print(f"R@1 sur Validation : {grand_gagnant_val['r1']*100:.2f} %")
print("⭐ "*15 + "\n")


# ==========================================
# 2. PHASE 2 : APPLICATION SUR LE TEST
# ==========================================
print("🔒 Hyperparamètres verrouillés. Déploiement sur le Set de Test...")
print("="*70)

# On extrait les paramètres parfaits
p = grand_gagnant_val["params"]

# 🎯 Un seul appel à la fonction sur le set de Test !
test_r1, dict_test, mask_test = evaluate_fusion_i2t(
    scores_bruts_test, 
    targets_i2t_gpu, 
    method=p["method"], 
    alpha=p["alpha"], 
    top_k=p["top_k"], 
    cascade_threshold=p["cascade_threshold"]
)

print(f"\n🏆 PERFORMANCE FINALE RÉELLE I2T (R@1 TEST) : {test_r1*100:.2f} % 🏆")
print(f"⚡ VLM économisés sur le Test : {sum(not v for v in mask_test.values())} requêtes\n")


# ==========================================
# 3. RECONSTRUCTION ET AUTOPSIE DU GAGNANT
# ==========================================
print(f"⚙️ Sauvegarde et Autopsie I2T de la configuration finale ({grand_gagnant_val['nom']})...")

sorted_idx_i2t_final = sorted_idx_i2t_test_base.clone()
for q_idx_str, new_top_k in dict_test.items():
    q_idx = int(q_idx_str)
    ordre_final = new_top_k + sorted_idx_i2t_test_base[q_idx, len(new_top_k):].tolist()
    sorted_idx_i2t_final[q_idx, :len(ordre_final)] = torch.tensor(ordre_final, device=device)

mask_vlm_called_list = [False] * limit_i2t
for q_idx_str, was_called in mask_test.items():
    mask_vlm_called_list[int(q_idx_str)] = was_called

file_suffix = f"_{grand_gagnant_val['nom'].replace(' ', '_').replace('.', '').lower()}_strict_test"

evaluate_and_save_results(
    sorted_idx_i2t_test_base[:limit_i2t], 
    sorted_idx_i2t_final[:limit_i2t], 
    targets_i2t_gpu[:limit_i2t], 
    is_i2t=True,  # IMPORTANT
    csv_path=config.metriques_i2t_csv.replace('.csv', f'{file_suffix}.csv'), 
    md_path=config.metriques_i2t_md.replace('.md', f'{file_suffix}.md'),
    temps_vlm=0.0
)

_ = compute_autopsy(
    sorted_idx_i2t_test_base, 
    sorted_idx_i2t_final, 
    targets_i2t_gpu[:limit_i2t], 
    is_i2t=True,  # IMPORTANT
    mask_vlm_called=mask_vlm_called_list
)

# %%
# ==========================================
# 📊 ANALYSE DÉTAILLÉE DU RECALL (R@1 à R@5)
# ==========================================
print("\n" + "="*50)
print("📊 DISTRIBUTION DES PERFORMANCES (R@1 à R@5) I2T")
print("="*50)

recalls = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
total_queries = len(dict_test)

for q_idx_str, new_top_k in dict_test.items():
    q_idx = int(q_idx_str)
    
    # Récupération sécurisée des targets (comme dans ta fonction)
    target_tensor = targets_i2t_gpu[q_idx]
    if target_tensor.dim() == 0:
        valid_targets = [target_tensor.item()]
    else:
        valid_targets = [int(t) for t in target_tensor.tolist() if t != -1]
        
    # Calcul des Hits pour chaque profondeur (de 1 à 5)
    for k in range(1, 6):
        # Si au moins l'une des descriptions valides est dans le Top K actuel
        if any(t in new_top_k[:k] for t in valid_targets):
            recalls[k] += 1

# Affichage des résultats
for k in range(1, 6):
    score = (recalls[k] / total_queries) * 100
    print(f"R@{k} : {score:.2f} %")
print("="*50 + "\n")

# %% [markdown]
# ## T2I

# %% [markdown]
# ### Pointwise Prompt Ensembling (Sans cascade)

# %%
'''
print("\n" + "="*50)
print("🚀 DÉMARRAGE T2I (POINTWISE SOTA + PROMPT ENSEMBLING)")
print("="*50)

# 1. Désactivation du LoRA
if hasattr(vlm.model, "disable_adapters") and getattr(vlm.model, "_hf_peft_config_loaded", False):
    print("⚙️ Désactivation des adaptateurs LoRA pour test Pointwise Zero-Shot...")
    vlm.model.disable_adapters()
else:
    print("✅ Le modèle est bien en mode de base (Zero-Shot).")

# 2. Initialisation d'un nouveau cache pour l'Ensembling
CACHE_ENSEMBLE_FILE = config.CACHE_T2I_FILE.replace('.json', '_ensemble.json')
cache_t2i = RerankingCache(CACHE_ENSEMBLE_FILE)

sorted_idx_t2i_base = torch.argsort(S_t2i_stage1, dim=1, descending=True)
sorted_idx_t2i_final = sorted_idx_t2i_base.clone()

temps_vlm_reel = 0.0
appels_vlm_reels = 0
ALPHA = 0.5 

for q_idx in tqdm(range(limit_t2i), desc="Inférence T2I Prompt Ensembling"):
    top_5 = sorted_idx_t2i_base[q_idx, :TOP_K_T2I].tolist()
    
    # Vérification du cache
    cached = cache_t2i.get(q_idx)
    if cached:
        new_top_5 = cached
    else:
        images_pil = [dataset[idx]['image'] for idx in top_5]
        requete = all_texts[q_idx]
        
        # 📝 LES 3 PROMPTS SOTA
        prompt_1 = f"Does the following image exactly match this description: '{requete}'? Answer strictly with 'Yes' or 'No'."
        prompt_2 = f"Look closely at the fine details and the background. Is this a perfect visual representation of: '{requete}'? Answer strictly with 'Yes' or 'No'."
        prompt_3 = f"Is the description '{requete}' 100% factually correct for every element visible in this image? Answer strictly with 'Yes' or 'No'."
        
        t0_call = time.time()
        
        # On score les 5 images pour chaque prompt (3 appels successifs au batch)
        scores_p1 = vlm.score_image_pointwise_batch([prompt_1] * 5, images_pil)
        scores_p2 = vlm.score_image_pointwise_batch([prompt_2] * 5, images_pil)
        scores_p3 = vlm.score_image_pointwise_batch([prompt_3] * 5, images_pil)
        
        # On fait la moyenne des 3 probabilités VLM pour chaque image
        scores_vlm_moyens = [(s1 + s2 + s3) / 3.0 for s1, s2, s3 in zip(scores_p1, scores_p2, scores_p3)]
        
        t1_call = time.time()

        temps_vlm_reel += (t1_call - t0_call)
        appels_vlm_reels += 1 

        # --- LATE FUSION ---
        scores_mirage_bruts = [S_t2i_stage1[q_idx, idx].item() for idx in top_5]
        
        # Normalisation Min-Max
        max_s = max(scores_mirage_bruts)
        min_s = min(scores_mirage_bruts)
        if max_s > min_s:
            scores_mirage_norm = [(s - min_s) / (max_s - min_s) for s in scores_mirage_bruts]
        else:
            scores_mirage_norm = [1.0] * 5

        # Fusion mathématique (Alpha * VLM_Moyen + (1-Alpha) * MIRAGE)
        final_scores = []
        for i, idx in enumerate(top_5):
            score_mixte = ((1 - ALPHA) * scores_mirage_norm[i]) + (ALPHA * scores_vlm_moyens[i])
            final_scores.append((score_mixte, idx))
            
        # Nouveau tri
        final_scores.sort(key=lambda x: x[0], reverse=True)
        new_top_5 = [idx for score, idx in final_scores]
        
        # Mise en cache
        cache_t2i.set(q_idx, new_top_5)
        if appels_vlm_reels % 5 == 0: 
            cache_t2i.save()
            
    ordre_final = new_top_5 + top_5[TOP_K_T2I:] if len(top_5) > TOP_K_T2I else new_top_5 + sorted_idx_t2i_base[q_idx, TOP_K_T2I:].tolist()
    sorted_idx_t2i_final[q_idx, :TOP_K_T2I] = torch.tensor(new_top_5, device=device)

cache_t2i.save()

# --- Calcul du temps projeté ---
temps_total_vlm_t2i = temps_vlm_reel if appels_vlm_reels > 0 else 0.0

# --- Bilan T2I ---
evaluate_and_save_results(
    sorted_idx_t2i_base[:limit_t2i], 
    sorted_idx_t2i_final[:limit_t2i], 
    targets_t2i_gpu[:limit_t2i], 
    is_i2t=False, 
    csv_path=config.metriques_t2i_csv.replace('.csv', '_ensemble.csv'), 
    md_path=config.metriques_t2i_md.replace('.md', '_ensemble.md'),
    temps_vlm=temps_total_vlm_t2i
)

# --- Autopsie T2I ---
_ = compute_autopsy(sorted_idx_t2i_base, sorted_idx_t2i_final, targets_t2i_gpu[:limit_t2i], is_i2t=False, mask_vlm_called=[True]*limit_t2i)
'''

# %% [markdown]
# ### Pointwise Top 5 (Sans cascade, Late Fusion α)

# %%
'''
print("\n" + "="*50)
print("🚀 DÉMARRAGE T2I (FULL POINTWISE SOTA - Sans Cascade)")
print("="*50)

# 1. Désactivation du LoRA (Le Pointwise fonctionne mieux sur le modèle de base non biaisé)
if hasattr(vlm.model, "disable_adapters") and getattr(vlm.model, "_hf_peft_config_loaded", False):
    print("⚙️ Désactivation des adaptateurs LoRA pour test Pointwise Zero-Shot...")
    vlm.model.disable_adapters()
else:
    print("✅ Le modèle est bien en mode de base (Zero-Shot).")

# 2. Initialisation
cache_t2i = RerankingCache(config.CACHE_T2I_FILE)
sorted_idx_t2i_base = torch.argsort(S_t2i_stage1, dim=1, descending=True)
sorted_idx_t2i_final = sorted_idx_t2i_base.clone()

temps_vlm_reel = 0.0
appels_vlm_reels = 0

# ⚖️ Poids de la fusion (0.5 = 50% MIRAGE / 50% VLM)
ALPHA = 0.5 

# Comme on envoie TOUT, le mask est 100% True
mask_vlm_called = [True] * limit_t2i

for q_idx in tqdm(range(limit_t2i), desc="Inférence T2I Full Pointwise"):
    top_5 = sorted_idx_t2i_base[q_idx, :TOP_K_T2I].tolist()
    
    # Vérification du cache
    cached = cache_t2i.get(q_idx)
    if cached:
        new_top_5 = cached
    else:
        images_pil = [dataset[idx]['image'] for idx in top_5]
        requete = all_texts[q_idx]
        
        # ⚡ VERSION BATCHÉE T2I ⚡
        prompt = f"Does the following image exactly and perfectly match this description: '{requete}'? Look at every tiny detail. Answer strictly with 'Yes' or 'No'."
        prompts_batch = [prompt] * 5
        
        t0_call = time.time()
        
        # On score les 5 images D'UN COUP avec le même texte
        scores_vlm = vlm.score_image_pointwise_batch(prompts_batch, images_pil)
        
        t1_call = time.time()

        temps_vlm_reel += (t1_call - t0_call)
        appels_vlm_reels += 1 # On compte ça comme 1 requête complète traitée

        # --- LATE FUSION ---
        # 1. On récupère les scores bruts de MIRAGE pour le top 5
        scores_mirage_bruts = [S_t2i_stage1[q_idx, idx].item() for idx in top_5]
        
        # 2. Normalisation Min-Max locale des scores MIRAGE pour qu'ils soient entre 0 et 1
        max_s = max(scores_mirage_bruts)
        min_s = min(scores_mirage_bruts)
        if max_s > min_s:
            scores_mirage_norm = [(s - min_s) / (max_s - min_s) for s in scores_mirage_bruts]
        else:
            scores_mirage_norm = [1.0] * 5

        # 3. Fusion mathématique (Alpha * VLM + (1-Alpha) * MIRAGE)
        final_scores = []
        for i, idx in enumerate(top_5):
            score_mixte = ((1 - ALPHA) * scores_mirage_norm[i]) + (ALPHA * scores_vlm[i])
            final_scores.append((score_mixte, idx))
            
        # 4. Nouveau tri basé sur le score fusionné
        final_scores.sort(key=lambda x: x[0], reverse=True)
        new_top_5 = [idx for score, idx in final_scores]
        
        # Mise en cache
        cache_t2i.set(q_idx, new_top_5)
        if appels_vlm_reels % 5 == 0: 
            cache_t2i.save()
            
    ordre_final = new_top_5 + top_5[TOP_K_T2I:] if len(top_5) > TOP_K_T2I else new_top_5 + sorted_idx_t2i_base[q_idx, TOP_K_T2I:].tolist()
    sorted_idx_t2i_final[q_idx, :TOP_K_T2I] = torch.tensor(new_top_5, device=device)

cache_t2i.save()

# --- Calcul du temps projeté ---
temps_total_vlm_t2i = temps_vlm_reel if appels_vlm_reels > 0 else 0.0

# --- Bilan T2I (Métriques et Sauvegarde) ---
evaluate_and_save_results(
    sorted_idx_t2i_base[:limit_t2i], 
    sorted_idx_t2i_final[:limit_t2i], 
    targets_t2i_gpu[:limit_t2i], 
    is_i2t=False, 
    csv_path=config.metriques_t2i_csv, 
    md_path=config.metriques_t2i_md,
    temps_vlm=temps_total_vlm_t2i
)

# --- Autopsie T2I (Sauvetages/Sabotages) ---
_ = compute_autopsy(sorted_idx_t2i_base, sorted_idx_t2i_final, targets_t2i_gpu[:limit_t2i], is_i2t=False, mask_vlm_called=mask_vlm_called)
'''

# %% [markdown]
# ### Listwise LoRA (Juge Final sur Top 5)

# %%
# Tressssss long
'''
print("\n" + "="*50)
print("🚀 DÉMARRAGE T2I ÉTAPE 3 : JUGE FINAL LISTWISE (LoRA)")
print("="*50)

# 1. On utilise le tenseur final de l'étape précédente (Top 10) comme nouveau point de départ
sorted_idx_t2i_stage2 = sorted_idx_t2i_final.clone()
sorted_idx_t2i_stage3 = sorted_idx_t2i_stage2.clone()

# 2. ⚙️ CHARGEMENT DU MODÈLE ELITE (LoRA)

if os.path.exists(config.LORA_OUTPUT_DIR):
    print(f"⚙️ Injection des poids LoRA ELITE CoT pour l'évaluation Listwise...")
    if not (getattr(vlm.model, "_hf_peft_config_loaded", False) or isinstance(vlm.model, PeftModel)):
        old_no_split = getattr(vlm.model, "_no_split_modules", None)
        vlm.model._no_split_modules = None
        vlm.model = PeftModel.from_pretrained(vlm.model, config.LORA_OUTPUT_DIR)
        vlm.model._no_split_modules = old_no_split
    vlm.model.eval()
    print("✅ Modèle Listwise prêt !")
else:
    print("⚠️ Attention : Dossier LoRA introuvable, le modèle tournera en Zero-Shot.")

# 3. Initialisation du cache pour cette 3ème étape
CACHE_STAGE3_FILE = config.CACHE_T2I_FILE.replace('.json', '_stage3_listwise.json')
cache_stage3 = RerankingCache(CACHE_STAGE3_FILE)

temps_vlm_reel = 0.0
appels_vlm_reels = 0

for q_idx in tqdm(range(limit_t2i), desc="Inférence T2I Stage 3 (Juge Final)"):
    # On ne prend QUE le Top 5 de la phase précédente (qui a déjà été pré-trié par le Pointwise)
    top_5 = sorted_idx_t2i_stage2[q_idx, :5].tolist()
    les_suivants = sorted_idx_t2i_stage2[q_idx, 5:].tolist()
    
    cached = cache_stage3.get(q_idx)
    if cached:
        new_top_5 = cached
    else:
        requete_texte = all_texts[q_idx]
        images_pil = [dataset[idx]['image'] for idx in top_5]
        
        # Mapping des positions (1 à 5) vers les vrais IDs d'images
        pos_to_id = {i+1: top_5[i] for i in range(5)}
        
        # 📝 PROMPT LISTWISE (Adapté pour comparer 5 images)
        prompt = (
            f"You are an expert visual verifier. These 5 images are pre-ranked by an AI for the query: '{requete_texte}'. "
            f"Image 1 is mathematically the most probable match. Your task is to verify this. "
            f"Compare the images directly. Do NOT demote Image 1 unless another image clearly matches the subtle details better. "
            f"Think step by step, penalize visual hallucinations, and output the Final Ranking."
            f"\nFormat:\nFinal Ranking: [PositionA, PositionB, PositionC, PositionD, PositionE]"
        )
        
        t0_call = time.time()
        response_text = vlm.generate_response(prompt, images_pil) 
        t1_call = time.time()
        
        temps_vlm_reel += (t1_call - t0_call)
        appels_vlm_reels += 1
        
        # --- PARSING ---
        match = re.search(r'Final Ranking:\s*(?:\[)?(.*?)(?:\])?$', response_text, re.IGNORECASE | re.MULTILINE)
        predicted_positions = None
        if match:
            nums = [int(n) for n in re.findall(r'\d+', match.group(1))]
            predicted_positions = []
            for x in nums:
                if 1 <= x <= 5 and x not in predicted_positions:
                    predicted_positions.append(x)
            missing = [i for i in range(1, 6) if i not in predicted_positions]
            predicted_positions.extend(missing)
        
        # Fallback de sécurité : Si le parsing échoue, on garde l'ordre de l'étape 2
        if predicted_positions is None or len(predicted_positions) != 5:
            new_top_5 = top_5
        else:
            # Reconversion des positions (1, 2, 3...) en IDs réels
            new_top_5 = [pos_to_id[pos] for pos in predicted_positions]
            
        # Mise en cache
        cache_stage3.set(q_idx, new_top_5)
        if appels_vlm_reels % 5 == 0:
            cache_stage3.save()
            
    # Mise à jour du tenseur avec le nouvel ordre
    ordre_final = new_top_5 + les_suivants
    sorted_idx_t2i_stage3[q_idx, :len(ordre_final)] = torch.tensor(ordre_final, device=device)

cache_stage3.save()

temps_total_vlm_t2i = temps_vlm_reel if appels_vlm_reels > 0 else 0.0

# =====================================================================
# ÉVALUATION GLOBALE (Stage 1 vs Stage 3)
# =====================================================================
print("\n" + "="*80)
print("🏆 COMPARAISON FINALE : MIRAGE (Base) vs PIPELINE 3 ÉTAGES (Coarse-to-Fine)")
print("="*80)

# On compare l'état initial (MIRAGE pur) avec l'état final (Après Pointwise + Listwise)
evaluate_and_save_results(
    sorted_idx_t2i_base[:limit_t2i], 
    sorted_idx_t2i_stage3[:limit_t2i], 
    targets_t2i_gpu[:limit_t2i], 
    is_i2t=False, 
    csv_path=config.metriques_t2i_csv.replace('.csv', '_stage3_final.csv'), 
    md_path=config.metriques_t2i_md.replace('.md', '_stage3_final.md'),
    temps_vlm=temps_total_vlm_t2i
)

# Autopsie globale : on veut voir combien d'erreurs de MIRAGE on a corrigées au total
_ = compute_autopsy(sorted_idx_t2i_base, sorted_idx_t2i_stage3, targets_t2i_gpu[:limit_t2i], is_i2t=False, mask_vlm_called=[True]*limit_t2i)
'''

# %% [markdown]
# ### Pointwise CoT (Avec Cascade & Late Fusion α)

# %%
'''
# ==========================================
# FONCTION DE PARSING POUR LE CoT (Score sur 100)
# ==========================================
def parse_cot_score(response_text):
    """Extrait la note finale (Score: X) générée par le VLM."""
    match = re.search(r"Score\s*:\s*(\d+)", response_text, re.IGNORECASE)
    if match:
        return float(match.group(1)) / 100.0 
    return 0.0 

print("\n" + "="*50)
print("🚀 DÉMARRAGE T2I (POINTWISE CoT + CASCADE LATE FUSION)")
print("="*50)

TOP_K_TEST = 10

if hasattr(vlm.model, "disable_adapters") and getattr(vlm.model, "_hf_peft_config_loaded", False):
    print("⚙️ Désactivation des adaptateurs LoRA...")
    vlm.model.disable_adapters()

# 1. Calibrage du seuil de confiance
print("🎯 Calibrage du seuil de confiance...")
seuil_t2i = calibrate_confidence_threshold(S_t2i_val_stage1, targets_t2i_val_gpu, is_i2t=False, target_recall=0.85)

CACHE_TOP10_COT_FILE = config.CACHE_T2I_FILE.replace('.json', '_top10_cot_cascade.json')
cache_t2i_cot = RerankingCache(CACHE_TOP10_COT_FILE)

sorted_idx_t2i_base = torch.argsort(S_t2i_stage1, dim=1, descending=True)
sorted_idx_t2i_final = sorted_idx_t2i_base.clone()

temps_vlm_reel = 0.0
appels_vlm_reels = 0
mask_vlm_called = [False] * limit_t2i
requetes_sauvees = 0

for q_idx in tqdm(range(limit_t2i), desc="Inférence T2I Pointwise CoT + Cascade"):
    top_10 = sorted_idx_t2i_base[q_idx, :TOP_K_TEST].tolist()
    
    # --- CASCADE RERANKING ---
    score_top1 = S_t2i_stage1[q_idx, top_10[0]].item()
    score_top2 = S_t2i_stage1[q_idx, top_10[1]].item()
    ecart_confiance = score_top1 - score_top2
    
    if ecart_confiance > seuil_t2i:
        new_top_10 = top_10
        requetes_sauvees += 1
    else:
        mask_vlm_called[q_idx] = True
        cached = cache_t2i_cot.get(q_idx)
        if cached:
            new_top_10 = cached
        else:
            requete = all_texts[q_idx]
            
            # PROMPT AVEC CHAIN OF THOUGHT
            prompt = f"""Analyze this image step-by-step against the description: '{requete}'. 
            Check every object, action, and attribute mentioned. List any missing or conflicting details. 
            Finally, rate the match on a scale from 0 to 100 on a new line exactly like this: 'Score: X'."""
            
            scores_vlm = []
            
            # Évaluation POINTWISE (1 image à la fois = très peu de VRAM)
            for idx in top_10:
                img = dataset[idx]['image']
                
                t0_call = time.time()
                response_text = vlm.generate_response(prompt, [img])
                t1_call = time.time()
                
                temps_vlm_reel += (t1_call - t0_call)
                appels_vlm_reels += 1
                
                score = parse_cot_score(response_text)
                scores_vlm.append(score)
            
            # --- LATE FUSION DOUCE ---
            # Au lieu d'éliminer cash (tournoi), on combine les certitudes
            scores_mirage_bruts = [S_t2i_stage1[q_idx, idx].item() for idx in top_10]
            max_s = max(scores_mirage_bruts)
            min_s = min(scores_mirage_bruts)
            scores_mirage_norm = [(s - min_s) / (max_s - min_s) if max_s > min_s else 1.0 for s in scores_mirage_bruts]

            # Alpha fixe (ou dynamique) : on donne 50% de poids à MIRAGE et 50% au VLM
            ALPHA = 0.5 
            final_scores = []
            for i, idx in enumerate(top_10):
                score_mixte = ((1 - ALPHA) * scores_mirage_norm[i]) + (ALPHA * scores_vlm[i])
                final_scores.append((score_mixte, idx))
                
            final_scores.sort(key=lambda x: x[0], reverse=True)
            new_top_10 = [idx for score, idx in final_scores]
            
            cache_t2i_cot.set(q_idx, new_top_10)
            if appels_vlm_reels % 30 == 0:
                cache_t2i_cot.save()
            
    sorted_idx_t2i_final[q_idx, :TOP_K_TEST] = torch.tensor(new_top_10, device=device)

cache_t2i_cot.save()

print(f"\n✅ Économie Cascade : {requetes_sauvees}/{limit_t2i} requêtes sans appel VLM !")

# --- Bilan ---
temps_total_vlm_t2i = temps_vlm_reel if appels_vlm_reels > 0 else 0.0

evaluate_and_save_results(
    sorted_idx_t2i_base[:limit_t2i], 
    sorted_idx_t2i_final[:limit_t2i], 
    targets_t2i_gpu[:limit_t2i], 
    is_i2t=False, 
    csv_path=config.metriques_t2i_csv.replace('.csv', '_cot_cascade.csv'), 
    md_path=config.metriques_t2i_md.replace('.md', '_cot_cascade.md'),
    temps_vlm=temps_total_vlm_t2i
)

_ = compute_autopsy(sorted_idx_t2i_base, sorted_idx_t2i_final, targets_t2i_gpu[:limit_t2i], is_i2t=False, mask_vlm_called=mask_vlm_called)
'''

# %% [markdown]
# ### Tournoi Pairwise (Avec Cascade)
# 

# %%
'''
# ==========================================
# FONCTION DE PARSING POUR LE TOURNOI
# ==========================================
def parse_pairwise_winner(response_text):
    """Extrait le choix du VLM (Image A ou Image B)."""
    # On cherche une réponse claire 'A' ou 'B'
    match = re.search(r'\b(A|B)\b', response_text.upper())
    if match:
        return match.group(1)
    # En cas d'hallucination ou de refus, on garde le champion en titre (A par défaut)
    return 'A'

print("\n" + "="*50)
print("🚀 DÉMARRAGE T2I (PAIRWISE TOURNAMENT + CASCADE CONFIDENCE - TOP 10)")
print("="*50)

TOP_K_TEST = 10

if hasattr(vlm.model, "disable_adapters") and getattr(vlm.model, "_hf_peft_config_loaded", False):
    print("⚙️ Désactivation des adaptateurs LoRA pour test Pairwise Zero-Shot...")
    vlm.model.disable_adapters()

# 1. Calibrage du seuil sur le set de validation (On cible 85% des erreurs)
print("🎯 Calibrage du seuil de confiance...")
seuil_t2i = calibrate_confidence_threshold(S_t2i_val_stage1, targets_t2i_val_gpu, is_i2t=False, target_recall=0.85)

# Nouveau cache pour l'approche Tournoi
CACHE_TOP10_PAIR_FILE = config.CACHE_T2I_FILE.replace('.json', '_top10_pairwise.json')
cache_t2i_pair = RerankingCache(CACHE_TOP10_PAIR_FILE)

sorted_idx_t2i_base = torch.argsort(S_t2i_stage1, dim=1, descending=True)
sorted_idx_t2i_final = sorted_idx_t2i_base.clone()

temps_vlm_reel = 0.0
appels_vlm_reels = 0

# Par défaut, on ne fait pas appel au VLM (on mettra True que si on a un doute)
mask_vlm_called = [False] * limit_t2i
requetes_sauvees = 0

for q_idx in tqdm(range(limit_t2i), desc="Inférence T2I Pairwise (Tournoi)"):
    top_10 = sorted_idx_t2i_base[q_idx, :TOP_K_TEST].tolist()
    
    # --- CASCADE RERANKING : Vérification de la confiance ---
    score_top1 = S_t2i_stage1[q_idx, top_10[0]].item()
    score_top2 = S_t2i_stage1[q_idx, top_10[1]].item()
    ecart_confiance = score_top1 - score_top2
    
    # Si l'écart est supérieur au seuil, on fait aveuglément confiance à la Phase 1
    if ecart_confiance > seuil_t2i:
        new_top_10 = top_10
        requetes_sauvees += 1
    else:
        # La Phase 1 hésite, on lance le VLM
        mask_vlm_called[q_idx] = True
        
        # Vérification du cache
        cached = cache_t2i_pair.get(q_idx)
        if cached:
            new_top_10 = cached
        else:
            requete = all_texts[q_idx]
            
            # Le champion initial est le Top 1 de MIRAGE
            champion_idx = top_10[0]
            champion_img = dataset[champion_idx]['image']
            
            # Le tournoi commence (du Top 2 au Top 10)
            for i in range(1, TOP_K_TEST):
                challenger_idx = top_10[i]
                challenger_img = dataset[challenger_idx]['image']
                
                prompt = f"""You are an expert evaluator. I am looking for the image that BEST matches this description: '{requete}'
                
                Image A is the first image. Image B is the second image.
                Compare both images carefully against every detail of the text. 
                Which image is a better match? Answer STRICTLY with a single letter: 'A' or 'B'."""
                
                t0_call = time.time()
                
                # On envoie les 2 images au VLM (A = champion, B = challenger)
                response_text = vlm.generate_response(prompt, [champion_img, challenger_img])
                
                t1_call = time.time()
                temps_vlm_reel += (t1_call - t0_call)
                appels_vlm_reels += 1
                
                winner = parse_pairwise_winner(response_text)
                
                # Si le challenger (B) gagne, il devient le nouveau champion
                if winner == 'B':
                    champion_idx = challenger_idx
                    champion_img = challenger_img
            
            # --- RECONSTRUCTION DU CLASSEMENT ---
            # Le champion prend la 1ère place. 
            # Les autres gardent leur ordre relatif dicté par MIRAGE (Phase 1).
            new_top_10 = [champion_idx]
            for idx in top_10:
                if idx != champion_idx:
                    new_top_10.append(idx)
                    
            # Mise en cache
            cache_t2i_pair.set(q_idx, new_top_10)
            if appels_vlm_reels % 45 == 0: # Sauvegarde régulière (toutes les ~5 requêtes complètes)
                cache_t2i_pair.save()
            
    # Mise à jour du tenseur final
    sorted_idx_t2i_final[q_idx, :TOP_K_TEST] = torch.tensor(new_top_10, device=device)

cache_t2i_pair.save()

print(f"\n✅ Économie Cascade : {requetes_sauvees}/{limit_t2i} requêtes ont été traitées sans appeler le VLM !")

# --- Bilan T2I ---
temps_total_vlm_t2i = temps_vlm_reel if appels_vlm_reels > 0 else 0.0

evaluate_and_save_results(
    sorted_idx_t2i_base[:limit_t2i], 
    sorted_idx_t2i_final[:limit_t2i], 
    targets_t2i_gpu[:limit_t2i], 
    is_i2t=False, 
    csv_path=config.metriques_t2i_csv.replace('.csv', '_top10_pairwise_cascade.csv'), 
    md_path=config.metriques_t2i_md.replace('.md', '_top10_pairwise_cascade.md'),
    temps_vlm=temps_total_vlm_t2i
)

_ = compute_autopsy(sorted_idx_t2i_base, sorted_idx_t2i_final, targets_t2i_gpu[:limit_t2i], is_i2t=False, mask_vlm_called=mask_vlm_called)
'''

# %% [markdown]
# ### Pointwise Top 10 (Sans cascade, Late Fusion α)
# 

# %%
'''
# %% [markdown]
# ## Reranking T2I (FULL POINTWISE SOTA - Top 10)

# %%
print("\n" + "="*50)
print("🚀 DÉMARRAGE T2I (FULL POINTWISE SOTA - TOP 10)")
print("="*50)

# 2. Désactivation du LoRA
if hasattr(vlm.model, "disable_adapters") and getattr(vlm.model, "_hf_peft_config_loaded", False):
    print("⚙️ Désactivation des adaptateurs LoRA pour test Pointwise Zero-Shot...")
    vlm.model.disable_adapters()
else:
    print("✅ Le modèle est bien en mode de base (Zero-Shot).")

# 3. Initialisation avec un nouveau cache spécifique au Top 10
CACHE_TOP10_FILE = config.CACHE_T2I_FILE.replace('.json', '_top10.json')
cache_t2i = RerankingCache(CACHE_TOP10_FILE)

sorted_idx_t2i_base = torch.argsort(S_t2i_stage1, dim=1, descending=True)
sorted_idx_t2i_final = sorted_idx_t2i_base.clone()

temps_vlm_reel = 0.0
appels_vlm_reels = 0
ALPHA = 0.5 

mask_vlm_called = [True] * limit_t2i

for q_idx in tqdm(range(limit_t2i), desc="Inférence T2I Full Pointwise (Top 10)"):
    top_10 = sorted_idx_t2i_base[q_idx, :TOP_K_T2I].tolist()
    
    # Vérification du cache
    cached = cache_t2i.get(q_idx)
    if cached:
        new_top_10 = cached
    else:
        images_pil = [dataset[idx]['image'] for idx in top_10]
        requete = all_texts[q_idx]
        
        prompt = f"Does the following image exactly and perfectly match this description: '{requete}'? Look at every tiny detail. Answer strictly with 'Yes' or 'No'."
        
        t0_call = time.time()
        
        # 🛡️ SÉCURITÉ VRAM : On coupe le batch de 10 en deux batchs de 5
        scores_vlm_part1 = vlm.score_image_pointwise_batch([prompt] * 5, images_pil[:5])
        scores_vlm_part2 = vlm.score_image_pointwise_batch([prompt] * 5, images_pil[5:])
        scores_vlm = scores_vlm_part1 + scores_vlm_part2 # On recolle les 10 scores
        
        t1_call = time.time()

        temps_vlm_reel += (t1_call - t0_call)
        appels_vlm_reels += 1 

        # --- LATE FUSION ---
        # 1. On récupère les 10 scores bruts de MIRAGE
        scores_mirage_bruts = [S_t2i_stage1[q_idx, idx].item() for idx in top_10]
        
        # 2. Normalisation Min-Max
        max_s = max(scores_mirage_bruts)
        min_s = min(scores_mirage_bruts)
        if max_s > min_s:
            scores_mirage_norm = [(s - min_s) / (max_s - min_s) for s in scores_mirage_bruts]
        else:
            scores_mirage_norm = [1.0] * 10

        # 3. Fusion mathématique (Alpha * VLM + (1-Alpha) * MIRAGE)
        final_scores = []
        for i, idx in enumerate(top_10):
            score_mixte = ((1 - ALPHA) * scores_mirage_norm[i]) + (ALPHA * scores_vlm[i])
            final_scores.append((score_mixte, idx))
            
        # 4. Nouveau tri basé sur le score fusionné
        final_scores.sort(key=lambda x: x[0], reverse=True)
        new_top_10 = [idx for score, idx in final_scores]
        
        # Mise en cache
        cache_t2i.set(q_idx, new_top_10)
        if appels_vlm_reels % 5 == 0: 
            cache_t2i.save()
            
    # Mise à jour du tenseur final
    ordre_final = new_top_10 + top_10[TOP_K_T2I:] if len(top_10) > TOP_K_T2I else new_top_10 + sorted_idx_t2i_base[q_idx, TOP_K_T2I:].tolist()
    sorted_idx_t2i_final[q_idx, :TOP_K_T2I] = torch.tensor(new_top_10, device=device)

cache_t2i.save()

# --- Calcul du temps projeté ---
temps_total_vlm_t2i = temps_vlm_reel if appels_vlm_reels > 0 else 0.0

# --- Bilan T2I ---
evaluate_and_save_results(
    sorted_idx_t2i_base[:limit_t2i], 
    sorted_idx_t2i_final[:limit_t2i], 
    targets_t2i_gpu[:limit_t2i], 
    is_i2t=False, 
    csv_path=config.metriques_t2i_csv.replace('.csv', '_top10.csv'), 
    md_path=config.metriques_t2i_md.replace('.md', '_top10.md'),
    temps_vlm=temps_total_vlm_t2i
)

# --- Autopsie T2I ---
_ = compute_autopsy(sorted_idx_t2i_base, sorted_idx_t2i_final, targets_t2i_gpu[:limit_t2i], is_i2t=False, mask_vlm_called=mask_vlm_called)
'''

# %%

passes_extraction = [
    {
        "nom": "Validation",
        "limite": len(targets_t2i_val_gpu),
        "S_base": S_t2i_val_stage1,
        "dataset": val_dataset,
        "textes": val_texts,
        "fichier_out": os.path.join(config.RESULTS_DIR, "scores_bruts_t2i_val.json")
    },
    {
        "nom": "Test",
        "limite": limit_t2i,
        "S_base": S_t2i_stage1,
        "dataset": dataset,
        "textes": all_texts,
        "fichier_out": os.path.join(config.RESULTS_DIR, "scores_bruts_t2i_test.json")
    }
]

# 3. BOUCLE D'EXTRACTION EXHAUSTIVE
for passe in passes_extraction:
    print(f"\n" + "="*50)
    print(f"🚀 DÉMARRAGE EXTRACTION EXHAUSTIVE : Set de {passe['nom']}")
    print("="*50)
    
    fichier_out = passe['fichier_out']
    S_base = passe['S_base']
    limite = passe['limite']
    data_images = passe['dataset']
    data_textes = passe['textes']
    
    # Recalcul du Top 10 de base pour le set en cours
    sorted_idx_base = torch.argsort(S_base, dim=1, descending=True)
    scores_all_queries = {}

    # Chargement du cache existant
    if os.path.exists(fichier_out):
        with open(fichier_out, 'r') as f:
            scores_all_queries = json.load(f)

    for q_idx in tqdm(range(limite), desc=f"Extraction Pointwise ({passe['nom']})"):
        str_qidx = str(q_idx)
        top_10 = sorted_idx_base[q_idx, :10].tolist()
        
        if str_qidx in scores_all_queries:
            continue

        # 1. Récupération des images et de la requête
        images_pil = [data_images[idx]['image'] for idx in top_10]
        requete = data_textes[q_idx]
        prompt = f"Does the following image exactly and perfectly match this description: '{requete}'? Answer strictly with 'Yes' or 'No'."
        
        # 2. Inférence VLM (Scoring Pointwise par Batch)
        scores_vlm_p1 = vlm.score_image_pointwise_batch([prompt] * 5, images_pil[:5])
        scores_vlm_p2 = vlm.score_image_pointwise_batch([prompt] * 5, images_pil[5:])
        scores_vlm = scores_vlm_p1 + scores_vlm_p2
        
        # 3. Récupération des scores MIRAGE bruts
        scores_mirage = [S_base[q_idx, idx].item() for idx in top_10]
        
        # 4. Sauvegarde des données brutes
        scores_all_queries[str_qidx] = {
            "candidate_ids": top_10,
            "vlm_scores": scores_vlm,     # Probabilités [0, 1]
            "mirage_scores": scores_mirage # Scores de similarité phase 1
        }
        

        with open(fichier_out, 'w') as f:
            json.dump(scores_all_queries, f)

    # Sauvegarde finale pour ce set
    with open(fichier_out, 'w') as f:
        json.dump(scores_all_queries, f)

    print(f"✅ Extraction totale terminée et sauvegardée dans : {fichier_out}")

# %%
def evaluate_fusion(scores_dict, targets_gpu, method="additive", alpha=0.5, rrf_k=60, top_k=10, cascade_threshold=None):
    """Calcule le R@1 en simulant dynamiquement le Top-K et la Cascade."""
    correct_count = 0
    total = len(scores_dict)
    reranked_results = {}
    mask_vlm_called = {} # Pour garder la trace des requêtes sauvées par la cascade
    
    for q_idx_str, data in scores_dict.items():
        q_idx = int(q_idx_str)
        target = targets_gpu[q_idx].item()
        
        ids = data["candidate_ids"][:top_k]
        vlm = np.array(data["vlm_scores"][:top_k])
        mirage = np.array(data["mirage_scores"][:top_k])
        
        # 🛡️ --- LOGIQUE DE CASCADE --- 🛡️
        ecart_confiance = mirage[0] - mirage[1] if len(mirage) > 1 else 0
        
        if cascade_threshold is not None and ecart_confiance > cascade_threshold:
            # MIRAGE est sûr de lui : on valide sans utiliser le VLM
            new_top_k = ids
            mask_vlm_called[q_idx] = False
        else:
            # MIRAGE hésite : on fait la fusion
            mask_vlm_called[q_idx] = True
            mirage_norm = (mirage - mirage.min()) / (mirage.max() - mirage.min() + 1e-8)
            
            if method == "additive":
                final_scores = (alpha * vlm) + ((1 - alpha) * mirage_norm)
            elif method == "multiplicative":
                final_scores = vlm * mirage_norm
            elif method == "maximum":
                final_scores = np.maximum(vlm, mirage_norm)
            elif method == "concatenation":
                final_scores = (vlm * 1000.0) + mirage_norm
            elif method == "rrf":
                rank_vlm = np.argsort(np.argsort(-vlm)) + 1
                rank_mirage = np.argsort(np.argsort(-mirage_norm)) + 1
                final_scores = (1.0 / (rrf_k + rank_vlm)) + (1.0 / (rrf_k + rank_mirage))
            elif method == "vlm_seul":
                final_scores = vlm
            elif method == "harmonique":
                final_scores = 2 * (vlm * mirage_norm) / (vlm + mirage_norm + 1e-8)
            elif method == "geometrique":
                final_scores = np.sqrt(vlm * mirage_norm)
            elif method == "rank_penalty":
                rank_mirage = np.argsort(np.argsort(-mirage))
                final_scores = vlm - (0.05 * rank_mirage)
            elif method == "tie_breaker_mirage":
                final_scores = vlm + (0.05 * mirage_norm)
            elif method == "safe_vlm":
                rank_mirage = np.argsort(np.argsort(-mirage))
                final_scores = vlm - (0.02 * rank_mirage)
                
            best_indices = np.argsort(-final_scores) 
            new_top_k = [ids[i] for i in best_indices]
            
        reranked_results[q_idx] = new_top_k
        if new_top_k[0] == target:
            correct_count += 1
            
    return correct_count / total, reranked_results, mask_vlm_called

# ==========================================
# 0. PRÉPARATION DES DONNÉES (VAL ET TEST)
# ==========================================
RAW_SCORES_VAL_FILE = os.path.join(config.RESULTS_DIR, "scores_bruts_t2i_val.json")
RAW_SCORES_TEST_FILE = os.path.join(config.RESULTS_DIR, "scores_bruts_t2i_test.json")

with open(RAW_SCORES_VAL_FILE, 'r') as f:
    scores_bruts_val = json.load(f)
with open(RAW_SCORES_TEST_FILE, 'r') as f:
    scores_bruts_test = json.load(f)

# On recalcule les classements de base pour les deux sets
sorted_idx_t2i_val_base = torch.argsort(S_t2i_val_stage1, dim=1, descending=True)
sorted_idx_t2i_test_base = torch.argsort(S_t2i_stage1, dim=1, descending=True)

# ==========================================
# 1. PHASE 1 : GRID SEARCH SUR VALIDATION
# ==========================================
print("\n" + "="*70)
print("🧪 PHASE 1 : RECHERCHE DES HYPERPARAMÈTRES (SET DE VALIDATION)")
print("="*70)

options_top_k = [5, 10]
methodes_simples = ["vlm_seul", "multiplicative", "maximum", "concatenation", "rrf", 
                    "harmonique", "geometrique", "rank_penalty", "tie_breaker_mirage", "safe_vlm"]
print("🎯 Calibrage des seuils de cascade sur le set de validation...")
options_cascade = {"Sans Cascade": None}
recalls_a_tester = [0.80, 0.90, 0.95, 0.98, 0.99, 0.995, 1.00]

for tr in recalls_a_tester:
    seuil = calibrate_confidence_threshold(S_t2i_val_stage1, targets_t2i_val_gpu, is_i2t=False, target_recall=tr)
    options_cascade[f"Avec (Recall={tr:.2f})"] = seuil

resultats_val = []
grand_gagnant_val = {"r1": 0, "nom": "", "params": {}}

# Exécution de toutes les combinaisons sur le VAL
for top_k in options_top_k:
    for nom_cascade, val_cascade in options_cascade.items():
        
        # 1. Méthodes simples
        for methode in methodes_simples:
            r1, _, mask = evaluate_fusion(scores_bruts_val, targets_t2i_val_gpu, method=methode, top_k=top_k, cascade_threshold=val_cascade)
            req_sauvees = sum(not v for v in mask.values())
            
            resultats_val.append({
                "Profondeur": f"Top {top_k}",
                "Cascade": nom_cascade.split(" ")[0],
                "Stratégie de Fusion": methode.capitalize(),
                "R@1 Val (%)": r1 * 100,
                "VLM Évités": req_sauvees
            })
            
            if r1 > grand_gagnant_val["r1"]:
                grand_gagnant_val = {"r1": r1, "nom": f"{methode.capitalize()}_Top{top_k}_{nom_cascade.split(' ')[0]}", 
                                     "params": {"method": methode, "alpha": 0.5, "top_k": top_k, "cascade_threshold": val_cascade}}

        # 2. Additif (On scanne tout l'espace de 0 à 1 pour être sûr de trouver le vrai optimum global)
        best_add_r1, best_alpha = 0, 0
        best_add_mask = {}
        for a in np.linspace(0, 1, 21):
            r1, _, mask = evaluate_fusion(scores_bruts_val, targets_t2i_val_gpu, method="additive", alpha=a, top_k=top_k, cascade_threshold=val_cascade)
            if r1 > best_add_r1:
                best_add_r1 = r1
                best_alpha = a
                best_add_mask = mask
                
        req_sauvees_add = sum(not v for v in best_add_mask.values())
        resultats_val.append({
            "Profondeur": f"Top {top_k}",
            "Cascade": nom_cascade.split(" ")[0],
            "Stratégie de Fusion": f"Grid Search (Alpha={best_alpha:.2f})",
            "R@1 Val (%)": best_add_r1 * 100,
            "VLM Évités": req_sauvees_add
        })
        
        if best_add_r1 > grand_gagnant_val["r1"]:
            grand_gagnant_val = {"r1": best_add_r1, "nom": f"Grid_Search_A{best_alpha:.2f}_Top{top_k}_{nom_cascade.split(' ')[0]}", 
                                 "params": {"method": "additive", "alpha": best_alpha, "top_k": top_k, "cascade_threshold": val_cascade}}

# Affichage du classement Validation
df_val = pd.DataFrame(resultats_val).sort_values(by=["R@1 Val (%)", "VLM Évités"], ascending=[False, False]).reset_index(drop=True)
display(df_val.head(10)) # On n'affiche que le Top 10 pour ne pas surcharger

print("\n" + "⭐ "*15)
print(f"LA MEILLEURE CONFIGURATION (VAL) EST : {grand_gagnant_val['nom']}")
print(f"R@1 sur Validation : {grand_gagnant_val['r1']*100:.2f} %")
print("⭐ "*15 + "\n")


# ==========================================
# 2. PHASE 2 : APPLICATION SUR LE TEST
# ==========================================
print("🔒 Hyperparamètres verrouillés. Déploiement sur le Set de Test...")
print("="*70)

# On extrait les paramètres parfaits
p = grand_gagnant_val["params"]

# 🎯 Un seul appel à la fonction sur le set de Test !
test_r1, dict_test, mask_test = evaluate_fusion(
    scores_bruts_test, 
    targets_t2i_gpu, 
    method=p["method"], 
    alpha=p["alpha"], 
    top_k=p["top_k"], 
    cascade_threshold=p["cascade_threshold"]
)

print(f"\n🏆 PERFORMANCE FINALE RÉELLE (R@1 TEST) : {test_r1*100:.2f} % 🏆")
print(f"⚡ VLM économisés sur le Test : {sum(not v for v in mask_test.values())} requêtes\n")

# ==========================================
# 3. AUTOPSIE ET SAUVEGARDE FINALE
# ==========================================
sorted_idx_t2i_final = sorted_idx_t2i_test_base.clone()

# Reconstruction
for q_idx_str, new_top_k in dict_test.items():
    q_idx = int(q_idx_str)
    ordre_final = new_top_k + sorted_idx_t2i_test_base[q_idx, len(new_top_k):].tolist()
    sorted_idx_t2i_final[q_idx, :len(ordre_final)] = torch.tensor(ordre_final, device=device)

# Masque pour autopsie
mask_vlm_called_list = [False] * limit_t2i
for q_idx_str, was_called in mask_test.items():
    mask_vlm_called_list[int(q_idx_str)] = was_called

file_suffix = f"_{grand_gagnant_val['nom'].replace(' ', '_').replace('.', '').lower()}_strict_test"

evaluate_and_save_results(
    sorted_idx_t2i_test_base[:limit_t2i], 
    sorted_idx_t2i_final[:limit_t2i], 
    targets_t2i_gpu[:limit_t2i], 
    is_i2t=False, 
    csv_path=config.metriques_t2i_csv.replace('.csv', f'{file_suffix}.csv'), 
    md_path=config.metriques_t2i_md.replace('.md', f'{file_suffix}.md'),
    temps_vlm=0.0
)

_ = compute_autopsy(
    sorted_idx_t2i_test_base, 
    sorted_idx_t2i_final, 
    targets_t2i_gpu[:limit_t2i], 
    is_i2t=False, 
    mask_vlm_called=mask_vlm_called_list
)

# %%
# ==========================================
# 📊 ANALYSE DÉTAILLÉE DU RECALL (R@1 à R@5)
# ==========================================
print("\n" + "="*50)
print("📊 DISTRIBUTION DES PERFORMANCES (R@1 à R@5) T2I")
print("="*50)

recalls_t2i = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
total_queries_t2i = len(dict_test)

for q_idx_str, new_top_k in dict_test.items():
    q_idx = int(q_idx_str)
    
    # Récupération de la target (T2I = une seule image cible)
    target = targets_t2i_gpu[q_idx].item()
        
    # Calcul des Hits pour chaque profondeur (de 1 à 5)
    for k in range(1, 6):
        if target in new_top_k[:k]:
            recalls_t2i[k] += 1

# Affichage des résultats
for k in range(1, 6):
    score = (recalls_t2i[k] / total_queries_t2i) * 100
    print(f"R@{k} : {score:.2f} %")
print("="*50 + "\n")

# %%
print("\n" + "="*70)
print("⚔️ PHASE 4 : DUEL 'SUDDEN DEATH' AVEC CoT (VLM SUR LE TOP 2)")
print("="*70)

# Fichier de cache pour sécuriser l'avancée
SUDDEN_DEATH_T2I_FILE = os.path.join(config.RESULTS_DIR, "scores_sudden_death_cot_t2i.json")

if os.path.exists(SUDDEN_DEATH_T2I_FILE):
    with open(SUDDEN_DEATH_T2I_FILE, 'r') as f:
        sudden_death_results = json.load(f)
else:
    sudden_death_results = {}

correct_count_sd = 0
total_sd = len(dict_test)
dict_test_final = {}

for raw_key, top_k_ids in tqdm(dict_test.items(), desc="Duel Pairwise CoT"):
    q_idx_str = str(raw_key) # 🛡️ On force en texte pour le JSON !
    q_idx = int(raw_key)
    target = targets_t2i_gpu[q_idx].item()
    
    id_A = top_k_ids[0]
    id_B = top_k_ids[1]
    
    # Si c'est déjà dans le cache, on passe directement à la suite
    if q_idx_str in sudden_death_results:
        continue
        
    # 1. Préparation du Duel
    requete_texte = all_texts[q_idx]
    image_A = dataset[id_A]['image']
    image_B = dataset[id_B]['image']
    
    # Le Prompt CoT
    prompt = f"""You are an expert visual detective. I will provide you with Image A and Image B.
    Your task is to determine which image perfectly matches this exact description: '{requete_texte}'

    Think step-by-step:
    1. Analyze Image A: Does it contain all elements of the description? Are there any contradictions?
    2. Analyze Image B: Does it contain all elements of the description? Are there any contradictions?
    3. Compare them: Which one is the undisputed perfect match?

    After your analysis, you MUST conclude your response with: "Winner: A" or "Winner: B"."""
    
    # 2. Formatage
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_A, "max_pixels": 262144},
        {"type": "image", "image": image_B, "max_pixels": 262144},
        {"type": "text", "text": prompt}
    ]}]
    
    text_prompt = vlm.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = vlm.processor(text=[text_prompt], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt").to(device)
    
    # 3. Inférence
    outputs = vlm.model.generate(**inputs, max_new_tokens=256)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)]
    reponse = vlm.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True)[0].strip()
    
    # 4. Parsing
    gagnant = "A" 
    import re
    match = re.search(r'Winner:\s*([AB])', reponse, re.IGNORECASE)
    if match:
        if match.group(1).upper() == "B":
            gagnant = "B"
    elif "image b" in reponse.lower().split(".")[-1]: 
        gagnant = "B"

    sudden_death_results[q_idx_str] = gagnant
    
    # Sauvegarde régulière
    if q_idx % 10 == 0:
        with open(SUDDEN_DEATH_T2I_FILE, 'w') as f:
            json.dump(sudden_death_results, f)

# On enregistre le cache final
with open(SUDDEN_DEATH_T2I_FILE, 'w') as f:
    json.dump(sudden_death_results, f)

# ==========================================
# RECONSTRUCTION ET ÉVALUATION FINALE (SUDDEN DEATH)
# ==========================================
print("\n" + "🎯 "*15)
print("ÉVALUATION COMPLÈTE DU DUEL CoT T2I")
print("🎯 "*15 + "\n")

sorted_idx_t2i_final_sd = sorted_idx_t2i_test_base.clone()
dict_test_final = {}
correct_count_sd = 0

# Reconstruction du dictionnaire final
for raw_key, top_k_ids in dict_test.items():
    q_idx_str = str(raw_key) # 🛡️ Pareil ici, on force en texte
    q_idx = int(raw_key)
    
    id_A = top_k_ids[0]
    id_B = top_k_ids[1]
    gagnant = sudden_death_results.get(q_idx_str, "A")
    
    nouveau_top_k = list(top_k_ids)
    if gagnant == "B":
        nouveau_top_k[0] = id_B
        nouveau_top_k[1] = id_A
        
    dict_test_final[q_idx_str] = nouveau_top_k
    
    # SÉCURITÉ MODE TEST
    if q_idx < limit_t2i:
        ordre_final = nouveau_top_k + sorted_idx_t2i_test_base[q_idx, len(nouveau_top_k):].tolist()
        sorted_idx_t2i_final_sd[q_idx, :len(ordre_final)] = torch.tensor(ordre_final, device=device)
        
        target = targets_t2i_gpu[q_idx].item()
        if nouveau_top_k[0] == target:
            correct_count_sd += 1

# Sauvegarde propre
file_suffix = "_sudden_death_cot_t2i"

evaluate_and_save_results(
    sorted_idx_t2i_test_base[:limit_t2i], 
    sorted_idx_t2i_final_sd[:limit_t2i], 
    targets_t2i_gpu[:limit_t2i], 
    is_i2t=False, 
    csv_path=config.metriques_t2i_csv.replace('.csv', f'{file_suffix}.csv'), 
    md_path=config.metriques_t2i_md.replace('.md', f'{file_suffix}.md'),
    temps_vlm=0.0
)

mask_vlm_sd = [True] * limit_t2i 

_ = compute_autopsy(
    sorted_idx_t2i_test_base, 
    sorted_idx_t2i_final_sd, 
    targets_t2i_gpu[:limit_t2i], 
    is_i2t=False, 
    mask_vlm_called=mask_vlm_sd
)


