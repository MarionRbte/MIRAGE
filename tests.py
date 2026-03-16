# %% [markdown]
# # Tests

# %%
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import config
from tqdm.auto import tqdm
from peft import PeftModel
import os
import time
import json
import re

# Import de tes modules personnalisés
from utils_data import load_reranking_data
from utils_reranking import RerankingCache, parse_vlm_ranking, calibrate_confidence_threshold
from utils_analysis import compute_autopsy, evaluate_and_save_results
from rerankers import Qwen2VLReranker

# ==========================================
# CONFIGURATION
# ==========================================
TOP_K_I2T = 5
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
val_dataset, _, S_t2i_val_stage1, S_i2t_val_stage1, targets_t2i_val_gpu, targets_i2t_val_gpu = load_reranking_data("val", device)

# 2. On charge le set de test pour l'évaluation finale
dataset, all_texts, S_t2i_stage1, S_i2t_stage1, targets_t2i_gpu, targets_i2t_gpu = load_reranking_data("test", device)

limit_i2t = NB_QUERIES_TEST if MODE_TEST else len(targets_i2t_gpu)
limit_t2i = NB_QUERIES_TEST if MODE_TEST else len(targets_t2i_gpu)

# %% [markdown]
# ### Enregistrement des scores et test des fusions

# %%
# Fichier de stockage des scores bruts
RAW_SCORES_FILE = os.path.join(config.RESULTS_DIR, "scores_bruts_t2i.json")

# On récupère le Top 10 de base
sorted_idx_t2i_base = torch.argsort(S_t2i_stage1, dim=1, descending=True)
scores_all_queries = {}

# Chargement du cache existant pour ne pas recalculer si vous avez déjà des résultats
if os.path.exists(RAW_SCORES_FILE):
    with open(RAW_SCORES_FILE, 'r') as f:
        scores_all_queries = json.load(f)

for q_idx in tqdm(range(limit_t2i), desc="Extraction des scores bruts"):
    str_qidx = str(q_idx)
    top_10 = sorted_idx_t2i_base[q_idx, :10].tolist()
    
    if str_qidx in scores_all_queries:
        continue

    # 1. Récupération des images et de la requête
    images_pil = [dataset[idx]['image'] for idx in top_10]
    requete = all_texts[q_idx]
    prompt = f"Does the following image exactly and perfectly match this description: '{requete}'? Answer strictly with 'Yes' or 'No'."
    
    # 2. Inférence VLM (Scoring Pointwise par Batch)
    # On coupe en deux batchs de 5 pour la VRAM
    scores_vlm_p1 = vlm.score_image_pointwise_batch([prompt] * 5, images_pil[:5])
    scores_vlm_p2 = vlm.score_image_pointwise_batch([prompt] * 5, images_pil[5:])
    scores_vlm = scores_vlm_p1 + scores_vlm_p2
    
    # 3. Récupération des scores MIRAGE bruts
    scores_mirage = [S_t2i_stage1[q_idx, idx].item() for idx in top_10]
    
    # Sauvegarde des données brutes
    scores_all_queries[str_qidx] = {
        "candidate_ids": top_10,
        "vlm_scores": scores_vlm,     # Probabilités [0, 1]
        "mirage_scores": scores_mirage # Scores de similarité phase 1
    }
    
    if q_idx % 10 == 0:
        with open(RAW_SCORES_FILE, 'w') as f:
            json.dump(scores_all_queries, f)

with open(RAW_SCORES_FILE, 'w') as f:
    json.dump(scores_all_queries, f)

print(f"✅ Scores bruts sauvegardés dans : {RAW_SCORES_FILE}")

# %%
# %% [markdown]
# ## ÉTAPE 2 : LABORATOIRE DE FUSION (TABLEAU COMPARATIF COMPLET)
# %%
import json
import numpy as np
import torch
import time
import pandas as pd
from tqdm.auto import tqdm
from IPython.display import display

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
                
            best_indices = np.argsort(-final_scores) 
            new_top_k = [ids[i] for i in best_indices]
            
        reranked_results[q_idx] = new_top_k
        if new_top_k[0] == target:
            correct_count += 1
            
    return correct_count / total, reranked_results, mask_vlm_called

# ==========================================
# 1. PRÉPARATION DES DONNÉES ET DES TESTS
# ==========================================
RAW_SCORES_FILE = os.path.join(config.RESULTS_DIR, "scores_bruts_t2i.json")

with open(RAW_SCORES_FILE, 'r') as f:
    scores_bruts = json.load(f)

# 🛠️ AJOUT ICI : On recalcule le classement de base de la Phase 1 et la limite
sorted_idx_t2i_base = torch.argsort(S_t2i_stage1, dim=1, descending=True)
limit_t2i = len(targets_t2i_gpu) 

print("\n" + "="*70)
print("🧪 LABORATOIRE DE FUSION : GÉNÉRATION DU TABLEAU COMPARATIF")
print("="*70)

options_top_k = [5, 10]
methodes_simples = ["multiplicative", "maximum", "concatenation", "rrf"]

# 🎯 NOUVEAUTÉ : Grid Search sur le Seuil de Cascade
print("🎯 Calibrage des multiples seuils de cascade sur le set de validation...")
options_cascade = {"Sans Cascade": None}

# On teste plusieurs niveaux d'exigence (de très permissif à très strict)
recalls_a_tester = [0.70, 0.80, 0.85, 0.90, 0.95, 0.99, 1.00]

for tr in recalls_a_tester:
    # Utilise S_t2i_val_stage1 et targets_t2i_val_gpu (déjà chargés dans ton notebook)
    seuil = calibrate_confidence_threshold(S_t2i_val_stage1, targets_t2i_val_gpu, is_i2t=False, target_recall=tr)
    
    # On ajoute ce seuil au dictionnaire des cascades à tester
    nom_cascade = f"Avec (Recall={tr:.2f})"
    options_cascade[nom_cascade] = seuil

resultats_tableau = []
grand_gagnant = {"r1": 0, "nom": "", "dict": {}, "mask": {}}

# ==========================================
# 2. EXÉCUTION DE TOUTES LES COMBINAISONS
# ==========================================
# On désactive l'affichage de tqdm pour que le tableau s'affiche proprement à la fin
for top_k in options_top_k:
    for nom_cascade, val_cascade in options_cascade.items():
        
        # 1. Tests des méthodes simples
        for methode in methodes_simples:
            r1, res_dict, mask = evaluate_fusion(scores_bruts, targets_t2i_gpu, method=methode, top_k=top_k, cascade_threshold=val_cascade)
            req_sauvees = sum(not v for v in mask.values())
            
            resultats_tableau.append({
                "Profondeur": f"Top {top_k}",
                "Cascade": nom_cascade.split(" ")[0], # Juste "Sans" ou "Avec"
                "Stratégie de Fusion": methode.capitalize(),
                "R@1 (%)": r1 * 100,
                "VLM Évités": req_sauvees
            })
            
            if r1 > grand_gagnant["r1"]:
                grand_gagnant = {"r1": r1, "nom": f"{methode.capitalize()}_Top{top_k}_{nom_cascade.split(' ')[0]}", "dict": res_dict, "mask": mask}

        # 2. Test Additif avec Grid Search Intégré
        best_add_r1, best_alpha, best_add_dict, best_add_mask = 0, 0, {}, {}
        for a in np.linspace(0, 1, 21):
            r1, res_dict, mask = evaluate_fusion(scores_bruts, targets_t2i_gpu, method="additive", alpha=a, top_k=top_k, cascade_threshold=val_cascade)
            if r1 > best_add_r1:
                best_add_r1 = r1
                best_alpha = a
                best_add_dict = res_dict
                best_add_mask = mask
                
        req_sauvees_add = sum(not v for v in best_add_mask.values())
        resultats_tableau.append({
            "Profondeur": f"Top {top_k}",
            "Cascade": nom_cascade.split(" ")[0],
            "Stratégie de Fusion": f"Grid Search (Alpha={best_alpha:.2f})",
            "R@1 (%)": best_add_r1 * 100,
            "VLM Évités": req_sauvees_add
        })
        
        if best_add_r1 > grand_gagnant["r1"]:
            grand_gagnant = {"r1": best_add_r1, "nom": f"Grid_Search_A{best_alpha:.2f}_Top{top_k}_{nom_cascade.split(' ')[0]}", "dict": best_add_dict, "mask": best_add_mask}

# ==========================================
# 3. AFFICHAGE DU CLASSEMENT
# ==========================================
# Création du DataFrame Pandas et tri par les meilleurs scores R@1
df_resultats = pd.DataFrame(resultats_tableau)
df_resultats = df_resultats.sort_values(by=["R@1 (%)", "VLM Évités"], ascending=[False, False]).reset_index(drop=True)

print("\n🏆 CLASSEMENT DES STRATÉGIES DE RERANKING :\n")
display(df_resultats) # Affiche un beau tableau HTML dans Jupyter

print("\n" + "⭐ "*15)
print(f"LE GRAND GAGNANT ABSOLU EST : {grand_gagnant['nom']}")
print(f"R@1 : {grand_gagnant['r1']*100:.2f} %")
print("⭐ "*15 + "\n")

# ==========================================
# 4. RECONSTRUCTION ET AUTOPSIE DU GAGNANT
# ==========================================
print(f"⚙️ Sauvegarde et Autopsie de la meilleure configuration ({grand_gagnant['nom']})...")

sorted_idx_t2i_final = sorted_idx_t2i_base.clone()
for q_idx_str, new_top_k in grand_gagnant["dict"].items():
    q_idx = int(q_idx_str)
    ordre_final = new_top_k + sorted_idx_t2i_base[q_idx, len(new_top_k):].tolist()
    sorted_idx_t2i_final[q_idx, :len(ordre_final)] = torch.tensor(ordre_final, device=device)

# Reconstruction de la liste mask_vlm_called pour l'autopsie
mask_vlm_called_list = [False] * limit_t2i
for q_idx_str, was_called in grand_gagnant["mask"].items():
    mask_vlm_called_list[int(q_idx_str)] = was_called

# Nommage propre du fichier CSV
file_suffix = f"_{grand_gagnant['nom'].replace(' ', '_').replace('.', '').lower()}"

evaluate_and_save_results(
    sorted_idx_t2i_base[:limit_t2i], 
    sorted_idx_t2i_final[:limit_t2i], 
    targets_t2i_gpu[:limit_t2i], 
    is_i2t=False, 
    csv_path=config.metriques_t2i_csv.replace('.csv', f'{file_suffix}.csv'), 
    md_path=config.metriques_t2i_md.replace('.md', f'{file_suffix}.md'),
    temps_vlm=0.0
)

_ = compute_autopsy(
    sorted_idx_t2i_base, 
    sorted_idx_t2i_final, 
    targets_t2i_gpu[:limit_t2i], 
    is_i2t=False, 
    mask_vlm_called=mask_vlm_called_list
)


