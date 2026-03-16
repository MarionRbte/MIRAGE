# %% [markdown]
# # Phase 3 : Reranking avec VLM

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
NB_QUERIES_TEST = 50

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
# ## Reranking I2T

# %%
print("\n" + "="*50)
print("🚀 DÉMARRAGE I2T (FULL LISTWISE ZERO-SHOT - SANS CASCADE)")
print("="*50)

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

# %% [markdown]
# ## Reranking T2I

# %% [markdown]
# ### Pointwise Top 10 (Sans cascade, Late Fusion α)
# 

# %%
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


