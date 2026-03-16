import os
import json
import numpy as np
import torch
from datasets import load_from_disk
from collections import defaultdict
import config
from utils_indexation import SearchIndex

def load_reranking_data(split="test", device="cuda"):
    """
    Charge les données, les vecteurs et construit les matrices fusionnées pour un split donné.
    split: "test" ou "val"
    """
    print(f"\n📦 Chargement des données pour le split : [{split.upper()}]...")
    dataset = load_from_disk(f"{config.RAW_DATA_DIR}/{split}")

    with open(config.BEST_WEIGHTS_FILE, 'r') as f:
        best_weights = json.load(f)

    prefix = f"{split}_"
    
    # On détecte automatiquement les modèles présents dans le dossier index
    model_names = list(set(
        f[len(prefix):-13] for f in os.listdir(config.INDEX_DIR) 
        if f.startswith(prefix) and f.endswith("_img_vecs.npy")
    ))

    txt_vecs, img_vecs = {}, {}
    for name in model_names:
        img_vecs[name] = np.load(f"{config.INDEX_DIR}/{prefix}{name}_img_vecs.npy")
        txt_vecs[name] = np.load(f"{config.INDEX_DIR}/{prefix}{name}_txt_vecs.npy")

    # --- Reconstruction des correspondances (Targets) ---
    idx_img = SearchIndex(0)
    idx_img.load_from_disk(f"{config.INDEX_DIR}/{prefix}{model_names[0]}_img")
    img_ids = idx_img.image_ids

    idx_txt = SearchIndex(0)
    idx_txt.load_from_disk(f"{config.INDEX_DIR}/{prefix}{model_names[0]}_txt")
    txt_to_img_id = idx_txt.image_ids

    img_id_to_idx = {img_id: idx for idx, img_id in enumerate(img_ids)}

    # T2I Targets
    targets_t2i = np.array([img_id_to_idx[iid] for iid in txt_to_img_id])
    targets_t2i_gpu = torch.tensor(targets_t2i.reshape(-1, 1), device=device)

    # I2T Targets (Gestion des légendes multiples avec padding -1)
    img_to_txt_indices = defaultdict(list)
    for txt_idx, iid in enumerate(txt_to_img_id):
        img_to_txt_indices[img_id_to_idx[iid]].append(txt_idx)
    
    targets_i2t = [img_to_txt_indices[i] for i in range(len(img_id_to_idx))]
    max_len_i2t = max(len(t) for t in targets_i2t)
    targets_i2t_padded = [t + [-1] * (max_len_i2t - len(t)) for t in targets_i2t]
    targets_i2t_gpu = torch.tensor(targets_i2t_padded, device=device)

    # --- Fusion des matrices (Late Fusion) ---
    def get_fused_matrix(weights_dict):
        return sum(
            w * (torch.tensor(txt_vecs[name], device=device) @ torch.tensor(img_vecs[name], device=device).T)
            for name, w in weights_dict.items() if w > 0
        )

    print("🔗 Calcul des matrices de similarité fusionnées...")
    S_t2i_stage1 = get_fused_matrix(best_weights['t2i']['R@1'])
    S_i2t_stage1 = get_fused_matrix(best_weights['i2t']['R@1']).t()

    # Liste plate de toutes les légendes pour un accès direct par ID
    all_texts = [
        str(c) for item in dataset 
        for c in (item['caption'] if isinstance(item['caption'], list) else [item['caption']])
    ]

    print("✅ Données prêtes !")
    return dataset, all_texts, S_t2i_stage1, S_i2t_stage1, targets_t2i_gpu, targets_i2t_gpu