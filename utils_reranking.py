import os
import json
import re
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score

class RerankingCache:
    """Gestionnaire de cache pour éviter de rappeler le VLM inutilement."""
    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.data = self._load()

    def _load(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def save(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.data, f)

    def get(self, key):
        return self.data.get(str(key))

    def set(self, key, value):
        self.data[str(key)] = value

def parse_vlm_ranking(response_text, candidats_valides):
    """Extrait les IDs de la réponse texte du VLM avec fallback de sécurité."""
    match = re.search(r'Final Ranking:\s*(?:\[)?(.*?)(?:\])?$', response_text, re.IGNORECASE | re.MULTILINE)
    new_order = []
    
    if match:
        predicted_ids = [int(n) for n in re.findall(r'\d+', match.group(1))]
        for pid in predicted_ids:
            if pid in candidats_valides and pid not in new_order:
                new_order.append(pid)
                
    # Fallback : on complète avec les candidats manquants dans l'ordre initial
    for cid in candidats_valides:
        if cid not in new_order:
            new_order.append(cid)
            
    return new_order


def calibrate_confidence_threshold(S_val_stage1, targets_val, is_i2t=False, target_recall=0.85):
    """
    Trouve le seuil nécessaire pour attraper X% des erreurs du Stage 1 sur le set de validation.
    """
    sorted_idx_val = torch.argsort(S_val_stage1, dim=1, descending=True)
    limit_val = len(targets_val)
    
    ecarts_erreurs = []
    
    for q_idx in range(limit_val):
        top_1_idx = sorted_idx_val[q_idx, 0].item()
        top_2_idx = sorted_idx_val[q_idx, 1].item()
        
        # 1. Le modèle s'est-il trompé ?
        if is_i2t:
            target_ids = [int(idx) for idx in targets_val[q_idx].tolist() if idx != -1]
            is_correct = top_1_idx in target_ids
        else:
            target_id = int(targets_val[q_idx].item())
            is_correct = top_1_idx == target_id
            
        # 2. Si c'est une ERREUR, on enregistre son score d'hésitation (l'écart)
        if not is_correct:
            ecart = S_val_stage1[q_idx, top_1_idx].item() - S_val_stage1[q_idx, top_2_idx].item()
            ecarts_erreurs.append(ecart)

    if not ecarts_erreurs:
        return 0.0

    # 3. On trouve le seuil qui englobe 'target_recall' (ex: 85%) de ces erreurs.
    # Les erreurs ont généralement de petits écarts. En prenant le 85ème percentile, 
    # on s'assure que 85% des erreurs sont "sous" ce seuil.
    seuil = np.percentile(ecarts_erreurs, target_recall * 100)
    
    print(f"📊 Seuil calibré pour attraper {target_recall*100}% des erreurs de validation : {seuil:.5f}")
    return float(seuil)