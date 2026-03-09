import torch
import numpy as np
import pandas as pd
import itertools
from tqdm.auto import tqdm

def evaluate_from_indices(sorted_indices_tensor: torch.Tensor, targets: torch.Tensor) -> dict:
    """Calcule les métriques pour des cibles uniques (T2I) ou multiples (I2T) avec padding -1."""
    
    # matches est une matrice booléenne [N_queries, N_gallery]
    matches = (sorted_indices_tensor.unsqueeze(-1) == targets.unsqueeze(1)).any(dim=-1)
    
    # Recall@K (Au moins 1 cible correcte dans le top K)
    r1  = float(matches[:, :1].any(dim=1).float().mean().item())
    r5  = float(matches[:, :5].any(dim=1).float().mean().item())
    r10 = float(matches[:, :10].any(dim=1).float().mean().item())
    
    # mAP (Mean Average Precision)
    cum_matches = matches.cumsum(dim=1)
    ranks = torch.arange(1, matches.shape[1] + 1, device=matches.device).float()
    precisions = cum_matches / ranks
    
    # Compte le nombre de cibles valides (ignore les -1 du padding)
    total_relevant = (targets != -1).sum(dim=1).clamp(min=1) 
    
    # Somme des précisions uniquement aux endroits où il y a un "match", puis on fait la moyenne
    ap = (precisions * matches).sum(dim=1) / total_relevant
    mAP = float(ap.mean().item())
    
    # NDCG
    discounts = 1.0 / torch.log2(ranks + 1.0)
    dcg = (matches.float() * discounts).sum(dim=1)
    
    # DCG Idéal (IDCG) : si toutes les cibles pertinentes étaient en haut du classement
    ideal_matches = torch.zeros_like(matches)
    for i, num_rel in enumerate(total_relevant):
        ideal_matches[i, :num_rel] = 1.0
    idcg = (ideal_matches * discounts).sum(dim=1)
    
    ndcg = float((dcg / idcg.clamp(min=1e-9)).mean().item())
    
    return {
        'R@1': r1,
        'R@5': r5,
        'R@10': r10,
        'mAP': mAP,
        'NDCG': ndcg,
    }

def evaluate_retrieval_gpu(similarity_matrix: torch.Tensor, targets: torch.Tensor) -> dict:
    """Trie une matrice de similarité puis calcule les métriques (Utilisé en Phase 2)."""
    # On trie les scores pour obtenir les indices
    sorted_indices = torch.argsort(similarity_matrix, dim=1, descending=True)
    # On appelle la fonction de base qui fait les calculs
    return evaluate_from_indices(sorted_indices, targets)

class GridSearchOptimizer:
    """Classe dédiée à la recherche des meilleurs poids de fusion (Late Fusion)."""
    def __init__(self, similarity_matrices: dict, targets_t2i: np.ndarray, targets_i2t: list, step: float = 0.1):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"GridSearch initialisé sur : {self.device}")

        self.model_names = list(similarity_matrices.keys())
        self.step = step

        # Transfert des matrices et cibles sur le GPU une seule fois pour accélérer
        self.matrices = {n: torch.tensor(mat, dtype=torch.float32, device=self.device) for n, mat in similarity_matrices.items()}
        self.targets_t2i = torch.tensor(np.array(targets_t2i).reshape(-1, 1), device=self.device)
        self.targets_i2t = torch.tensor(targets_i2t, device=self.device)

    def optimize(self, task="mean"):
        weights_range = np.arange(0, 1.0 + self.step / 2, self.step)
        all_combinations = list(itertools.product(weights_range, repeat=len(self.model_names)))
        tested_normalized = set()
        results_history = []

        print(f"\nOptimisation pour la tâche : {task.upper()}...")
        for weights in tqdm(all_combinations):
            total = sum(weights)
            if total == 0: continue
            
            # Normalisation des poids (ex: 0.1, 0.2 -> 0.33, 0.67)
            norm_w = tuple(np.round(w / total, 3) for w in weights)
            if norm_w in tested_normalized: continue
            tested_normalized.add(norm_w)

            # Fusion sur GPU
            S_fused = sum(w * self.matrices[n] for w, n in zip(norm_w, self.model_names))

            # Évaluation
            if task == "t2i": 
                metrics = evaluate_retrieval_gpu(S_fused, self.targets_t2i)
            else : 
                metrics = evaluate_retrieval_gpu(S_fused.t(), self.targets_i2t)

            # Enregistrement des résultats
            row = {self.model_names[i]: norm_w[i] for i in range(len(self.model_names))}
            row.update({k: round(v, 4) for k, v in metrics.items()})
            results_history.append(row)

        df = pd.DataFrame(results_history)
        
        # Résumé des meilleurs scores
        summary_data = []
        for m in ['R@1', 'R@5', 'R@10', 'mAP', 'NDCG']:
            best = df.loc[df[m].idxmax()]
            summary_data.append({
                'Métrique': m,
                'Meilleurs Poids': " + ".join(f"{best[n]*100:.1f}% {n}" for n in self.model_names),
                'Score Maximal': best[m],
            })

        print(f"\n--- Meilleures combinaisons ({task.upper()}) ---")
        print(pd.DataFrame(summary_data).to_string(index=False))
        return df