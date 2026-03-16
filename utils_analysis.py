import pandas as pd
import os

def compute_autopsy(sorted_idx_stage1, sorted_idx_final, targets_eval, is_i2t=False, limit=None, mask_vlm_called=None):
    """
    Calcule les sauvetages, sabotages, etc.
    - is_i2t: True si les cibles sont des listes, False si cible unique.
    - mask_vlm_called: Liste de booléens (optionnel) pour ne compter que les requêtes passées au VLM.
    """
    limit = limit or len(targets_eval)
    stats = {"total": 0, "sauvetages": 0, "sabotages": 0, "doubles_echecs": 0, "confirmations": 0}

    for q_idx in range(limit):
        if mask_vlm_called is not None and not mask_vlm_called[q_idx]:
            continue # On ignore si le VLM n'a pas été appelé (cas de la Cascade T2I)
            
        stats["total"] += 1
        
        # Récupération des cibles valides
        if is_i2t:
            target_ids = [idx for idx in targets_eval[q_idx].tolist() if idx != -1]
        else:
            target_ids = [int(targets_eval[q_idx].item())]

        top1_stage1 = int(sorted_idx_stage1[q_idx, 0].item())
        top1_final = int(sorted_idx_final[q_idx, 0].item())

        stage1_correct = top1_stage1 in target_ids
        final_correct = top1_final in target_ids

        if not stage1_correct and final_correct:
            stats["sauvetages"] += 1
        elif not stage1_correct and not final_correct:
            stats["doubles_echecs"] += 1
        elif stage1_correct and not final_correct:
            stats["sabotages"] += 1
        elif stage1_correct and final_correct:
            stats["confirmations"] += 1

    gain_net = stats["sauvetages"] - stats["sabotages"]
    pct = (gain_net / limit * 100) if limit > 0 else 0
    
    print(f"\n--- BILAN COMPTABLE {'I2T' if is_i2t else 'T2I'} ---")
    print(f"Total requêtes analysées : {stats['total']}")
    print(f"✅ Sauvetages (Faux -> Vrai) : {stats['sauvetages']}")
    print(f"⚠️ Sabotages (Vrai -> Faux) : {stats['sabotages']}")
    print(f"❌ Doubles Échecs (Faux -> Faux) : {stats['doubles_echecs']}")
    print(f"🆗 Confirmations (Vrai -> Vrai) : {stats['confirmations']}")
    print(f"📈 GAIN NET DE R@1 : {gain_net} requêtes (soit {pct:+.2f}%)")
    
    return stats


def evaluate_and_save_results(sorted_idx_base, sorted_idx_final, targets, is_i2t, csv_path, md_path, temps_vlm):
    """
    Calcule les métriques, récupère le temps du Stage 1, additionne le temps VLM
    et met à jour les fichiers CSV/MD.
    """
    from utils_evaluation import evaluate_from_indices 
    
    m_base = evaluate_from_indices(sorted_idx_base, targets)
    m_final = evaluate_from_indices(sorted_idx_final, targets)
    
    task_name = "I2T" if is_i2t else "T2I"
    method_name = "MIRAGE + VLM (Zero-Shot)" if is_i2t else "MIRAGE + VLM (Cascade LoRA)"
    
    # 1. Récupération du temps du Stage 1 depuis le CSV
    temps_stage1 = 0.0
    df_existing = None
    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path, index_col=0)
        if "MIRAGE (Grid Search)" in df_existing.index and "Temps (s)" in df_existing.columns:
            temps_stage1 = df_existing.loc["MIRAGE (Grid Search)", "Temps (s)"]

    # 2. Ajout des temps aux métriques
    m_base["Temps (s)"] = temps_stage1
    if temps_vlm == 0.0 and df_existing is not None and method_name in df_existing.index and "Temps (s)" in df_existing.columns:
        m_final["Temps (s)"] = df_existing.loc[method_name, "Temps (s)"]
    else:
        m_final["Temps (s)"] = temps_stage1 + temps_vlm
    
    # 3. Affichage
    print("\n" + "="*80)
    print(f"COMPARAISON FINALE : LATE FUSION vs VLM ({task_name})")
    print("="*80)
    
    df_metrics = pd.DataFrame(
        [m_base, m_final], 
        index=["1. Avant (Stage 1 Fusion)", f"2. Après ({method_name})"]
    )
    print(df_metrics.to_string(float_format=lambda x: f"{x:.3f}"))
    
    # 4. Sauvegarde
    if df_existing is not None:
        if method_name in df_existing.index:
            df_existing = df_existing.drop(index=method_name)
        df_new = pd.DataFrame([m_final], index=[method_name])
        df_updated = pd.concat([df_existing, df_new])
    else:
        df_updated = pd.DataFrame([m_final], index=[method_name])
        
    df_updated.to_csv(csv_path, float_format="%.4f")
    # Pour le markdown, on remet le temps le plus rapide en gras si on veut, 
    # mais Pandas le fera simplement avec 3 décimales via floatfmt.
    df_updated.to_markdown(md_path, floatfmt=".4f")
    
    print(f"\n✅ Résultats globaux (avec temps) sauvegardés dans {os.path.basename(csv_path)} et .md")