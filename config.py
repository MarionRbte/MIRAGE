import os

# --- JEU DE DONNÉES ---
DATASET_NAME = "flickr30k"

# --- RACINES DES RÉPERTOIRES ---
BASE_DIR = f"./data/{DATASET_NAME}"
RAW_DATA_DIR = f"{BASE_DIR}/raw_data"
GRID_SEARCH_DIR = f"{BASE_DIR}/grid_search"
INDEX_DIR = f"{BASE_DIR}/index_sauvegardes"
FINETUNING_DIR = f"{BASE_DIR}/finetuning"
IMAGES_TEMP_DIR = f"{BASE_DIR}/images_temp_train"
RESULTS_DIR = f"{BASE_DIR}/results"

# --- FICHIERS SPÉCIFIQUES ---
BEST_WEIGHTS_FILE = f"{GRID_SEARCH_DIR}/best_weights.json"
TRAIN_JSONL = f"{FINETUNING_DIR}/train_t2i_rerank.jsonl"
LORA_OUTPUT_DIR = f"{FINETUNING_DIR}/qwen2vl_t2i_lora"
CACHE_I2T_FILE = f"{FINETUNING_DIR}/rerank_i2t_top5.json"
CACHE_T2I_FILE = f"{FINETUNING_DIR}/rerank_t2i_top5.json"
TEMPS_INDEX_CSV = f"{INDEX_DIR}/temps_indexation.csv"
TEMPS_INDEX_FILE  = f"{INDEX_DIR}/temps_indexation.md"
metriques_i2t_csv = f"{RESULTS_DIR}/resultats_i2t.csv"
metriques_i2t_md = f"{RESULTS_DIR}/resultats_i2t.md"
metriques_t2i_csv = f"{RESULTS_DIR}/resultats_t2i.csv"
metriques_t2i_md = f"{RESULTS_DIR}/resultats_t2i.md"

# --- CRÉATION AUTOMATIQUE ---
# Ce bloc crée les dossiers s'ils n'existent pas encore sur une nouvelle machine
for path in [BASE_DIR, RAW_DATA_DIR, GRID_SEARCH_DIR, INDEX_DIR, FINETUNING_DIR,RESULTS_DIR]:
    os.makedirs(path, exist_ok=True)

