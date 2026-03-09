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

# --- FICHIERS SPÉCIFIQUES ---
BEST_WEIGHTS_FILE = f"{GRID_SEARCH_DIR}/best_weights.json"
TRAIN_JSONL = f"{FINETUNING_DIR}/train_t2i_rerank.jsonl"
LORA_OUTPUT_DIR = f"{FINETUNING_DIR}/qwen2vl_t2i_lora"
CACHE_I2T_FILE = f"{FINETUNING_DIR}/rerank_i2t_top5.json"
CACHE_T2I_FILE = f"{FINETUNING_DIR}/rerank_t2i_top5.json"


# --- CRÉATION AUTOMATIQUE ---
# Ce bloc crée les dossiers s'ils n'existent pas encore sur une nouvelle machine
for path in [BASE_DIR, RAW_DATA_DIR, GRID_SEARCH_DIR, INDEX_DIR, FINETUNING_DIR]:
    os.makedirs(path, exist_ok=True)

