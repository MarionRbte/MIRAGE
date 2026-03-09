import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import random
import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, load_from_disk
from transformers import (
    BitsAndBytesConfig, 
    Qwen2VLForConditionalGeneration, 
    AutoProcessor, 
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from qwen_vl_utils import process_vision_info
from encoders import build_encoder 

# =====================================================================
# 1. CONFIGURATION
# =====================================================================
DATASET_NAME = "flickr30k"
BASE_DIR = f"./data/{DATASET_NAME}"
DOSSIER_FINETUNING = f"{BASE_DIR}/finetuning"
DOSSIER_DATASET = f"{BASE_DIR}/raw_data" # Assure-toi que les chemins correspondent à la Phase 1
DOSSIER_SAUVEGARDE = f"{BASE_DIR}/index_sauvegardes"
DOSSIER_GRID_SEARCH = f"{BASE_DIR}/grid_search"

CHEMIN_POIDS = f"{DOSSIER_GRID_SEARCH}/best_weights.json"
DOSSIER_LORA_T2I = f"{DOSSIER_FINETUNING}/qwen2vl_t2i_lora" 
PATH_TRAIN_JSONL = f"{DOSSIER_FINETUNING}/train_t2i_rerank.jsonl"
DOSSIER_IMAGES_TEMP = f"{BASE_DIR}/images_temp_train"

os.makedirs(DOSSIER_IMAGES_TEMP, exist_ok=True)
os.makedirs(DOSSIER_FINETUNING, exist_ok=True)

NUM_SAMPLES_TRAIN = 3000 # Parfait pour le Fine-Tuning LoRA
TOP_K_MINING = 5 

# =====================================================================
# 2. GÉNÉRATION DU DATASET AVEC MINING FUSIONNÉ ET CoT DYNAMIQUE
# =====================================================================
def generate_sft_dataset_elite():
    print("🚀 Chargement du dataset...")
    train_ds = load_from_disk(f"{DOSSIER_DATASET}/train")
    
    # Optionnel : mélanger le dataset une fois avec un seed fixe pour avoir des images variées
    train_subset = train_ds.shuffle(seed=42).select(range(NUM_SAMPLES_TRAIN))
    
    images_pil = [x['image'] for x in train_subset]
    
    # On garde TOUTES les légendes pour construire un meilleur raisonnement
    captions_list = [x['caption'] if isinstance(x['caption'], list) else [x['caption']] for x in train_subset]
    texts_for_encoding = [caps[0] for caps in captions_list]

    # --- CHARGEMENT DYNAMIQUE DES POIDS MIRAGE ---
    print(f"⚖️ Récupération des poids optimaux depuis {CHEMIN_POIDS}...")
    with open(CHEMIN_POIDS, 'r') as f:
        best_weights_data = json.load(f)
        
    # On choisit la métrique cible (mAP ou R@1)
    weights_t2i = best_weights_data['t2i']['R@1']
    
    # On filtre pour ne garder que les modèles qui ont un poids strictement supérieur à 0
    model_names = [name for name, w in weights_t2i.items() if w > 0.0]
    poids = [weights_t2i[name] for name in model_names]
    
    print(f"Modèles retenus par MIRAGE : {model_names}")
    print(f"Poids associés : {poids}")
    
    S_fused = np.zeros((NUM_SAMPLES_TRAIN, NUM_SAMPLES_TRAIN))
    
    for name, w in zip(model_names, poids):
        print(f"🧠 Encodage avec {name} (Poids: {w:.2f})...")
        enc = build_encoder(name)
        
        # Encodage images
        i_vecs = []
        for i in tqdm(range(0, NUM_SAMPLES_TRAIN, 32), desc=f"Images {name}"):
            i_vecs.append(enc.encode_image(images_pil[i:i+32]))
        i_vecs = np.vstack(i_vecs)
        
        # Encodage textes
        t_vecs = []
        for i in tqdm(range(0, NUM_SAMPLES_TRAIN, 32), desc=f"Textes {name}"):
            t_vecs.append(enc.encode_text(texts_for_encoding[i:i+32]))
        t_vecs = np.vstack(t_vecs)
        
        S_fused += w * (t_vecs @ i_vecs.T)
        del enc; torch.cuda.empty_cache()

    print(f"✍️ Préparation du fichier {PATH_TRAIN_JSONL} avec CoT Sémantique...")
    with open(PATH_TRAIN_JSONL, 'w') as f:
        for i in range(NUM_SAMPLES_TRAIN):
            # Mining des Hard Negatives (La fusion nous donne les pièges les plus réalistes)
            top_indices = np.argsort(S_fused[i])[::-1][:15].tolist()
            negatives = [idx for idx in top_indices if idx != i][:TOP_K_MINING-1]
            candidates = negatives + [i]
            
            # CRUCIAL : On trie les candidats EXACTEMENT comme le ferait le Stage 1 (Correction du biais)
            candidates.sort(key=lambda idx: S_fused[i, idx], reverse=True)
            
            # Ajout d'une très légère dose de hasard (20%) pour la robustesse du VLM
            if random.random() < 0.2:
                random.shuffle(candidates)
            
            target_pos = candidates.index(i) + 1
            
            # --- LE NOUVEAU CoT DYNAMIQUE ---
            query = captions_list[i][0] 
            alt_captions = captions_list[i][1:3] 
            details_visuels = " ".join(alt_captions) if alt_captions else query
            
            reasoning = (
                f"Analysis: The user is looking for '{query}'. "
                f"I will carefully verify the pre-ranked candidates. "
                f"Image {target_pos} perfectly matches the specific details described, including: '{details_visuels}'. "
                f"The other candidate images are hard negatives: they might share the general scene or objects, "
                f"but they lack the precise visual constraints requested in the query. "
                f"Therefore, Image {target_pos} must be ranked highest."
            )

            content_user = []
            for img_idx in candidates:
                img_path = os.path.join(DOSSIER_IMAGES_TEMP, f"train_image_{img_idx}.jpg")
                if not os.path.exists(img_path):
                    images_pil[img_idx].convert("RGB").save(img_path, "JPEG")
                content_user.append({"type": "image", "image": f"file://{os.path.abspath(img_path)}"})
            
            prompt_text = (
                f"You are an expert visual verifier. These {TOP_K_MINING} images are pre-ranked by an AI for the query: '{query}'. "
                f"Image 1 is mathematically the most probable match. Your task is to verify this. "
                f"Do NOT demote Image 1 unless another image clearly matches the subtle details better. "
                f"Think step by step, penalize visual hallucinations, and output the Final Ranking."
            )
            content_user.append({"type": "text", "text": prompt_text})
            
            ordered_positions = [target_pos]
            for p in range(1, TOP_K_MINING + 1):
                if p != target_pos:
                    ordered_positions.append(p)
            
            ranking_text = ", ".join([str(p) for p in ordered_positions])

            msg = {
                "messages": [
                    {"role": "user", "content": content_user},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": f"{reasoning} Final Ranking: [{ranking_text}]"}
                    ]}
                ]
            }
            f.write(json.dumps(msg) + '\n')

# =====================================================================
# 3. CHARGEMENT MODÈLE ET PROCESSOR
# =====================================================================
if not os.path.exists(PATH_TRAIN_JSONL):
    generate_sft_dataset_elite()
    
print("💎 Chargement du modèle et du processor...")
model_id = "Qwen/Qwen2-VL-7B-Instruct"

processor = AutoProcessor.from_pretrained(
    model_id, 
    min_pixels=256*28*28, # ~ 450x450
    max_pixels=512*28*28  # ~ 630x630 (Laisse le VLM voir les détails fins !)
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

model = prepare_model_for_kbit_training(model)

# =====================================================================
# 4. CONFIGURATION LORA
# =====================================================================
lora_config = LoraConfig(
    r=32, 
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# =====================================================================
# 5. DATA COLLATOR
# =====================================================================
def clean_messages(messages):
    cleaned = []
    for msg in messages:
        new_msg = {"role": msg["role"]}
        if isinstance(msg["content"], list):
            new_content = [{k: v for k, v in elem.items() if v is not None} for elem in msg["content"]]
            new_msg["content"] = new_content
        else:
            new_msg["content"] = msg["content"]
        cleaned.append(new_msg)
    return cleaned

def collate_fn(examples):
    cleaned_messages = [clean_messages(example["messages"]) for example in examples]
    image_inputs, video_inputs = process_vision_info(cleaned_messages)
    
    texts = [processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False) for msg in cleaned_messages]
    
    inputs = processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    labels = inputs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    
    assistant_token_id = processor.tokenizer.encode("<|im_start|>assistant", add_special_tokens=False)[-1]
    
    for i in range(labels.shape[0]):
        token_list = labels[i].tolist()
        try:
            assistant_idx = token_list.index(assistant_token_id)
            labels[i, :assistant_idx + 2] = -100
        except ValueError:
            pass 
            
    inputs["labels"] = labels
    return inputs

dataset = load_dataset("json", data_files=PATH_TRAIN_JSONL, split="train")

# =====================================================================
# 6. ENTRAÎNEMENT
# =====================================================================
training_args = TrainingArguments(
    output_dir=DOSSIER_LORA_T2I,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=5e-5,
    warmup_ratio=0.1,
    num_train_epochs=2,
    bf16=True,
    logging_steps=5,
    optim="paged_adamw_8bit",
    save_strategy="no",
    gradient_checkpointing=True,
    remove_unused_columns=False,
)

print("🚀 Lancement de l'entraînement...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=collate_fn,
)

model.config.use_cache = False 
trainer.train()

# =====================================================================
# 7. SAUVEGARDE
# =====================================================================
trainer.model.save_pretrained(DOSSIER_LORA_T2I)
processor.save_pretrained(DOSSIER_LORA_T2I)
print(f"✨ Terminé ! Les poids LoRA sont dans : {DOSSIER_LORA_T2I}")