import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import json
import random
import torch
import numpy as np
import config
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
NUM_SAMPLES_TRAIN = 3000 # Parfait pour le Fine-Tuning LoRA
TOP_K_MINING = 5 

# =====================================================================
# 2. GÉNÉRATION DU DATASET AVEC MINING FUSIONNÉ ET CoT DYNAMIQUE
# =====================================================================
def generate_sft_dataset_elite():
    print("🚀 Chargement du dataset...")
    train_ds = load_from_disk(f"{config.RAW_DATA_DIR}/train")
    
    # Optionnel : mélanger le dataset une fois avec un seed fixe pour avoir des images variées
    train_subset = train_ds.shuffle(seed=42).select(range(NUM_SAMPLES_TRAIN))
    
    images_pil = [x['image'] for x in train_subset]
    
    # On garde TOUTES les légendes pour construire un meilleur raisonnement
    captions_list = [x['caption'] if isinstance(x['caption'], list) else [x['caption']] for x in train_subset]
    texts_for_encoding = [caps[0] for caps in captions_list]

    # --- CHARGEMENT DYNAMIQUE DES POIDS MIRAGE ---
    print(f"⚖️ Récupération des poids optimaux depuis {config.BEST_WEIGHTS_FILE}...")
    with open(config.BEST_WEIGHTS_FILE, 'r') as f:
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

    print(f"✍️ Préparation du fichier {config.TRAIN_JSONL} avec CoT Sémantique...")
    with open(config.TRAIN_JSONL, 'w') as f:
        for i in range(NUM_SAMPLES_TRAIN):
            # Mining des Hard Negatives (La fusion nous donne les pièges les plus réalistes)
            top_indices = np.argsort(S_fused[i])[::-1][:15].tolist()

            true_target_idx = i
            negatives = [idx for idx in top_indices if idx != i][:TOP_K_MINING-1]
            
            # On simule la performance du Stage 2 : 
            # 85% du temps, le Stage 2 a eu juste (La target est en position 1)
            # 15% du temps, le Stage 2 s'est trompé (La target est reléguée plus bas)
            if random.random() < 0.85:
                candidates = [true_target_idx] + negatives
            else:
                # Simulation d'erreur : La vraie image n'est pas en position 1.
                # On force la position 1 à être un mauvais candidat (le meilleur distracteur)
                candidates = [negatives[0], true_target_idx] + negatives[1:]
                
                # On mélange les positions 2 à 5 (indices 1 à 4) pour que la vraie image 
                # puisse atterrir n'importe où entre la position 2 et la position 5.
                temp = candidates[1:]
                random.shuffle(temp)
                candidates = [candidates[0]] + temp
            
            target_pos = candidates.index(true_target_idx) + 1 # Position de 1 à 5
            
            query = captions_list[i][0] 
            alt_captions = captions_list[i][1:3] 
            details_visuels = " ".join(alt_captions) if alt_captions else query
            
            # --- LE NOUVEAU CoT STRUCTUREL ---
            # Au lieu d'un texte générique, on force le modèle à adopter une structure d'analyse
            reasoning = (
                f"Analysis of the query '{query}':\n"
                f"I need to find the exact match among the {TOP_K_MINING} candidates, paying close attention to specific details.\n"
                f"Evaluation:\n"
            )
            
            # On simule une réflexion image par image pour forcer l'attention
            for p in range(1, TOP_K_MINING + 1):
                if p == target_pos:
                    reasoning += f"- Image {p}: This image accurately depicts the specific details requested, specifically aligning with '{details_visuels}'.\n"
                else:
                    reasoning += f"- Image {p}: While it shares similarities, it lacks the precise visual constraints or fine-grained details of the query.\n"
                    
            reasoning += f"\nConclusion: Image {target_pos} is the only perfect match. Therefore, it must be ranked highest."

            content_user = []
            for img_idx in candidates:
                img_path = os.path.join(config.IMAGES_TEMP_DIR, f"train_image_{img_idx}.jpg")
                if not os.path.exists(img_path):
                    images_pil[img_idx].convert("RGB").save(img_path, "JPEG")
                content_user.append({"type": "image", "image": f"file://{os.path.abspath(img_path)}"})
            
            prompt_text = (
                f"You are an expert visual verifier. These {TOP_K_MINING} images are pre-ranked by an AI for the query: '{query}'. "
                f"Image 1 is mathematically the most probable match. Your task is to verify this. "
                f"Compare the images directly. Do NOT demote Image 1 unless another image clearly matches the subtle details better. "
                f"Think step by step, penalize visual hallucinations, and output the Final Ranking.\n"
                f"Format:\nFinal Ranking: [PositionA, PositionB, PositionC, PositionD, PositionE]"
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
if not os.path.exists(config.TRAIN_JSONL):
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

dataset = load_dataset("json", data_files=config.TRAIN_JSONL, split="train")

# =====================================================================
# 6. ENTRAÎNEMENT
# =====================================================================
training_args = TrainingArguments(
    output_dir=config.LORA_OUTPUT_DIR,
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
trainer.model.save_pretrained(config.LORA_OUTPUT_DIR)
processor.save_pretrained(config.LORA_OUTPUT_DIR)
print(f"✨ Terminé ! Les poids LoRA sont dans : {config.LORA_OUTPUT_DIR}")