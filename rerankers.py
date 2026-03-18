import torch
from abc import ABC, abstractmethod
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import re

class BaseReranker(ABC):
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def generate_response(self, prompt: str, images: list = None) -> str:
        pass


class Qwen2VLReranker(BaseReranker):
    def __init__(self, model_id="Qwen/Qwen2-VL-7B-Instruct", device=None):
        super().__init__(device)
        print(f"⏳ Chargement de {model_id} en Bfloat16 (Sans quantification)...")
        
        # Processeur avec résolution optimisée pour voir les détails fins
        self.processor = AutoProcessor.from_pretrained(model_id)

        # Chargement en Bfloat16 natif (Précision maximale pour une RTX 4080)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, # Format haute précision
            device_map="auto",           # Placement automatique sur la GPU
            attn_implementation="flash_attention_2"
        ).eval()
        
        print("✅ Qwen2-VL chargé avec succès !")

    @torch.no_grad()
    def generate_response(self, prompt_text: str, images_pil: list = None) -> str:
        content = []
        if images_pil:
            for img in images_pil:
                # 🎯 LE SECRET EST ICI : On force l'équivalent 512x512 dynamique
                content.append({"type": "image", "image": img})
        
        content.append({"type": "text", "text": prompt_text})
        messages = [{"role": "user", "content": content}]

        # Préparation des inputs
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        
        inputs = self.processor(
            text=[text_prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Génération (Augmentée à 512 pour le raisonnement CoT)
        generated_ids = self.model.generate(**inputs, max_new_tokens=512) 

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return response
    @torch.no_grad()
    def score_image_pointwise_batch(self, prompt_texts: list, images_pil: list) -> list:
        """
        Version ultra-rapide (Batch) du Pointwise Reranking.
        """
        messages_batch = []
        for prompt, img in zip(prompt_texts, images_pil):
            content = [
                # 🎯 LE SECRET EST ICI AUSSI
                {"type": "image", "image": img},
                {"type": "text", "text": prompt}
            ]
            messages_batch.append([{"role": "user", "content": content}])
        
        # Préparation du batch
        texts_batch = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_batch]
        image_inputs, video_inputs = process_vision_info(messages_batch)
        
        inputs = self.processor(
            text=texts_batch,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # Génération du batch en parallèle
        outputs = self.model.generate(
            **inputs, 
            max_new_tokens=1, 
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        # On récupère les probabilités pour les 5 éléments du batch
        logits_batch = outputs.scores[0]
        yes_token_id = self.processor.tokenizer.encode("Yes", add_special_tokens=False)[0]
        no_token_id = self.processor.tokenizer.encode("No", add_special_tokens=False)[0]
        
        import math
        probs_yes = []
        for i in range(len(prompt_texts)):
            yes_logit = logits_batch[i][yes_token_id].item()
            no_logit = logits_batch[i][no_token_id].item()
            prob = math.exp(yes_logit) / (math.exp(yes_logit) + math.exp(no_logit))
            probs_yes.append(prob)
            
        return probs_yes
    @torch.no_grad()
    def score_image_cot_batch(self, prompt_texts: list, images_pil: list) -> list:
        """
        Version Chain of Thought (CoT). Le modèle génère un raisonnement
        puis donne un score sur 100 qui est converti en probabilité [0, 1].
        """
        messages_batch = []
        for prompt, img in zip(prompt_texts, images_pil):
            content = [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt}
            ]
            messages_batch.append([{"role": "user", "content": content}])
        
        texts_batch = [self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) for msg in messages_batch]
        image_inputs, video_inputs = process_vision_info(messages_batch)
        
        inputs = self.processor(
            text=texts_batch,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        # On donne plus de tokens (256) pour laisser le modèle "réfléchir"
        outputs = self.model.generate(**inputs, max_new_tokens=80)
        
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, outputs)
        ]
        
        responses = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # Extraction par Regex du score final
        import re
        scores = []
        for resp in responses:
            # Cherche "Score: 85" ou "Score: 100" à la fin du texte
            match = re.search(r'Score:\s*(\d+)', resp, re.IGNORECASE)
            if match:
                score = min(float(match.group(1)) / 100.0, 1.0) # Normalise entre 0 et 1
            else:
                score = 0.5 # Fallback neutre si le modèle a oublié de formater la note
            scores.append(score)
            
        return scores