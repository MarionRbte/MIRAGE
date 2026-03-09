import torch
import re
from abc import ABC, abstractmethod
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class BaseReranker(ABC):
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def generate_response(self, prompt: str, images: list = None) -> str:
        pass

    def parse_ranking(self, text: str, expected_len: int, keyword="Image") -> list:
            # 1. On cherche la ligne "Final Ranking:" peu importe ce qui suit
            match = re.search(r'Final Ranking:\s*(?:\[)?(.*?)(?:\])?$', text, re.IGNORECASE | re.MULTILINE)
            
            if match:
                ranking_str = match.group(1)
                # 2. On extrait tous les nombres trouvés dans cette ligne
                numbers = [int(n) for n in re.findall(r'\d+', ranking_str)]
                
                final_order = []
                for x in numbers:
                    if 1 <= x <= expected_len and x not in final_order:
                        final_order.append(x)
                
                # 3. Complétion si le modèle a oublié des éléments
                missing = [i for i in range(1, expected_len + 1) if i not in final_order]
                final_order.extend(missing)
                
                if len(final_order) == expected_len:
                    return final_order
            
            # 4. Si le modèle n'a pas généré de "Final Ranking", on renvoie None pour gérer le fallback intelligemment
            print(f"⚠️ Échec du parsing pour le texte. Application du fallback de sécurité.")
            return None


class Qwen2VLReranker(BaseReranker):
    def __init__(self, model_id="Qwen/Qwen2-VL-7B-Instruct", device=None):
        super().__init__(device)
        print(f"⏳ Chargement de {model_id} en Bfloat16 (Sans quantification)...")
        
        # Processeur avec résolution optimisée pour voir les détails fins
        self.processor = AutoProcessor.from_pretrained(
            model_id, 
            min_pixels=256*28*28, # ~ 450x450
            max_pixels=512*28*28
        )

        # Chargement en Bfloat16 natif (Précision maximale pour une RTX 4080)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16, # Format haute précision
            device_map="auto"           # Placement automatique sur la GPU
        ).eval()
        
        print("✅ Qwen2-VL chargé avec succès !")

    @torch.no_grad()
    def generate_response(self, prompt_text: str, images_pil: list = None) -> str:
        content = []
        if images_pil:
            for img in images_pil:
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
        ).to(self.device)

        # Génération (Augmentée à 512 pour le raisonnement CoT)
        generated_ids = self.model.generate(**inputs, max_new_tokens=512) 

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        return response