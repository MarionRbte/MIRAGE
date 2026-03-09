import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from abc import ABC, abstractmethod
from transformers import (
    AutoProcessor, AutoModel, AutoTokenizer, 
    SiglipProcessor, SiglipModel, 
    BlipProcessor, BlipForImageTextRetrieval
)

# --- REGISTRE ---
ENCODER_REGISTRY: dict[str, type] = {}

def register_encoder(name: str):
    def decorator(cls):
        ENCODER_REGISTRY[name] = cls
        return cls
    return decorator

def build_encoder(name: str, **kwargs) -> "BaseEncoder":
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"Encodeur inconnu: '{name}'")
    return ENCODER_REGISTRY[name](**kwargs)

# --- CLASSES DE BASE ---
class BaseEncoder(ABC):
    needs_calibration: bool = False
    def __init__(self, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def encode_image(self, images: list) -> np.ndarray: ...
    @abstractmethod
    def encode_text(self, texts: list) -> np.ndarray: ...
    def calibrate(self, sample_images: list, sample_texts: list, **kwargs): pass

class BaseTrainableEncoder(BaseEncoder, ABC):
    needs_calibration: bool = True
    def __init__(self, device: str | None = None):
        super().__init__(device)
        self.is_calibrated: bool = False

    def _check_calibrated(self):
        if not self.is_calibrated:
            raise RuntimeError(f"{self.__class__.__name__} n'est pas calibré.")

    def encode_image(self, images: list) -> np.ndarray:
        self._check_calibrated()
        return self._encode_image_projected(images)

    def encode_text(self, texts: list) -> np.ndarray:
        self._check_calibrated()
        return self._encode_text_projected(texts)

    @abstractmethod
    def _encode_image_projected(self, images: list) -> np.ndarray: ...
    @abstractmethod
    def _encode_text_projected(self, texts: list) -> np.ndarray: ...

# --- UTILITAIRES ---
class DualProjector(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 2048, shared_dim: int = 512):
        super().__init__()
        self.txt_proj = self._mlp(input_dim, hidden_dim, shared_dim)
        self.img_proj = self._mlp(input_dim, hidden_dim, shared_dim)

    @staticmethod
    def _mlp(in_dim, hidden_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward_txt(self, x: torch.Tensor) -> torch.Tensor: return self.txt_proj(x)
    def forward_img(self, x: torch.Tensor) -> torch.Tensor: return self.img_proj(x)

def contrastive_train(connector, X_tensor, Y_tensor, device, batch_size=512, epochs=20, lr=2e-4):
    # STABILITÉ : On force le calcul en Float32 pour éviter les NaNs
    logit_scale = nn.Parameter(torch.tensor(np.log(1 / 0.07), dtype=torch.float32, device=device))
    optimizer = optim.AdamW(list(connector.parameters()) + [logit_scale], lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    loader = DataLoader(TensorDataset(X_tensor, Y_tensor), batch_size=batch_size, shuffle=True)
    
    connector.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            
            # Calcul en float32 (plus stable pour la loss)
            tf = connector.forward_txt(bx.float()) 
            imf = connector.forward_img(by.float())
            
            # Normalisation avec epsilon pour éviter division par zéro
            tf = tf / (tf.norm(p=2, dim=-1, keepdim=True) + 1e-8)
            imf = imf / (imf.norm(p=2, dim=-1, keepdim=True) + 1e-8)

            scale = logit_scale.exp().clamp(max=100)
            logits = scale * tf @ imf.t()
            labels = torch.arange(len(bx), dtype=torch.long, device=device)
            
            loss = (criterion(logits, labels) + criterion(logits.t(), labels)) / 2.0
            
            if not torch.isnan(loss):
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        print(f"  Époque {epoch+1:02d}/{epochs} | Loss: {total_loss/len(loader):.4f}")
    connector.eval()

# --- ENCODEURS ---
@register_encoder("blip")
class BlipEncoder(BaseEncoder):
    def __init__(self, model_id="Salesforce/blip-itm-large-coco", device=None):
        super().__init__(device)
        self.processor = BlipProcessor.from_pretrained(model_id)
        self.model = BlipForImageTextRetrieval.from_pretrained(model_id, dtype=torch.float16).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device, torch.float16)
        out = self.model.vision_model(**inputs)
        feats = self.model.vision_proj(out[0][:, 0, :])
        return (feats / (feats.norm(p=2, dim=-1, keepdim=True) + 1e-8)).cpu().float().numpy()

    @torch.no_grad()
    def encode_text(self, texts):
        inputs = self.processor(text=texts, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(self.device)
        out = self.model.text_encoder(**inputs)
        feats = self.model.text_proj(out[0][:, 0, :])
        return (feats / (feats.norm(p=2, dim=-1, keepdim=True) + 1e-8)).cpu().float().numpy()

@register_encoder("siglip")
class SiglipEncoder(BaseEncoder):
    def __init__(self, model_id="google/siglip-so400m-patch14-384", device=None):
        super().__init__(device)
        # CORRECTION : Utilisation de SiglipModel explicite pour éviter les AttributeError
        self.processor = SiglipProcessor.from_pretrained(model_id) 
        self.model = SiglipModel.from_pretrained(model_id, dtype=torch.float16).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, images):
        inputs = self.processor(images=images, return_tensors="pt").to(self.device, torch.float16)
        out = self.model.get_image_features(**inputs)
        
        # SÉCURITÉ : On extrait le tenseur brut s'il est encapsulé dans un objet Output
        feats = out.pooler_output if hasattr(out, 'pooler_output') else out
        if not isinstance(feats, torch.Tensor):
             feats = out[0] if isinstance(out, (list, tuple)) else out

        return (feats / (feats.norm(p=2, dim=-1, keepdim=True) + 1e-8)).cpu().float().numpy()

    @torch.no_grad()
    def encode_text(self, texts):
        inputs = self.processor(text=texts, padding="max_length", truncation=True, return_tensors="pt").to(self.device)
        out = self.model.get_text_features(**inputs)
        
        # SÉCURITÉ : Même extraction pour le texte
        feats = out.pooler_output if hasattr(out, 'pooler_output') else out
        if not isinstance(feats, torch.Tensor):
             feats = out[0] if isinstance(out, (list, tuple)) else out
            
        return (feats / (feats.norm(p=2, dim=-1, keepdim=True) + 1e-8)).cpu().float().numpy()

@register_encoder("dino_bert")
class DinoBertEncoder(BaseTrainableEncoder):
    def __init__(self, dino_id="facebook/dinov2-base", bert_id="sentence-transformers/all-mpnet-base-v2", device=None):
        super().__init__(device)
        self.processor_img = AutoProcessor.from_pretrained(dino_id)
        self.model_img = AutoModel.from_pretrained(dino_id, dtype=torch.float16).to(self.device)
        self.model_txt = AutoModel.from_pretrained(bert_id, dtype=torch.float16).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(bert_id)
        # STABILITÉ : On laisse le connector en float32 (pas de .half())
        self.connector = DualProjector().to(self.device)

    @torch.no_grad()
    def _encode_image_raw(self, images):
        inputs = self.processor_img(images=images, return_tensors="pt").to(self.device, torch.float16)
        return self.model_img(**inputs).last_hidden_state[:, 0, :].cpu().float().numpy()

    @torch.no_grad()
    def _encode_text_raw(self, texts):
        inputs = self.tokenizer([str(t) for t in texts], return_tensors="pt", padding=True, truncation=True).to(self.device)
        out = self.model_txt(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).expand(out.last_hidden_state.size()).float()
        feats = (out.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return feats.cpu().float().numpy()

    @torch.no_grad()
    def _encode_image_projected(self, images):
        # On travaille en float32 pour le projecteur
        raw = torch.tensor(self._encode_image_raw(images), dtype=torch.float32).to(self.device)
        feats = self.connector.forward_img(raw)
        return (feats / (feats.norm(p=2, dim=-1, keepdim=True) + 1e-8)).cpu().numpy()

    @torch.no_grad()
    def _encode_text_projected(self, texts):
        raw = torch.tensor(self._encode_text_raw(texts), dtype=torch.float32).to(self.device)
        feats = self.connector.forward_txt(raw)
        return (feats / (feats.norm(p=2, dim=-1, keepdim=True) + 1e-8)).cpu().numpy()

    def calibrate(self, sample_images, sample_texts, batch_size=32, epochs=20):
        print("1. Extraction des vecteurs bruts...")
        X, Y = [], []
        for i in range(0, len(sample_images), batch_size):
            X.append(self._encode_text_raw(sample_texts[i:i + batch_size]))
            Y.append(self._encode_image_raw(sample_images[i:i + batch_size]))
        
        # On passe les données en float32 pour l'entraînement du petit MLP
        X_t = torch.tensor(np.vstack(X), dtype=torch.float32).to(self.device)
        Y_t = torch.tensor(np.vstack(Y), dtype=torch.float32).to(self.device)
        
        print(f"2. Entraînement contrastif stable...")
        contrastive_train(self.connector, X_t, Y_t, self.device, epochs=epochs)
        self.is_calibrated = True

@register_encoder("jina_clip")
class JinaClipEncoder(BaseEncoder):
    def __init__(self, model_id="jinaai/jina-clip-v1", device=None):
        super().__init__(device)
        
        from transformers import AutoConfig, AutoModel
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        import os

        print(f"Chargement de {model_id} (Final Fix)...")

        # 1. Config
        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        
        # 2. Création du modèle vide sur CPU (évite le bug meta-tensor)
        with torch.device("cpu"):
            self.model = AutoModel.from_config(config, trust_remote_code=True)
        
        # 3. Chargement des poids
        model_path = snapshot_download(repo_id=model_id)
        weights_path = os.path.join(model_path, "model.safetensors")
        
        state_dict = load_file(weights_path)
        
        # CORRECTION : strict=False permet d'ignorer les clés inattendues comme 'rope'
        # qui font planter le chargement strict.
        self.model.load_state_dict(state_dict, strict=False)
        
        # 4. Transfert vers le device
        self.model.to(self.device)
        if "cuda" in str(self.device):
            self.model = self.model.half()
            
        self.model.eval()
        print("Jina CLIP est opérationnel !")

    @torch.no_grad()
    def encode_image(self, images: list) -> np.ndarray:
        # On s'assure que les images sont au bon format pour Jina
        feats = self.model.encode_image(images)
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats).to(self.device).float()
        norm_feats = feats / (feats.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        return norm_feats.cpu().numpy()

    @torch.no_grad()
    def encode_text(self, texts: list) -> np.ndarray:
        feats = self.model.encode_text(texts)
        if isinstance(feats, np.ndarray):
            feats = torch.from_numpy(feats).to(self.device).float()
        norm_feats = feats / (feats.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        return norm_feats.cpu().numpy()

import open_clip

@register_encoder("convnext_clip")
class ConvNextCLIPEncoder(BaseEncoder):
    def __init__(self, device: str | None = None):
        super().__init__(device)
        print("Chargement de ConvNeXt-CLIP (XXLarge)...")
        self.model_id = "convnext_xxlarge"
        self.pretrained = "laion2b_s34b_b82k_augreg" # Version très performante
        
        # Chargement via open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_id, 
            pretrained=self.pretrained, 
            device=self.device,
            precision='fp16'
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_id)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, images: list) -> np.ndarray:
        # 1. Transformation des images PIL
        # 2. .to(self.device) -> envoi sur GPU
        # 3. .half() -> conversion en float16 pour correspondre au modèle
        processed_imgs = torch.stack([self.preprocess(img) for img in images]).to(self.device).half()
        
        image_features = self.model.encode_image(processed_imgs)
        
        # Normalisation L2
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().float().numpy()

    @torch.no_grad()
    def encode_text(self, texts: list) -> np.ndarray:
        text_tokens = self.tokenizer(texts).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        # Normalisation $L_2$
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().float().numpy()
    
@register_encoder("eva_clip")
class EvaClipEncoder(BaseEncoder):
    def __init__(self, device: str | None = None):
        super().__init__(device)
        print("Chargement de EVA-CLIP...")
        self.model_id = "EVA02-L-14"
        
        import open_clip
        
        # 1. On récupère dynamiquement les tags disponibles pour ce modèle
        available_tags = open_clip.list_pretrained_tags_by_model(self.model_id)
        
        # Fallback de sécurité si ta version de open_clip est plus ancienne
        if not available_tags:
            print(f"⚠️ {self.model_id} non trouvé dans cette version, fallback sur EVA01-g-14...")
            self.model_id = "EVA01-g-14"
            available_tags = open_clip.list_pretrained_tags_by_model(self.model_id)
            
        # 2. On prend automatiquement le premier set de poids valide
        self.pretrained = available_tags[0] 
        print(f"-> Modèle sélectionné : {self.model_id} | Poids : {self.pretrained}")
        
        # Chargement via open_clip
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_id, 
            pretrained=self.pretrained, 
            device=self.device,
            precision='fp16'
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_id)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, images: list) -> np.ndarray:
        processed_imgs = torch.stack([self.preprocess(img) for img in images]).to(self.device).half()
        image_features = self.model.encode_image(processed_imgs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().float().numpy()

    @torch.no_grad()
    def encode_text(self, texts: list) -> np.ndarray:
        text_tokens = self.tokenizer(texts).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().float().numpy()


@register_encoder("coca_clip")
class CocaClipEncoder(BaseEncoder):
    def __init__(self, device: str | None = None):
        super().__init__(device)
        print("Chargement de CoCa (Fine-tuné MS-COCO) en FP32 (Mode stable)...")
        import open_clip
        self.model_id = "coca_ViT-L-14"
        self.pretrained = "mscoco_finetuned_laion2B-s13B-b90k" 
        
        # FIX : On utilise la précision FP32 (par défaut) pour éviter les crashs
        # liés au LayerNormFp32 interne de l'architecture CoCa.
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_id, 
            pretrained=self.pretrained, 
            device=self.device,
            precision='fp32' 
        )
        
        self.tokenizer = open_clip.get_tokenizer(self.model_id)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, images: list) -> np.ndarray:
        # FIX : Plus de .half() sur les inputs !
        processed_imgs = torch.stack([self.preprocess(img) for img in images]).to(self.device)
        image_features = self.model.encode_image(processed_imgs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().float().numpy()

    @torch.no_grad()
    def encode_text(self, texts: list) -> np.ndarray:
        text_tokens = self.tokenizer(texts).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().float().numpy()

@register_encoder("dfn_clip")
class DFNClipEncoder(BaseEncoder):
    def __init__(self, device: str | None = None):
        super().__init__(device)
        print("Chargement de DFN-CLIP (Apple ViT-Huge)...")
        import open_clip
        self.model_id = "ViT-H-14" # Modèle "Huge" (très puissant)
        self.pretrained = "dfn5b"  # Entraîné sur le dataset ultra-filtré d'Apple
        
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_id, 
            pretrained=self.pretrained, 
            device=self.device,
            precision='fp16'
        )
        self.tokenizer = open_clip.get_tokenizer(self.model_id)
        self.model.eval()

    @torch.no_grad()
    def encode_image(self, images: list) -> np.ndarray:
        processed_imgs = torch.stack([self.preprocess(img) for img in images]).to(self.device).half()
        image_features = self.model.encode_image(processed_imgs)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().float().numpy()

    @torch.no_grad()
    def encode_text(self, texts: list) -> np.ndarray:
        text_tokens = self.tokenizer(texts).to(self.device)
        text_features = self.model.encode_text(text_tokens)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().float().numpy()