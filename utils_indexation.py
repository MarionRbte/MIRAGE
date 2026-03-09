import os
import pickle
import numpy as np
import faiss
from dataclasses import dataclass, field
from tqdm.auto import tqdm
import time

class SearchIndex:
    def __init__(self, dimension):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.image_ids = []

    def add_vectors(self, vectors: np.ndarray, ids: list):
        self.index.add(vectors.astype(np.float32))
        self.image_ids.extend(ids)

    def search(self, query_vector: np.ndarray, top_k=10):
        distances, indices = self.index.search(query_vector.astype(np.float32), top_k)
        return [(self.image_ids[idx], distances[0][i]) for i, idx in enumerate(indices[0])]

    def save_to_disk(self, prefix_path: str):
        faiss.write_index(self.index, f"{prefix_path}_index.bin")
        with open(f"{prefix_path}_ids.pkl", "wb") as f:
            pickle.dump(self.image_ids, f)
        print(f"  Sauvegardé : {prefix_path}_index.bin + .pkl")

    def load_from_disk(self, prefix_path: str):
        if os.path.exists(f"{prefix_path}_index.bin"):
            self.index = faiss.read_index(f"{prefix_path}_index.bin")
            with open(f"{prefix_path}_ids.pkl", "rb") as f:
                self.image_ids = pickle.load(f)
            print(f"  Chargé : {prefix_path}_index.bin ({self.index.ntotal} vecteurs)")
        else:
            print(f"  Erreur : fichiers introuvables pour {prefix_path}")


@dataclass
class IndexedCorpus:
    indices_img   : dict
    indices_txt   : dict
    txt_to_img_id : list
    img_vecs      : dict = field(default_factory=dict)
    txt_vecs      : dict = field(default_factory=dict)
    timing_stats  : dict = field(default_factory=dict)

def build_indices(
    dataset,
    encoder_registry : dict,
    image_field      : str  = "image",
    caption_field    : str  = "alt_text",
    batch_size       : int  = 64,
    save_dir         : str  = None,
    prefix           : str  = "",
    store_vectors    : bool = True,
    ) -> IndexedCorpus:


    # ── 1. Extraction images & textes ────────────────────────────────────────
    images, img_ids         = [], []
    texts,  txt_to_img_id   = [], []

    for i, item in enumerate(dataset):
        img_id = str(i)
        images.append(item[image_field])
        img_ids.append(img_id)

        captions = item[caption_field]
        if isinstance(captions, str):
            captions = [captions]

        for caption in captions:
            texts.append(str(caption))
            txt_to_img_id.append(img_id)
    n_imgs          = len(images)
    n_texts         = len(texts)
    captions_ratio  = n_texts // n_imgs

    print(f"\n[build_indices | {prefix or 'dataset'}]")
    print(f"  {n_imgs} images | {n_texts} textes ({captions_ratio} caption(s)/image)")

    # ── 2. Initialisation des index et des accumulateurs ─────────────────────
    indices_img = {name: SearchIndex(dim) for name, (_, dim) in encoder_registry.items()}
    indices_txt = {name: SearchIndex(dim) for name, (_, dim) in encoder_registry.items()}
    img_vecs_acc = {name: [] for name in encoder_registry}
    txt_vecs_acc = {name: [] for name in encoder_registry}

    timing_stats = {name: {"Images (s)": 0.0, "Textes (s)": 0.0, "Total (s)": 0.0} for name in encoder_registry}

    # ── 3. Indexation des images ─────────────────────────────────────────────
    for i in tqdm(range(0, n_imgs, batch_size), desc=f"{prefix}images"):
        b_imgs = images [i:i + batch_size]
        b_ids  = img_ids[i:i + batch_size]
        for name, (encoder, _) in encoder_registry.items():
            t0 = time.time()
            vecs = encoder.encode_image(b_imgs)
            t1 = time.time()
            timing_stats[name]["Images (s)"] += (t1 - t0)
            indices_img[name].add_vectors(vecs, b_ids)
            if store_vectors:
                img_vecs_acc[name].append(vecs)

    # ── 4. Indexation des textes ─────────────────────────────────────────────
    for i in tqdm(range(0, n_texts, batch_size), desc=f"{prefix}textes"):
        b_txts = texts        [i:i + batch_size]
        b_ids  = txt_to_img_id[i:i + batch_size]
        for name, (encoder, _) in encoder_registry.items():
            t0 = time.time()
            vecs = encoder.encode_text(b_txts)
            t1 = time.time()
            timing_stats[name]["Textes (s)"] += (t1 - t0)
            indices_txt[name].add_vectors(vecs, b_ids)
            if store_vectors:
                txt_vecs_acc[name].append(vecs)

    # ── 5. Consolidation des matrices numpy ───────────────────────────────────
    img_vecs = {name: np.vstack(v) for name, v in img_vecs_acc.items()} if store_vectors else {}
    txt_vecs = {name: np.vstack(v) for name, v in txt_vecs_acc.items()} if store_vectors else {}

    for name in encoder_registry:
        timing_stats[name]["Total (s)"] = timing_stats[name]["Images (s)"] + timing_stats[name]["Textes (s)"]

    # ── 6. Vérification ──────────────────────────────────────────────────────
    for name in encoder_registry:
        assert indices_img[name].index.ntotal == n_imgs,  f"[{name}] images manquantes"
        assert indices_txt[name].index.ntotal == n_texts, f"[{name}] textes manquants"
        print(f"  [{name}] img={indices_img[name].index.ntotal} | "
              f"txt={indices_txt[name].index.ntotal}  ✓")

    # ── 7. Sauvegarde optionnelle ─────────────────────────────────────────────
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"  Sauvegarde dans {save_dir}...")
        for name in encoder_registry:
            safe = name.lower().replace(" ", "_")
            indices_img[name].save_to_disk(f"{save_dir}/{prefix}{safe}_img")
            indices_txt[name].save_to_disk(f"{save_dir}/{prefix}{safe}_txt")
        if store_vectors:
            for name in encoder_registry:
                safe = name.lower().replace(" ", "_")
                np.save(f"{save_dir}/{prefix}{safe}_img_vecs.npy", img_vecs[name])
                np.save(f"{save_dir}/{prefix}{safe}_txt_vecs.npy", txt_vecs[name])
        print("Sauvegarde terminée.")

    return IndexedCorpus(
        indices_img   = indices_img,
        indices_txt   = indices_txt,
        txt_to_img_id = txt_to_img_id,
        img_vecs      = img_vecs,
        txt_vecs      = txt_vecs,
        timing_stats  = timing_stats,
    )