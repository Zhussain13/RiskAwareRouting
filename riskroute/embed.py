"""Sentence embeddings (MiniLM) and FAQ retrieval. Demo mode uses a hashing embedder."""
import hashlib
import re

import numpy as np
import pandas as pd

import config

_model = None

_STOP = set("a an the is are was were be what when where who how why which do does did i my me "
            "you your of in on for to and or with can will it this that there at by from as please "
            "tell".split())


def hash_embed(texts, dim: int = 384, drop_stopwords: bool = False) -> np.ndarray:
    """Word + character-trigram counts folded into `dim` buckets (SmartModelRouter's demo)."""
    out = np.zeros((len(texts), dim))
    for i, t in enumerate(texts):
        t = re.sub(r"[^a-z0-9 ]", " ", str(t).lower())
        if drop_stopwords:
            t = " ".join(w for w in t.split() if w not in _STOP)
        grams = t.split() + [t[j:j + 3] for j in range(len(t) - 2) if " " not in t[j:j + 3]]
        for g in grams:
            out[i, int(hashlib.md5(g.encode()).hexdigest()[:8], 16) % dim] += 1
        n = np.linalg.norm(out[i])
        if n:
            out[i] /= n
    return out


def embed(texts, for_faq: bool = False) -> np.ndarray:
    texts = [str(t) for t in texts]
    if config.DEMO:
        return hash_embed(texts, drop_stopwords=for_faq)
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(config.EMBED_MODEL)
    return np.asarray(_model.encode(texts, normalize_embeddings=True, batch_size=64))


class FAQ:
    def __init__(self, path: str = config.FAQ_PATH):
        self.df = pd.read_csv(path).reset_index(drop=True)
        self.vecs = embed(self.df["question"], for_faq=True)

    def lookup_many(self, queries):
        """For each query: (best similarity, margin over the 2nd best, best FAQ id)."""
        Q = embed(queries, for_faq=True)
        S = Q @ self.vecs.T
        order = np.argsort(-S, axis=1)
        best = S[np.arange(len(S)), order[:, 0]]
        second = S[np.arange(len(S)), order[:, 1]] if S.shape[1] > 1 else np.zeros(len(S))
        return best, best - second, order[:, 0]

    def entry(self, i: int) -> str:
        r = self.df.iloc[int(i)]
        return f"Q: {r['question']}\nA: {r['answer']}"

    def answer(self, i: int) -> str:
        return str(self.df.iloc[int(i)]["answer"])
