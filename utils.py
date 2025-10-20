import os
import numpy as np

def read_text(path):
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()

def build_char_vocab(text):
    chars = sorted(list(set(text)))
    stoi = {ch:i for i,ch in enumerate(chars)}
    itos = {i:ch for ch,i in stoi.items()}
    return stoi, itos

def vectorize_text(text, stoi):
    return np.array([stoi[ch] for ch in text if ch in stoi], dtype=np.int32)
