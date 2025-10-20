import json, argparse, os, numpy as np
from tensorflow import keras
from sampling import sample_with_temperature, top_k_top_p_filtering

def load_vocab(path):
    with open(path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    stoi = {k:int(v) if isinstance(v, str) and v.isdigit() else v for k,v in d['stoi'].items()}
    itos = {int(k):v for k,v in d['itos'].items()}
    return stoi, itos

def generate_text(model, stoi, itos, seed, length=500, temperature=0.8, top_k=0, top_p=0.0):
    x = [stoi.get(ch, None) for ch in seed if ch in stoi]
    x = [i for i in x if i is not None]
    if not x:
        x = [np.random.randint(0, len(stoi))]

    out = seed
    for _ in range(length):
        inp = np.array(x[-256:], dtype=np.int32)[None, ...]
        logits = model.predict(inp, verbose=0)[0, -1]
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        idx = sample_with_temperature(logits, temperature=max(temperature, 1e-3))
        out += itos.get(idx, '')
        x.append(idx)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', type=str, default='checkpoints/best.keras')
    ap.add_argument('--vocab', type=str, default='checkpoints/vocab.json')
    ap.add_argument('--seed', type=str, default='Надійшла весна')
    ap.add_argument('--length', type=int, default=500)
    ap.add_argument('--temperature', type=float, default=0.8)
    ap.add_argument('--top_k', type=int, default=0)
    ap.add_argument('--top_p', type=float, default=0.0)
    args = ap.parse_args()

    model = keras.models.load_model(args.checkpoint, compile=False)
    stoi, itos = load_vocab(args.vocab)
    text = generate_text(model, stoi, itos, args.seed, args.length, args.temperature, args.top_k, args.top_p)
    print(text)

if __name__ == '__main__':
    main()
