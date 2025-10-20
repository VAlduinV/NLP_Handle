import numpy as np
import heapq

def sample_with_temperature(logits, temperature=1.0):
    """Temperature scaling for logits -> sample single index."""
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = np.asarray(logits).astype(np.float64)
    logits = logits / temperature
    # numerically stable softmax
    logits = logits - np.max(logits)
    probs = np.exp(logits)
    probs = probs / probs.sum()
    return np.random.choice(len(probs), p=probs)

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0):
    """Filter logits: keep top_k or minimal set summing to top_p (nucleus)."""
    logits = np.array(logits, dtype=np.float64)
    # Top-k
    if top_k and top_k > 0:
        # find threshold using heapq.nlargest
        top_vals = heapq.nlargest(top_k, logits)
        kth = top_vals[-1]
        mask = logits < kth
        logits[mask] = -1e9
    # Top-p (nucleus)
    if top_p and 0.0 < top_p < 1.0:
        # Convert to probs
        shift = np.max(logits)
        probs = np.exp(logits - shift)
        probs = probs / probs.sum()
        idxs = np.argsort(probs)[::-1]
        cum = 0.0
        keep = []
        for i in idxs:
            keep.append(i)
            cum += probs[i]
            if cum >= top_p:
                break
        mask = np.ones_like(logits, dtype=bool)
        mask[keep] = False
        logits[mask] = -1e9
    return logits
