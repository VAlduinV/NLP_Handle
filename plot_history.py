# plot_history.py
import os, json
import pandas as pd
import matplotlib.pyplot as plt
import mplcyberpunk

def _load_history(path: str):
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    if path.endswith(".csv"):
        return pd.read_csv(path).to_dict(orient="list")
    raise ValueError("History file must be .json or .csv")

def run(history_path: str = "checkpoints/history.json",
        out_path: str = "img/training_curves.png") -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    hist = _load_history(history_path)

    plt.style.use("cyberpunk")

    fig, axs = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True)

    # Accuracy
    if "sparse_categorical_accuracy" in hist:
        axs[0].plot(hist["sparse_categorical_accuracy"], label="Training Accuracy")
    if "val_sparse_categorical_accuracy" in hist:
        axs[0].plot(hist["val_sparse_categorical_accuracy"], label="Validation Accuracy")
    axs[0].set_ylabel("Accuracy"); axs[0].legend()


    # Loss
    axs[1].plot(hist["loss"], label="Training Loss")
    if "val_loss" in hist:
        axs[1].plot(hist["val_loss"], label="Validation Loss")
    axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Loss"); axs[1].legend()

    for ax in axs:  # axs = np.array([ax_acc, ax_loss])
        mplcyberpunk.add_glow_effects(ax=ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)
    return out_path
