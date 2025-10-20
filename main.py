import argparse
import pandas as pd, json, os
import numpy as np

# Keras / TensorFlow
from tensorflow import keras  # PyCharm може підкреслювати як warning, але це працює нормально.

# Local modules
import train_char_rnn as tr
from utils import read_text, build_char_vocab, vectorize_text
from sampling import sample_with_temperature, top_k_top_p_filtering
import generate as genmod
import plot_history as ph  # ← поряд з іншими імпортами


def cmd_train(args):
    # Load and vectorize text
    text = read_text(args.data_path)
    stoi, itos = build_char_vocab(text)
    encoded = vectorize_text(text, stoi)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
        json.dump({'stoi': stoi, 'itos': itos}, f, ensure_ascii=False)

    # Dataset
    ds = tr.make_dataset(encoded, seq_len=args.seq_len, batch=args.batch)

    # Build model
    model = tr.build_model(vocab_size=len(stoi), embed_dim=args.embed_dim, rnn_units=args.rnn_units)

    # Callbacks
    ck = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.out_dir, 'best.keras'),
        monitor='val_sparse_categorical_accuracy',
        save_best_only=True,
        save_weights_only=False
    )
    es = keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=3, restore_best_weights=True)

    # Train/Val split
    ds_list = list(ds)
    n_train = int(0.9 * len(ds_list))
    train_ds = ds.take(n_train)
    val_ds = ds.skip(n_train)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[ck, es]
    )
    model.save(os.path.join(args.out_dir, 'final.keras'))
    print(f"Training complete. Checkpoints in: {args.out_dir}")

    hist = history.history
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "history.json"), "w", encoding="utf-8") as hf:
        json.dump(hist, hf, ensure_ascii=False, indent=2)
    pd.DataFrame(hist).to_csv(os.path.join(args.out_dir, "history.csv"), index=False)

    # Save final model
    model.save(os.path.join(args.out_dir, 'final.keras'))
    print(f"Training complete. Checkpoints in: {args.out_dir}")


def cmd_generate(args):
    # Load model & vocab
    model = keras.models.load_model(args.checkpoint, compile=False)
    stoi, itos = genmod.load_vocab(args.vocab)

    text = genmod.generate_text(
        model, stoi, itos,
        seed=args.seed,
        length=args.length,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p
    )
    print(text)

def cmd_plot_history(args):
    saved = ph.run(history_path=args.history, out_path=args.out)
    print(f"Saved: {saved}")

def build_arg_parser():
    p = argparse.ArgumentParser(description="Ukrainian TextGen (LSTM) – single entry point")
    sub = p.add_subparsers(dest="command", required=True)

    # train subcommand
    pt = sub.add_parser("train", help="Train LSTM char-level model")
    pt.add_argument('--data_path', type=str, default='data/lys_mykyta.txt')
    pt.add_argument('--seq_len', type=int, default=120)
    pt.add_argument('--batch', type=int, default=128)
    pt.add_argument('--epochs', type=int, default=20)
    pt.add_argument('--embed_dim', type=int, default=128)
    pt.add_argument('--rnn_units', type=int, default=512)
    pt.add_argument('--out_dir', type=str, default='checkpoints')
    pt.set_defaults(func=cmd_train)

    # generate subcommand
    pg = sub.add_parser("generate", help="Generate text from a checkpoint")
    pg.add_argument('--checkpoint', type=str, default='checkpoints/best.keras')
    pg.add_argument('--vocab', type=str, default='checkpoints/vocab.json')
    pg.add_argument('--seed', type=str, default='Надійшла весна')
    pg.add_argument('--length', type=int, default=500)
    pg.add_argument('--temperature', type=float, default=0.8)
    pg.add_argument('--top_k', type=int, default=0)
    pg.add_argument('--top_p', type=float, default=0.0)
    pg.set_defaults(func=cmd_generate)

    # plot-history subcommand
    phs = sub.add_parser("plot-history", help="Plot curves from checkpoints/history.json or history.csv")
    phs.add_argument("--history", type=str, default="checkpoints/history.json")
    phs.add_argument("--out", type=str, default="img/training_curves.png")
    phs.set_defaults(func=cmd_plot_history)

    return p


def main():
    parser = build_arg_parser()
    import sys
    if len(sys.argv) == 1:
        # No subcommand provided: print friendly help and examples
        parser.print_help()
        print(
            "\nExamples:\n"
            "  python main.py train --data_path data/lys_mykyta.txt --seq_len 120 --batch 128 --epochs 20\n"
            "  python main.py generate --checkpoint checkpoints/best.keras --seed \"Надійшла весна\" --length 800 --temperature 0.8\n"
            "  python main.py plot-history --history checkpoints/history.json --out img/training_curves.png"
        )
        sys.exit(2)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
