import os
import argparse
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

from utils import read_text, build_char_vocab, vectorize_text

def make_dataset(encoded, seq_len=120, batch=128, buffer=10000):
    ds = tf.data.Dataset.from_tensor_slices(encoded)
    ds = ds.window(seq_len + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(seq_len + 1))
    ds = ds.map(lambda s: (s[:-1], s[1:]))
    ds = ds.shuffle(buffer).batch(batch).prefetch(tf.data.AUTOTUNE)
    return ds

def build_model(vocab_size, embed_dim=128, rnn_units=512):
    inputs = keras.layers.Input(shape=(None,), dtype="int32")
    x = keras.layers.Embedding(vocab_size, embed_dim)(inputs)
    x = keras.layers.LSTM(rnn_units, return_sequences=True)(x)
    x = keras.layers.LSTM(rnn_units, return_sequences=True)(x)
    outputs = keras.layers.Dense(vocab_size)(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-4),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/lys_mykyta.txt')
    parser.add_argument('--seq_len', type=int, default=120)
    parser.add_argument('--batch', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--rnn_units', type=int, default=512)
    parser.add_argument('--out_dir', type=str, default='checkpoints')
    args = parser.parse_args()

    text = read_text(args.data_path)
    stoi, itos = build_char_vocab(text)
    encoded = vectorize_text(text, stoi)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, 'vocab.json'), 'w', encoding='utf-8') as f:
        import json
        json.dump({'stoi': stoi, 'itos': itos}, f, ensure_ascii=False)

    ds = make_dataset(encoded, seq_len=args.seq_len, batch=args.batch)

    model = build_model(vocab_size=len(stoi), embed_dim=args.embed_dim, rnn_units=args.rnn_units)
    ck = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(args.out_dir, 'best.keras'),
        monitor='val_sparse_categorical_accuracy',
        save_best_only=True,
        save_weights_only=False
    )
    es = keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=3, restore_best_weights=True)
    history = model.fit(
        ds.take( int(0.9 * len(list(ds))) ),
        validation_data=ds.skip( int(0.9 * len(list(ds))) ),
        epochs=args.epochs,
        callbacks=[ck, es]
    )
    model.save(os.path.join(args.out_dir, 'final.keras'))

if __name__ == '__main__':
    main()
