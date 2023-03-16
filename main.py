import os
import struct

import numpy as np
from more_itertools import flatten
from tqdm import tqdm
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

from src.transformer_layers import *


def prepare_dataset(datafolder="data/fuzz-results"):
    inputs = []
    outputs = []
    pbar = tqdm(total=50000)
    with open(os.path.join(datafolder, "events.bin"), "rb") as f:
        while length := f.read(4):
            n = struct.unpack(">i", length)[0]
            pbar.update(1)
            inputs.append(list(flatten(struct.iter_unpack(">h", f.read(n * 2)))))
    pbar.close()
    corpus = os.path.join(datafolder, "corpus")
    for filename in tqdm(sorted(os.listdir(corpus))):
        with open(os.path.join(corpus, filename), 'rb') as file:
            outputs.append(list(file.read()))
    return train_test_split(inputs, outputs)


def main():
    maxlen = 1000
    x_train, x_val, y_train, y_val = prepare_dataset()
    print(len(x_train), "Training sequences")
    print(len(x_val), "Validation sequences")
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

    y_train = keras.preprocessing.sequence.pad_sequences(y_train)
    y_val = keras.preprocessing.sequence.pad_sequences(y_val)

    num_cov_points = 100  # total number of branches in the PuT
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, num_cov_points, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val)
    )
    print("X")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()
