import os
import string
import re
import struct
import random

import numpy as np
import tensorflow as tf
from more_itertools import flatten
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
from tqdm import tqdm

from src.transformer_layers import *


# Prepare dataset <br>
# input -> input sequences <br>
# output -> event sequences covered for those inputs<br>
def prepare_dataset(datafolder="data/fuzz-results"):
    # prepare inputs
    inputs = []
    pbar = tqdm(total=50000)
    with open(os.path.join(datafolder, "events.bin"), "rb") as f:
        while length := f.read(4):
            n = struct.unpack(">i", length)[0]
            pbar.update(1)
            inputs.append(" ".join(map(str, flatten(struct.iter_unpack(">h", f.read(n * 2))))))
    pbar.close()

    # prepare outputs
    outputs = []
    corpus = os.path.join(datafolder, "corpus")
    for fname in tqdm(sorted(os.listdir(corpus))):
        with open(os.path.join(corpus, fname), "rb") as file:
            outputs.append(" ".join(map(str, flatten(struct.iter_unpack(">B", file.read())))))

    print(inputs[0])
    print(len(inputs))
    print(outputs[0])
    print(len(outputs))

    return zip(inputs, outputs)


def prepare_random(num=50000):
    inputs, outputs = [], []
    pbar = tqdm(total=50000)
    while len(inputs) < num:
        inputs.append(" ".join([str(random.randint(1, 30)) for i in range(random.randint(1, 30))]))
        outputs.append(" ".join([str(random.randint(1, 30)) for i in range(random.randint(1, 30))]))
        pbar.update(1)
    pbar.close()
    return zip(inputs, outputs)


# zip_pairs = prepare_random()

zip_pairs = prepare_dataset()

text_pairs = []
for inp, out in zip_pairs:
    out = "[start] " + out + " [end]"
    text_pairs.append((inp, out))

random.shuffle(text_pairs)
num_val_samples = int(0.15 * len(text_pairs))
num_train_samples = len(text_pairs) - 2 * num_val_samples
train_pairs = text_pairs[:num_train_samples]
val_pairs = text_pairs[num_train_samples: num_train_samples + num_val_samples]
test_pairs = text_pairs[num_train_samples + num_val_samples:]

print(f"{len(text_pairs)} total pairs")
print(f"{len(train_pairs)} training pairs")
print(f"{len(val_pairs)} validation pairs")
print(f"{len(test_pairs)} test pairs")

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")

vocab_size = 100
sequence_length = 20
batch_size = 64


def custom_standardization(input_string):
    lowercase = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercase, "[%s]" % re.escape(strip_chars), "")


eng_vectorization = TextVectorization(
    max_tokens=vocab_size, output_mode="int", output_sequence_length=sequence_length,
)
spa_vectorization = TextVectorization(
    max_tokens=vocab_size,
    output_mode="int",
    output_sequence_length=sequence_length + 1,
    standardize=custom_standardization,
)
train_eng_texts = [pair[0] for pair in train_pairs]
train_spa_texts = [pair[1] for pair in train_pairs]
eng_vectorization.adapt(train_eng_texts)
spa_vectorization.adapt(train_spa_texts)


def format_dataset(eng, spa):
    eng = eng_vectorization(eng)
    spa = spa_vectorization(spa)
    return {"encoder_inputs": eng, "decoder_inputs": spa[:, :-1], }, spa[:, 1:]


def make_dataset(pairs):
    eng_texts, spa_texts = zip(*pairs)
    eng_texts = list(eng_texts)
    spa_texts = list(spa_texts)
    dataset = tf.data.Dataset.from_tensor_slices((eng_texts, spa_texts))
    dataset = dataset.batch(batch_size)
    dataset = dataset.map(format_dataset)
    return dataset.shuffle(2048).prefetch(16).cache()


train_ds = make_dataset(train_pairs)
val_ds = make_dataset(val_pairs)

for inputs, targets in train_ds.take(1):
    print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
    print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
    print(f"targets.shape: {targets.shape}")

embed_dim = 256
latent_dim = 2048
num_heads = 8

encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(encoder_inputs)
encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads)(x)
encoder = keras.Model(encoder_inputs, encoder_outputs)

decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
x = PositionalEmbedding(sequence_length, vocab_size, embed_dim)(decoder_inputs)
x = TransformerDecoder(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
x = layers.Dropout(0.5)(x)
decoder_outputs = layers.Dense(vocab_size, activation="softmax")(x)
decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs)

decoder_outputs = decoder([decoder_inputs, encoder_outputs])
transformer = keras.Model(
    [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
)

epochs = 1  # This should be at least 30 for convergence

transformer.summary()
transformer.compile(
    "rmsprop", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

transformer.fit(train_ds, epochs=epochs, validation_data=val_ds)

# Testing

spa_vocab = spa_vectorization.get_vocabulary()
spa_index_lookup = dict(zip(range(len(spa_vocab)), spa_vocab))
max_decoded_sentence_length = 20


def decode_sequence(input_sentence):
    tokenized_input_sentence = eng_vectorization([input_sentence])
    decoded_sentence = "[start]"
    for i in range(max_decoded_sentence_length):
        tokenized_target_sentence = spa_vectorization([decoded_sentence])[:, :-1]
        predictions = transformer([tokenized_input_sentence, tokenized_target_sentence])

        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = spa_index_lookup[sampled_token_index]
        decoded_sentence += " " + sampled_token

        if sampled_token == "[end]":
            break
    return decoded_sentence


test_eng_texts = [pair[0] for pair in test_pairs]
for _ in range(30):
    input_sentence = random.choice(test_eng_texts)
    translated = decode_sequence(input_sentence)
