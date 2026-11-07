"""Task 2.1 - tf.data input pipeline.

Stages:  split CSV -> (path, label) -> decode + resize -> cache -> [shuffle] -> [augment]
         -> batch -> prefetch
Augmentation is applied to the TRAINING set only; val/test images must stay untouched,
otherwise their scores would describe distorted images instead of real inputs.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import keras
from keras import layers

from visionguard.config import BATCH_SIZE, ROOT, SEED, SPLITS_DIR
from visionguard.preprocess import load_for_model


def read_split(name: str) -> pd.DataFrame:
    return pd.read_csv(SPLITS_DIR / f"{name}.csv")


def make_augmenter(seed: int = SEED) -> keras.Sequential:
    """Mild, label-preserving changes. Nothing that alters how 'AI-like' an image looks
    (no blur, no noise, no JPEG re-compression), because those are the signals we classify."""
    return keras.Sequential([
        layers.RandomFlip("horizontal", seed=seed),
        layers.RandomRotation(0.05, fill_mode="reflect", seed=seed),   # +-18 degrees
        layers.RandomZoom(0.10, fill_mode="reflect", seed=seed),
        layers.RandomContrast(0.10, seed=seed),
    ], name="augmentation")


def make_dataset(split: str, img_size: int, batch_size: int = BATCH_SIZE,
                 augment: bool = False, shuffle: bool = False, seed: int = SEED) -> tf.data.Dataset:
    df = read_split(split)
    paths = [str(ROOT / p) for p in df["rel_path"]]
    labels = df["label"].astype("float32").to_numpy()

    def _load(path):
        # Pillow decoding (via numpy_function) instead of tf.io.decode_image: it handles WebP,
        # animated GIF, transparency and EXIF rotation, and it is the SAME code the API uses.
        return load_for_model(path.decode(), img_size)

    def decode(path, label):
        img = tf.numpy_function(_load, [path], tf.float32)
        img.set_shape((img_size, img_size, 3))
        return img, label

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()  # decode each file once; later epochs read from memory
    if shuffle:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.batch(batch_size)
    if augment:
        aug = make_augmenter(seed)
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)  # prepare the next batch while the model trains


def labels_of(split: str) -> np.ndarray:
    return read_split(split)["label"].to_numpy()


def paths_of(split: str) -> list:
    return read_split(split)["rel_path"].tolist()
