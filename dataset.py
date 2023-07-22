from pathlib import Path
from typing import Literal
import numpy as np
import plotly.express as px
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.python.data import AUTOTUNE


ASL_PATH: Path = Path('data/asl_alphabet_train/asl_alphabet_train')
ASL_REAL_PATH: Path= Path('data/asl_alphabet_real/asl_alphabet_real')
LABELS: list[str, ...] = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
SEED: int = 42
CROP_RATIO: float = 0.96
IMAGE_SHAPE: tuple[int, int] = (96, 96)
subset = Literal['train', 'validation', 'test']


class Dataset:
    def __init__(self, split_threshold: float, batch_size: int) -> None:
        self.batch_size: int = batch_size
        self.split_threshold: float = split_threshold
        self.class_labels: list[str, ...]  = LABELS
        self.class_mapping: dict[int, str] =  {i:label for i, label in enumerate(LABELS)}
        self.train: tf.data.Dataset = keras.utils.image_dataset_from_directory(ASL_PATH, batch_size=batch_size, validation_split=split_threshold, subset='training', seed=SEED, class_names=LABELS)
        self.validation: tf.data.Dataset = keras.utils.image_dataset_from_directory(ASL_PATH, batch_size=batch_size, validation_split=split_threshold, subset='validation', seed=SEED, class_names=LABELS)
        self.test: tf.data.Dataset = keras.utils.image_dataset_from_directory(ASL_REAL_PATH, batch_size=batch_size)


    def print_num_batches(self) -> None:
        print('Number of train batches:', int(tf.data.experimental.cardinality(self.train)))
        print('Number of validation batches:', int(tf.data.experimental.cardinality(self.validation)))
        print('Number of test batches:', int(tf.data.experimental.cardinality(self.test)))


    def select_classes(self, classes: list[str, ...]) -> None:
        # Update class labels and mappings
        self.class_labels.append('no gesture')
        self.class_mapping = {i:label for i, label in enumerate(self.class_labels)}
        inverted_dict = {label: i for i, label in enumerate(self.class_labels)}

        # "nothing" and "no gesture" classes must always be present
        if 'nothing' not in classes:
            classes.append('nothing')
        classes.append('no gesture')

        # Convert the class labels to indices
        classes_indices = [inverted_dict[label] for label in classes]

        # Filter out the classes
        class_filter = lambda _, y: tf.reduce_any(tf.equal(y, classes_indices))
        self.train = self.train.unbatch().filter(class_filter).batch(self.batch_size)
        self.validation = self.validation.unbatch().filter(class_filter).batch(self.batch_size)
        self.test = self.test.unbatch().filter(class_filter).batch(self.batch_size)


    def preprocess(self, resize: bool) -> None:
        # Enable prefetching to increase performance
        self.train = self.train.prefetch(buffer_size=AUTOTUNE)
        self.validation = self.validation.prefetch(buffer_size=AUTOTUNE)
        self.test = self.test.prefetch(buffer_size=AUTOTUNE)

        # Crop to remove the blue border
        crop = lambda x, y: (tf.image.central_crop(x, CROP_RATIO), y)
        self.train = self.train.map(crop, num_parallel_calls=AUTOTUNE)
        self.validation = self.validation.map(crop, num_parallel_calls=AUTOTUNE)

        # Resize if specified
        if resize:
            resize_image = lambda x, y: (tf.image.resize(x, IMAGE_SHAPE), y)
            self.train = self.train.map(resize_image, num_parallel_calls=AUTOTUNE)
            self.validation = self.validation.map(resize_image, num_parallel_calls=AUTOTUNE)
            self.test = self.test.map(resize_image, num_parallel_calls=AUTOTUNE)


    def cache(self) -> None:
        self.train = self.train.cache()
        self.validation = self.validation.cache()
        self.test = self.test.cache()


    def visualize_images(self, split: subset):
        if split == 'train': dataset = self.train
        elif split == 'validation': dataset = self.validation
        elif split == 'test': dataset = self.test
        else: return

        plt.figure(figsize=(10, 10))
        for i, (image, label) in enumerate(dataset.unbatch().take(9)):
            _ = plt.subplot(3, 3, i + 1)
            plt.imshow(image.numpy().astype('uint8'))
            plt.title(self.class_mapping[int(label)])
            plt.axis("off")
        plt.show()


    def plot_class_distribution(self, split: subset, y_lim: int = 3100) -> None:
        if split == 'train': dataset = self.train
        elif split == 'validation': dataset = self.validation
        elif split == 'test': dataset = self.test
        else: return

        labels = np.concatenate([y for x, y in dataset], axis=0)
        labels_str = np.array([self.class_mapping[i] for i in labels])
        fig = px.histogram(x=labels_str)
        fig.update_layout(
            title_text='Class distribution',
            xaxis_title_text='Class name',
            yaxis_title_text='Count',
            bargap=0.5,
            yaxis_range=[0, y_lim]
        )
        fig.show()