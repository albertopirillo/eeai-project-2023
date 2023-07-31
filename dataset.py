import math
import os
from pathlib import Path
import random
from typing import Literal
import numpy as np
import plotly.express as px
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tqdm import tqdm
from tensorflow.python.data import AUTOTUNE


LABELS: list[str, ...] = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'nothing', 'O', 'P', 'Q', 'R', 'S', 'space', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
SEED: int = 42
CROP_RATIO: float = 0.96
IMAGE_SHAPE: tuple[int, int] = (96, 96)
subset = Literal['train', 'validation', 'test']


class Dataset:
    def __init__(self, split_threshold: float, batch_size: int, path: Path | str, labels: list[str, ...] = None) -> None:
        self.batch_size: int = batch_size
        self.split_threshold: float = split_threshold

        if labels is None: labels = LABELS
        self.class_labels: list[str, ...]  = labels
        self.class_mapping: dict[int, str] =  {i:label for i, label in enumerate(labels)}

        if split_threshold == 0:
            self.train: tf.data.Dataset = keras.utils.image_dataset_from_directory(path, batch_size=batch_size, seed=SEED, class_names=labels)
            self.validation: tf.data.Dataset = keras.utils.image_dataset_from_directory(path, batch_size=batch_size, seed=SEED, class_names=labels)
        else:
            self.train: tf.data.Dataset = keras.utils.image_dataset_from_directory(path, batch_size=batch_size, validation_split=split_threshold, subset='training', seed=SEED, class_names=labels)
            self.validation: tf.data.Dataset = keras.utils.image_dataset_from_directory(path, batch_size=batch_size, validation_split=split_threshold, subset='validation', seed=SEED, class_names=labels)


    def print_num_batches(self) -> None:
        print('Number of train batches:', int(self.train.cardinality()))
        print('Number of validation batches:', int(self.validation.cardinality()))
        # print('Number of test batches:', int(self.test.cardinality()))


    def select_classes(self, classes: list[str, ...]) -> None:
        # "nothing" and "no gesture" classes must always be present
        if 'nothing' not in classes:
            classes.append('nothing')
        new_classes = [label for label in LABELS if label in classes]
        classes = new_classes + ['no gesture']

        # Update class labels and mappings
        inverted_mapping = {label: i for i, label in enumerate(self.class_labels + ['no gesture'])}
        self.class_labels = classes
        self.class_mapping = {i:label for i, label in enumerate(self.class_labels)}

        # Convert the class labels to indices
        classes_indices = [inverted_mapping[label] for label in classes]

        # Helper lambda unctions
        class_filter = lambda _, y: tf.reduce_any(tf.equal(y, classes_indices))
        excluded_classes_filter = lambda _, y: tf.reduce_any(tf.not_equal(y, classes_indices))
        rename_f = lambda x, y: (x, inverted_mapping['no gesture'])

        # Function to perform all operations
        def process(dataset: tf.data.Dataset) -> tf.data.Dataset:
            cardinality = dataset.cardinality() * self.batch_size
            one_class_cardinality = dataset.cardinality() * self.batch_size // len(LABELS)
            dataset = dataset.unbatch()
            # Generate the 'no gesture' class by sampling randomly the excluded classes
            dataset_excluded = dataset.filter(excluded_classes_filter).take(one_class_cardinality).map(rename_f, num_parallel_calls=AUTOTUNE)
            # Filter the classes and concatenate the two datasets
            dataset_included = dataset.filter(class_filter)
            dataset = dataset_included.concatenate(dataset_excluded)

            # Map into the new labels
            new_labels_list = [inverted_mapping[i] if i in classes else -1 for i in (LABELS + ['no gesture'])]
            count = 0
            for i in range(len(new_labels_list)):
                if new_labels_list[i] != -1:
                    new_labels_list[i] = count
                    count += 1

            table = tf.lookup.StaticHashTable(
                initializer=tf.lookup.KeyValueTensorInitializer(
                    keys=tf.constant(list(range(len(LABELS) + 1)), dtype=tf.int32),
                    values=tf.constant(new_labels_list, dtype=tf.int32)
                ),
                default_value=tf.constant(-5, dtype=tf.int32)
            )

            dataset = dataset.map(lambda x, y: (x, table.lookup(y)), num_parallel_calls=AUTOTUNE)

            # Shuffle and batch the dataset
            dataset = dataset.shuffle(cardinality * (len(classes) + 1), seed=SEED)
            return dataset.batch(self.batch_size)

        # Apply all operations on the 3 splits
        self.train = process(self.train)
        self.validation = process(self.validation)
        # self.test = process(self.test)


    def preprocess(self, resize: bool, crop: bool) -> None:
        # Enable prefetching to increase performance
        self.train = self.train.prefetch(buffer_size=AUTOTUNE)
        self.validation = self.validation.prefetch(buffer_size=AUTOTUNE)
        # self.test = self.test.prefetch(buffer_size=AUTOTUNE)

        # Crop to remove the blue border if specified
        if crop:
            crop = lambda x, y: (tf.image.central_crop(x, CROP_RATIO), y)
            self.train = self.train.map(crop, num_parallel_calls=AUTOTUNE)
            self.validation = self.validation.map(crop, num_parallel_calls=AUTOTUNE)
            # self.test = self.test.map(crop, num_parallel_calls=AUTOTUNE)

        # Resize if specified
        if resize:
            resize_image = lambda x, y: (tf.image.resize(x, IMAGE_SHAPE), y)
            self.train = self.train.map(resize_image, num_parallel_calls=AUTOTUNE)
            self.validation = self.validation.map(resize_image, num_parallel_calls=AUTOTUNE)
            # self.test = self.test.map(resize_image, num_parallel_calls=AUTOTUNE)


    def rescale(self) -> None:
        # Scale pixels between -1 and 1
        scale_pixels = lambda x, y: (keras.applications.mobilenet.preprocess_input(x), y)
        self.train = self.train.map(scale_pixels, num_parallel_calls=AUTOTUNE)
        self.validation = self.validation.map(scale_pixels, num_parallel_calls=AUTOTUNE)
        # self.test = self.test.map(scale_pixels, num_parallel_calls=AUTOTUNE)


    def cache(self) -> None:
        self.train = self.train.cache()
        self.validation = self.validation.cache()
        # self.test = self.test.cache()


    def visualize_images(self, split: subset, num_images: int = 3) -> None:
        if split == 'train': dataset = self.train
        elif split == 'validation': dataset = self.validation
        # elif split == 'test': dataset = self.test
        else: return

        plt.figure(figsize=(12, 12))
        for i, (image, label) in enumerate(dataset.unbatch().take(num_images ** 2)):
            _ = plt.subplot(num_images, num_images, i + 1)
            plt.imshow(image.numpy().astype('uint8'))
            plt.title(self.class_mapping[int(label)])
            plt.axis("off")
        plt.show()


    def plot_class_distribution(self, split: subset, y_lim: int = 3100) -> None:
        if split == 'train': dataset = self.train
        elif split == 'validation': dataset = self.validation
        # elif split == 'test': dataset = self.test
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


    def export(self, split: subset, save_path: str, tag: str, subsample: bool = False) -> None:
        if split == 'train': dataset = self.train
        elif split == 'validation': dataset = self.validation
        # elif split == 'test': dataset = self.test
        else: return

        # Create class folders if they don't exist
        for name in self.class_labels:
            if not os.path.exists(f'{save_path}/{name}'):
                os.makedirs(f'{save_path}/{name}')

        # Save the images
        count_written = {class_name: 0 for class_name in self.class_labels}
        iteration_counter = {class_name: 0 for class_name in self.class_labels}
        for image, label in tqdm(dataset.unbatch()):
            class_name = self.class_labels[int(label)]
            if subsample:
                iteration_counter[class_name] += 1
            if iteration_counter[class_name] % 100 == 0:
                count_written[class_name] += 1
                path = f'{save_path}/{class_name}/{class_name}_{tag}_{count_written[class_name]}.jpg'
                keras.preprocessing.image.save_img(path, image, data_format='channels_last', file_format='JPEG')

        print('Saving complete.')


    def use_augmentation(self) -> None:
        def augment_image(image, label):
            # Flip the image randomly
            image = tf.image.random_flip_left_right(image)

            # Increase the image size, then randomly crop it down to the original dimensions
            resize_factor = random.uniform(1, 1.2)
            new_height = math.floor(resize_factor * IMAGE_SHAPE[0])
            new_width = math.floor(resize_factor * IMAGE_SHAPE[1])
            image = tf.image.resize_with_crop_or_pad(image, new_height, new_width)
            image = tf.image.random_crop(image, size=(IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3))

            # Var the brightness of the image
            image = tf.image.random_brightness(image, max_delta=0.2)

            # Vary the contrast of the image
            # image = tf.image.random_contrast(image, lower=0.3, upper=0.7)

            # Vary the saturation of the image
            # image = tf.image.random_saturation(image, lower=0.3, upper=0.7)

            return image, label

        # Augmentation is applied to the training dataset only
        self.train = self.train.unbatch().map(augment_image, num_parallel_calls=AUTOTUNE).batch(self.batch_size)
