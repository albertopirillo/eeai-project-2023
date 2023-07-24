import os
from pathlib import Path
from typing import Literal
import numpy as np
import plotly.express as px
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tqdm import tqdm
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
        print('Number of train batches:', int(self.train.cardinality()))
        print('Number of validation batches:', int(self.validation.cardinality()))
        print('Number of test batches:', int(self.test.cardinality()))


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
        self.test = process(self.test)


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


    def export(self, split: subset, save_path: str) -> None:
        if split == 'train': dataset = self.train
        elif split == 'validation': dataset = self.validation
        elif split == 'test': dataset = self.test
        else: return

        # Create class folders if they don't exist
        for name in self.class_labels:
            if not os.path.exists(f'{save_path}/{name}'):
                os.makedirs(f'{save_path}/{name}')

        # Save the images
        count_written = {class_name: 0 for class_name in self.class_labels}
        for image, label in tqdm(dataset.unbatch()):
            class_name = self.class_labels[int(label)]
            count_written[class_name] += 1
            path = f'{save_path}/{class_name}/{class_name}{count_written[class_name]}.jpg'
            keras.preprocessing.image.save_img(path, image, data_format='channels_last', file_format='JPEG')

        print('Saving complete.')
