from datetime import datetime
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from typing import Literal, Any
import matplotlib.pyplot as plt


WEIGHTS_PATH: Path = Path('weights/MobileNetV1.0_2.96x96.color.bsize_96.lr_0_05.epoch_170.val_loss_3.61.val_accuracy_0.27.hdf5')
IMAGE_SHAPE: tuple[int, int, int] = (96, 96, 3)
WIDTH_MULTIPLIER = 0.2


class Model:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.fit_history = None
        self.fine_tuning_history = None
        self.base_model = keras.applications.MobileNet(
            include_top=True,
            weights = WEIGHTS_PATH,
            alpha=WIDTH_MULTIPLIER,
            input_shape=IMAGE_SHAPE
        )
        # Remove the classification head of the model
        self.base_model = keras.Sequential(self.base_model.layers[:-5], name=self.base_model.name)
        # Freeze the base model
        self.base_model.trainable = False

        # Build the final model
        self.model = keras.Sequential([
            # Input layer
            keras.Input(shape=IMAGE_SHAPE, name='first_layer'),
            # Resizing layer
            # keras.layers.Resizing(96, 96, interpolation='nearest', name='resize'),
            # Pre-processing for MobileNetV1
            keras.layers.Lambda(tf.keras.applications.mobilenet.preprocess_input, name='preprocessing'),
            # MobileNetV1
            self.base_model,
            # Convert tensors to vectors
            keras.layers.GlobalMaxPooling2D(name='pooling'),
            # Fully-connected classifier
            keras.layers.Dense(num_classes, name='classifier')
        ])


    def create_callbacks(self, log_dir: Path, verbose: int = 1) -> list[keras.callbacks]:
        # Custom callback to stop training at 99% accuracy
        def check_accuracy(_, logs):
            if logs.get('val_accuracy') >= 0.99:
                self.model.stop_training = True

        return [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.01,
                patience=10,
                restore_best_weights=True,
                verbose=verbose
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                min_delta=0.025,
                factor=0.1,
                patience=3,
                min_lr=1e-7,
                verbose=verbose
            ),
            keras.callbacks.TerminateOnNaN(),
            keras.callbacks.LambdaCallback(on_epoch_end=check_accuracy),
            # Enable TensorBoard to monitor training
            keras.callbacks.TensorBoard(log_dir=log_dir / datetime.now().strftime("%d-%m-%H-%M"), histogram_freq=1)
        ]


    def summary(self, model: Literal['base', 'full']) -> None:
        if model == 'base': self.base_model.summary()
        elif model == 'full': self.model.summary()
        else: return


    def plot_model(self, dpi: int = 300) -> Any:
        return keras.utils.plot_model(self.model, to_file='model.png', show_shapes=True, show_layer_names=True, dpi=dpi)


    def compile(self, learning_rate: float) -> None:
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                           loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])


    def fit(self, train_ds: tf.data.Dataset, validation_ds: tf.data.Dataset, epochs: int, log_dir: Path) -> None:
        self.fit_history = self.model.fit(train_ds, validation_data=validation_ds, epochs=epochs, callbacks=self.create_callbacks(log_dir / 'fit'))


    def fine_tune(self, train_ds: tf.data.Dataset, validation_ds: tf.data.Dataset, epochs: int, learning_rate: float, log_dir: Path) -> None:
        # Unfreeze the base model
        self.base_model.trainable = True
        # Recompile the model to apply the changes
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                           loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])
        self.fine_tuning_history = self.model.fit(train_ds, validation_data=validation_ds, epochs=epochs,
                                                  callbacks=self.create_callbacks(log_dir / 'fine-tuning'))


    def plot_history(self, phase: Literal['fit', 'fine_tuning']) -> None:
        if phase == 'fit': history = self.fit_history
        elif phase == 'fine_tuning': history = self.fine_tuning_history
        else: return

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy')
        plt.plot(loss, label='Training Loss')
        plt.plot(val_loss, label='Validation Loss')

        plt.subplot(1, 2, 2)
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.ylim([min(plt.ylim()), 1])


    def save(self, save_path: Path) -> None:
        self.model.save(save_path / 'asl_mobilenet_tuned.keras', save_format='keras')
        self.model.save(save_path / 'asl_mobilenet_tuned', save_format='tf')
