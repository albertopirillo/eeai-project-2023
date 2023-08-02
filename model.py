from datetime import datetime
from pathlib import Path
from typing import Literal, Any
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

WEIGHTS_PATH: Path = Path('weights/MobileNetV1.0_25.128x128.color.h5')
IMAGE_SHAPE: tuple[int, int, int] = (96, 96, 3)
WIDTH_MULTIPLIER: float = 0.25
FE_DROPOUT_RATE: float = 1e-2
DENSE_DROPOUT_RATE: float = 0.3


class Model:
    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self.fit_history = None
        self.fine_tuning_history = None
        self.base_model = keras.applications.MobileNet(
            include_top=True,
            weights = WEIGHTS_PATH,
            dropout=FE_DROPOUT_RATE,
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
            # MobileNetV1
            self.base_model,
            # Reshape the output of the base model
            keras.layers.Reshape((-1, self.base_model.layers[-1].output.shape[3])),
            # Dropout to prevent overfitting
            keras.layers.Dropout(DENSE_DROPOUT_RATE),
            # Flatten the output tensors into a single vector
            keras.layers.Flatten(),
            # Fully-connected classifier
            keras.layers.Dense(num_classes, name='classifier'),
            # Softmax to convert logits to probabilities
            keras.layers.Softmax()
        ])


    def create_callbacks(self, log_dir: Path, verbose: int = 1) -> list[keras.callbacks]:
        # Custom callback to stop training at 99% accuracy
        def check_accuracy(_, logs):
            if logs.get('accuracy') >= 0.99 and logs.get('val_accuracy') >= 0.99:
                print('\nEarly stopping at 99% accuracy.')
                self.model.stop_training = True

        return [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                min_delta=0.01,
                patience=30,
                restore_best_weights=True,
                verbose=verbose
            ),
            # keras.callbacks.ReduceLROnPlateau(
            #     monitor="val_loss",
            #     min_delta=0.025,
            #     factor=0.1,
            #     patience=10,
            #     min_lr=1e-8,
            #     verbose=verbose
            # ),
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
                           loss=keras.losses.SparseCategoricalCrossentropy(),
                           metrics=['accuracy'])


    def fit(self, train_ds: tf.data.Dataset, validation_ds: tf.data.Dataset, epochs: int, log_dir: Path) -> None:
        self.fit_history = self.model.fit(train_ds, validation_data=validation_ds, epochs=epochs, callbacks=self.create_callbacks(log_dir / 'fit'))


    def fine_tune(self, train_ds: tf.data.Dataset, validation_ds: tf.data.Dataset, epochs: int, learning_rate: float, log_dir: Path) -> None:
        # Unfreeze the base model
        self.base_model.trainable = True
        # Recompile the model to apply the changes
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                           loss=keras.losses.SparseCategoricalCrossentropy(),
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
        plt.semilogy(loss, label='Training Loss')
        plt.semilogy(val_loss, label='Validation Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.plot(acc, label='Training Accuracy')
        plt.plot(val_acc, label='Validation Accuracy')
        plt.ylim([min(plt.ylim()), 1])
        plt.legend()

        plt.show()


    def quantize(self, representative_data: tf.data.Dataset, save_path: Path) -> None:
        def representative_data_gen():
            for image, _ in representative_data.unbatch().batch(1).take(1000):
                yield [image]

        # Full-integer quantization
        self.model.save('temp/model', save_format='tf')
        converter = tf.lite.TFLiteConverter.from_saved_model('temp/model')
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_data_gen
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.uint8
        converter.inference_output_type = tf.uint8

        # Save the quantized model
        tflite_model = converter.convert()
        with open(save_path / 'asl_mobilenet_quant.tflite', 'wb') as f:
            f.write(tflite_model)

    def save(self, save_path: Path) -> None:
        self.model.save(save_path / 'asl_mobilenet_tuned.keras', save_format='keras')
        self.model.save(save_path / 'asl_mobilenet_tuned', save_format='tf')
