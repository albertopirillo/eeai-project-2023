import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from tensorflow import keras
from sklearn import metrics
import plotly.express as px
from tqdm import tqdm


class Evaluator:
    def __init__(self, model: keras.Model | tf.lite.Interpreter, dataset: tf.data.Dataset,
                 class_labels: tuple[str, ...], quantized: bool = False) -> None:
        self.model = model
        self.dataset = dataset
        self.class_labels = class_labels
        self.quantized = quantized
        if self.quantized:
            self.true_labels, self.pred_labels = self.compute_labels_quantized()
        else:
            self.true_labels, self.pred_labels = self.compute_labels()


    def compute_labels(self) -> tuple[np.ndarray, np.ndarray]:
        true_labels = np.array([])
        pred_labels = np.array([])

        for batch in tqdm(self.dataset):
            # Tuple unpacking
            images, t_labels = batch

            # Compute the predicted labels
            p_labels = self.model.predict_on_batch(images)
            p_labels = np.argmax(p_labels, axis=1)

            # Concatenate in a single vector
            true_labels = np.concatenate([true_labels, t_labels])
            pred_labels = np.concatenate([pred_labels, p_labels])

        return true_labels, pred_labels


    def compute_labels_quantized(self) -> tuple[np.ndarray, np.ndarray]:
        true_labels = np.array([])
        pred_labels = np.array([])

        # Set the input shape of the interpreter
        input_details = self.model.get_input_details()[0]
        self.model.resize_tensor_input(input_details['index'], [1, 96, 96, 3])
        self.model.allocate_tensors()

        for batch in tqdm(self.dataset.unbatch().batch(1)):
            # Tuple unpacking
            images, t_labels = batch

            # Quantize the input
            input_details = self.model.get_input_details()[0]
            input_scale, input_zero_point = input_details["quantization"]
            images = images / input_scale + input_zero_point
            images = images.numpy().astype("uint8")

            # Compute the predicted labels
            self.model.set_tensor(input_details['index'], images)
            self.model.invoke()
            output_details = self.model.get_output_details()[0]
            p_labels = self.model.get_tensor(output_details['index'])
            p_labels = np.argmax(p_labels, axis=1)

            # Concatenate in a single vector
            true_labels = np.concatenate([true_labels, t_labels])
            pred_labels = np.concatenate([pred_labels, p_labels])

        return true_labels, pred_labels


    def evaluate(self):
        accuracy = metrics.accuracy_score(self.true_labels, self.pred_labels)
        ohe_true_labels = LabelBinarizer().fit_transform(self.true_labels)
        ohe_pred_labels = LabelBinarizer().fit_transform(self.pred_labels)
        loss = metrics.log_loss(ohe_true_labels, ohe_pred_labels)
        print(f'Loss function: {loss:.3f}')
        print(f'Accuracy: {accuracy:.2%}')


    def classification_report(self) -> None:
        print(metrics.classification_report(self.true_labels, self.pred_labels, target_names=self.class_labels))


    def confusion_matrix(self, size: int = 1000, save_path: str = None) -> None:
        cm = metrics.confusion_matrix(self.true_labels, self.pred_labels)
        fig = px.imshow(cm, x=self.class_labels, y=self.class_labels, text_auto=True, width=size, height=size,
                        color_continuous_scale='blues')
        fig.update_layout(
            title_text='Multiclass confusion matrix',
            xaxis_title_text='Actual class',
            yaxis_title_text='Predicted class',
        )
        if save_path is not None:
            fig.write_image(save_path, scale=2.5)
        fig.show()



    def error_per_class(self, save_path: str = None) -> None:
        cm = metrics.confusion_matrix(self.true_labels, self.pred_labels)
        total_errors = cm.sum(axis=1) - cm.diagonal()
        fig = px.bar(x=self.class_labels, y=total_errors, orientation='v')
        fig.update_layout(
            title_text='Total miss-classifications per class',
            xaxis_title_text='Class name',
            yaxis_title_text='Count',
            bargap=0.3,
        )
        if save_path is not None:
            fig.write_image(save_path, scale=2.5)
        fig.show()


    def relative_errors(self, save_path: str) -> None:
        self.get_relative_errors(self.true_labels, self.pred_labels, self.class_labels, save_path)


    @staticmethod
    def get_relative_errors(true_labels, pred_labels, class_labels, save_path: str = None) -> None:
        cm = metrics.confusion_matrix(true_labels, pred_labels)
        correct_percentage = cm.diagonal() / cm.sum(axis=1)
        rel_errors = 1 - correct_percentage
        fig = px.bar(x=class_labels, y=rel_errors, orientation='v')
        fig.update_layout(
            title_text='Percentage of miss-classifications per class',
            xaxis_title_text='Class name',
            yaxis_title_text='Count',
            bargap=0.3,
        )
        if save_path is not None:
            fig.write_image(save_path, scale=2.5)
        fig.show()
