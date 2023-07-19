import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn import metrics
import plotly.express as px
from tqdm.notebook import tqdm

class Evaluator:
    def __init__(self, model: keras.Model, dataset: tf.data.Dataset, class_labels: tuple[str, ...]) -> None:
        self.model = model
        self.dataset = dataset
        self.class_labels = class_labels
        self.true_labels, self.pred_labels = self.compute_labels()


    def compute_labels(self) -> tuple[np.ndarray, np.ndarray]:
        true_labels = np.array([])
        pred_labels = np.array([])

        for batch in tqdm(self.dataset):
            # Tuple unpacking
            images, t_labels = batch

            # Compute new labels
            p_labels = self.model.predict_on_batch(images)
            p_labels = np.argmax(p_labels, axis=1)

            # Concatenate in a single vector
            true_labels = np.concatenate([true_labels, t_labels])
            pred_labels = np.concatenate([pred_labels, p_labels])

        return true_labels, pred_labels


    def evaluate(self):
        loss, accuracy = self.model.evaluate(self.dataset)
        print(f'Loss function: {loss:.3f}')
        print(f'Accuracy: {accuracy:.2%}')


    def classification_report(self) -> None:
        print(metrics.classification_report(self.true_labels, self.pred_labels, target_names=self.class_labels))


    def confusion_matrix(self, size: int) -> None:
        cm = metrics.confusion_matrix(self.true_labels, self.pred_labels)
        fig = px.imshow(cm, x=self.class_labels, y=self.class_labels, text_auto=True, width=size, height=size,color_continuous_scale='blues')
        fig.update_layout(
            title_text='Multiclass confusion matrix',
            xaxis_title_text='Actual class',
            yaxis_title_text='Predicted class',
        )
        fig.show()


    def error_per_class(self) -> None:
        cm = metrics.confusion_matrix(self.true_labels, self.pred_labels)
        total_errors = cm.sum(axis=1) - cm.diagonal()
        fig = px.bar(x=self.class_labels, y=total_errors, orientation='v')
        fig.update_layout(
            title_text='Total miss-classifications per class',
            xaxis_title_text='Class name',
            yaxis_title_text='Count',
            bargap=0.3,
        )
        fig.show()


    def relative_errors(self) -> None:
        cm = metrics.confusion_matrix(self.true_labels, self.pred_labels)
        correct_percentage =  cm.diagonal() / cm.sum(axis=1)
        rel_errors = 1 - correct_percentage
        fig = px.bar(x=self.class_labels, y=rel_errors, orientation='v')
        fig.update_layout(
            title_text='Percentage of miss-classifications per class',
            xaxis_title_text='Class name',
            yaxis_title_text='Count',
            bargap=0.3,
        )
        fig.show()