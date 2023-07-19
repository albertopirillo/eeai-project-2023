import os
from pathlib import Path
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(format='%(asctime)s >> %(message)s', datefmt='%H:%M:%S', level=logging.INFO)

import tensorflow as tf
from dataset import Dataset
from model import Model
from evaluator import Evaluator

LOGS_PATH = Path('logs')
SAVE_PATH = Path('models')

if __name__ == '__main__':
    logging.info(f'Detected GPUs: {tf.config.list_physical_devices("GPU")}')
    logging.info('Loading dataset...')
    data = Dataset(split_threshold=0.2, batch_size=32)
    data.preprocess(resize=True)
    model = Model(num_classes=29)

    # Fit the model
    model.compile(learning_rate=1e-3)
    logging.info('Starting training...')
    model.fit(data.train, data.validation, epochs=30, log_dir=LOGS_PATH)
    model.plot_history('fit')
    logging.info('Training complete.')

    # Fine-tune the model
    logging.info('Starting fine-tuning...')
    model.fine_tune(data.train, data.validation, epochs=30, learning_rate=0.5e-4, log_dir=LOGS_PATH)
    model.plot_history('fine_tuning')
    logging.info('Fine-tuning complete.')
    model.save(SAVE_PATH)
    logging.info('Model saved.')

    # Evaluate the model
    logging.info('Starting model evaluation...')
    valid_evaluator = Evaluator(model.model, data.validation, data.class_labels)
    test_evaluator = Evaluator(model.model, data.test, data.class_labels)
    valid_evaluator.evaluate()
    test_evaluator.evaluate()

    # Confusion matrix on the validation dataset
    valid_evaluator.confusion_matrix(size=1000)
    valid_evaluator.relative_errors()
    logging.info('Evaluation complete.')
