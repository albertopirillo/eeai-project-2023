import logging
import os

import numpy as np
from sklearn import metrics
from tqdm import tqdm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
logging.basicConfig(format='%(asctime)s >> %(message)s', datefmt='%H:%M:%S', level=logging.INFO)

import tensorflow as tf
from dataset import Dataset
from evaluator import Evaluator

SAVED_MODEL_PATH = 'models/asl_mobilenet_tuned'
TFLITE_MODEL_PATH = 'models/asl_mobilenet_full_quant.tflite'


def get_details(interpreter: tf.lite.Interpreter):
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    print(f'Input shape: {input_details["shape"]}')
    print(f'Input datatype: {input_details["dtype"]}')
    print(f'Output shape: {output_details["shape"]}')
    print(f'Output datatype: {output_details["dtype"]}')

def resize_input_shape(interpreter: tf.lite.Interpreter):
  # Resize input shape for dynamic shape model and allocate tensors
  input_details = interpreter.get_input_details()[0]
  interpreter.resize_tensor_input(input_details['index'], [32, 96, 96, 3])
  interpreter.allocate_tensors()
  print(f'Resized input shape: {interpreter.get_input_details()[0]["shape"]}')


if __name__ == '__main__':
    logging.info(f'Detected GPUs: {tf.config.list_physical_devices("GPU")}')
    logging.info('Loading dataset...')
    data = Dataset(split_threshold=0.2, batch_size=32)
    data.preprocess(resize=True)

    def representative_data_gen():
        for image, _ in data.validation.unbatch().batch(1).take(1000):
            yield [image]

    # Post Training Quantization
    logging.info('Starting model quantization...')
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_PATH)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Full-integer quantization
    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    # Save the TFLite model
    tflite_model = converter.convert()
    logging.info('Quantization complete.')
    with open(TFLITE_MODEL_PATH, 'wb') as f:
        print(f"\nModel saved. Bytes written: {f.write(tflite_model) / 1024} KB")

    # Evaluate the model
    interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
    get_details(interpreter)
    resize_input_shape(interpreter)
    logging.info('Starting model evaluation...')

    true_labels = np.array([])
    pred_labels = np.array([])
    for batch in tqdm(data.validation):
        # Tuple unpacking
        images, t_labels = batch

        if images.shape[0] == data.batch_size:
            # Quantize the input
            input_details = interpreter.get_input_details()[0]
            input_scale, input_zero_point = input_details["quantization"]
            images = images / input_scale + input_zero_point
            images = images.numpy().astype("uint8")

            # Compute the predicted labels
            interpreter.set_tensor(input_details['index'], images)
            interpreter.invoke()
            output_details = interpreter.get_output_details()[0]
            p_labels = interpreter.get_tensor(output_details['index'])
            p_labels = np.argmax(p_labels, axis=1)

            # Concatenate in a single vector
            true_labels = np.concatenate([true_labels, t_labels])
            pred_labels = np.concatenate([pred_labels, p_labels])

    # Compute the accuracy
    logging.info('Evaluation complete.')
    print(f'Accuracy of the quantized model: {metrics.accuracy_score(true_labels, pred_labels)}')

    # Plot the relative errors
    Evaluator.get_relative_errors(true_labels, pred_labels, data.class_labels)
