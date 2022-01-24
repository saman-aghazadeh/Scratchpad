import argparse

import tensorflow as tf
import numpy as np

class AgeResNet:
    def __init__(self, model_path, batch_size=1, shape=(120, 120, 3)):
        """Implements TFlite version of AgeResNet 
        Aguments
            model_path (str): path to tflite model
            shape (tuple): input shape of the images (width x height x channels)
        Returns
            AgeResNet Object
        """

        # load model
        self._model = tf.lite.Interpreter(model_path)

        # get input, output details
        self._input_details = self._model.get_input_details()
        self._output_details = self._model.get_output_details()
        print(self._output_details)

        # set shape (maynot work since this model is not dynamic.)
        self._model.resize_tensor_input(self._input_details[0]['index'], 
                                        (batch_size, *shape))

        # allocate memory
        self._model.allocate_tensors()

        # linear bias layer for post processing 
        self._linear_1_bias = np.load('/home/local/ASUAD/sbiookag/guise/age/age/model_current_05_03_2021.npy')#model_path.replace(".tflite", ".npy"))
        
    @staticmethod
    def _sigmoid(x):
        """Sigmoid Operation
        Arguments
            x (ndarray): numpy array
        Returns:
            sigmoid applied on  x
        """
        return 1 / (1 + np.exp(-x))

    def _post_process(self, x):
        """Converts TFLite Incomplete logits to complete logits and probas
        Arguments
            x (ndarray): output of the tflite model
        Returns:
            logits (ndarray): logits of inputs
            probas (ndarray): probabilities over logits
        """
        logits = x + self._linear_1_bias
        
        probas = self._sigmoid(logits)
        
        return logits, probas

    def __call__(self, inputs):
        """Runs inference on inputs
        Arguments
            inputs (ndarray): input image of (1, *shape)
        Returns:
            logits (ndarray): logits of inputs
            probas (ndarray): probabilities over logits
        """
        # set input tensor
        self._model.set_tensor(
            self._input_details[0]['index'], 
            inputs.astype('float32')
            )
        
        # run
        self._model.invoke()

        # get raw output
        raw_pred = self._model.get_tensor(
            self._output_details[0]['index']
            )
        raw_pred=(raw_pred-28)*0.3449
        
        # apply postprocess
        return self._post_process(raw_pred)

    def predict(self, inputs):
        """Alternative Function name for self.__call__
        """
        return self.__call__(inputs)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight', type=str, default='weights/best_model.tflite', help='model.tflite path. Also keep model.npy file in the same path.')
    parser.add_argument('--img-size', nargs='+', type=int, default=[120, 120], help='[width, height] image sizes')
    
    opt = parser.parse_args()

    model = AgeResNet(opt.weight)
    inputs = np.random.randn(1, *opt.img_size, 3)
    logits, probas = model(inputs)
    print('Logits: ', logits)
    print('Probabilities: ', probas)
