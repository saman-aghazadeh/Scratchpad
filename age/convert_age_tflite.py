import tensorflow as tf
import glob
import tensorflow.lite as tflite
import numpy as np
from tensorflow import keras
from time import time
import cv2
import tensorflow.lite as lite
import onnx
from onnx2keras import onnx_to_keras

def representative_dataset():
    a=[]
    for i in set(glob.glob('/home/local/ASUAD/sbiookag/guise/age/age dataset 02-03-2021/Age_Dataset_02_03_2021_comb/train__*.jpg')):
        img = cv2.imread(i)
        img = cv2.resize(img, (120, 120))
        img = img / 255.0
        img = img.astype(np.float32)
        a.append(img)
    a = np.array(a)
    print(a.shape)
    img = tf.data.Dataset.from_tensor_slices(a).batch(1)
    for i in img.take(2000):
        yield [i]

model = keras.models.load_model('/home/local/ASUAD/sbiookag/guise/age/age/model_current_05_03_2021.h5')
model.compile()
converter = lite.TFLiteConverter.from_keras_model(model)

converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type =tf.int8
converter.inference_output_type = tf.int8
tflite_quant_model = converter.convert()

interpreter=tf.lite.Interpreter(model_content=tflite_quant_model)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print(input_details)
print(output_details)


file = open('age_gender_int8.tflite', 'wb' ) 
file.write(tflite_quant_model)
