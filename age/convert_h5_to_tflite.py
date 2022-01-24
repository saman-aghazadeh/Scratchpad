import tensorflow as tf
model = tf.keras.models.load_model('model_current_05_03_2021.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflmodel = converter.convert()
file = open( 'model_current_05_03_2021.tflite' , 'wb' ) 
file.write( tflmodel )
