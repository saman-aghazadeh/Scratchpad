import tensorflow as tf
from tensorflow.keras import backend
from keras_applications import imagenet_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.layers import VersionAwareLayers
from tensorflow.python.keras.utils import data_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.util.tf_export import keras_export

# import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import ModelCheckpoint
import larq

TF_WEIGHTS_PATH = (
    'https://storage.googleapis.com/tensorflow/keras-applications/'
    'xception/xception_weights_tf_dim_ordering_tf_kernels.h5')
TF_WEIGHTS_PATH_NO_TOP = (
    'https://storage.googleapis.com/tensorflow/keras-applications/'
    'xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5')

layers = VersionAwareLayers()

def Xception(
    include_top=True,
    weights='imagenet',
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
    classifier_activation='softmax'):
  
  

  # Determine proper input shape
  input_shape = imagenet_utils._obtain_input_shape(
      input_shape,
      default_size=299,
      min_size=71,
      data_format=backend.image_data_format(),
      require_flatten=include_top,
      weights=weights)

  if input_tensor is None:
    img_input = layers.Input(shape=input_shape)
  else:
    if not backend.is_keras_tensor(input_tensor):
      img_input = layers.Input(tensor=input_tensor, shape=input_shape)
    else:
      img_input = input_tensor

  channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
  kwargs = dict(kernel_quantizer="ste_sign",kernel_constraint='weight_clip')
  kwargs_d= dict(pointwise_quantizer="ste_sign",pointwise_constraint='weight_clip')

  x = larq.layers.QuantConv2D(
      32, (3, 3),
      strides=(2, 2),
      use_bias=False,
      name='block1_conv1')(img_input)
  x = layers.Activation('relu', name='block1_conv1_act')(x)
  x = layers.BatchNormalization(axis=channel_axis, name='block1_conv1_bn')(x)
  
  x = larq.layers.QuantConv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
  x = layers.Activation('relu', name='block1_conv2_act')(x)
  x = layers.BatchNormalization(axis=channel_axis, name='block1_conv2_bn')(x)
  

  residual = larq.layers.QuantConv2D(
      128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
  residual = layers.BatchNormalization(axis=channel_axis)(residual)

  x = larq.layers.QuantSeparableConv2D(
      128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
  x = layers.Activation('relu', name='block2_sepconv2_act')(x)
  x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv1_bn')(x)
  
  x = larq.layers.QuantSeparableConv2D(
      128, (3, 3), padding='same',use_bias=False, name='block2_sepconv2')(x)
  x = layers.BatchNormalization(axis=channel_axis, name='block2_sepconv2_bn')(x)

  x = layers.MaxPooling2D((3, 3),
                          strides=(2, 2),
                          padding='same',
                          name='block2_pool')(x)
  x = layers.add([x, residual])

  residual = larq.layers.QuantConv2D(
      256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
  residual = layers.BatchNormalization(axis=channel_axis)(residual)

#   x = layers.Activation('relu', name='block3_sepconv1_act')(x)
  x = larq.layers.QuantSeparableConv2D(
      256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
  x = layers.Activation('relu', name='block3_sepconv2_act')(x)
  x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv1_bn')(x)
  
  x = larq.layers.QuantSeparableConv2D(
      256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
  x = layers.BatchNormalization(axis=channel_axis, name='block3_sepconv2_bn')(x)

  x = layers.MaxPooling2D((3, 3),
                          strides=(2, 2),
                          padding='same',
                          name='block3_pool')(x)
  x = layers.add([x, residual])

  residual = larq.layers.QuantConv2D(
      728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
  residual = layers.BatchNormalization(axis=channel_axis)(residual)

#   x = layers.Activation('relu', name='block4_sepconv1_act')(x)
  x = larq.layers.QuantSeparableConv2D(
      728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
  x = layers.Activation('relu', name='block4_sepconv2_act')(x)
  x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv1_bn')(x)
  
  x = larq.layers.QuantSeparableConv2D(
      728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
  x = layers.BatchNormalization(axis=channel_axis, name='block4_sepconv2_bn')(x)

  x = layers.MaxPooling2D((3, 3),
                          strides=(2, 2),
                          padding='same',
                          name='block4_pool')(x)
  x = layers.add([x, residual])

  for i in range(8):
    residual = x
    prefix = 'block' + str(i + 5)

    # x = layers.Activation('relu', name=prefix + '_sepconv1_act')(x)
    x = larq.layers.QuantSeparableConv2D(
        728, (3, 3),
        padding='same',
        use_bias=False,
        name=prefix + '_sepconv1')(x)
    x = layers.Activation('relu', name=prefix + '_sepconv2_act')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name=prefix + '_sepconv1_bn')(x)
    
    x = larq.layers.QuantSeparableConv2D(
        728, (3, 3),
        padding='same',
        use_bias=False,
        name=prefix + '_sepconv2')(x)
    x = layers.Activation('relu', name=prefix + '_sepconv3_act')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name=prefix + '_sepconv2_bn')(x)
    
    x = larq.layers.QuantSeparableConv2D(
        728, (3, 3),
        padding='same',
        use_bias=False,
        name=prefix + '_sepconv3')(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name=prefix + '_sepconv3_bn')(x)

    x = layers.add([x, residual])

  residual = larq.layers.QuantConv2D(
      1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
  residual = layers.BatchNormalization(axis=channel_axis)(residual)

#   x = layers.Activation('relu', name='block13_sepconv1_act')(x)
  x = larq.layers.QuantSeparableConv2D(
      728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
  x = layers.Activation('relu', name='block13_sepconv2_act')(x)
  x = layers.BatchNormalization(
      axis=channel_axis, name='block13_sepconv1_bn')(x)
  
  x = larq.layers.QuantSeparableConv2D(
      1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
  x = layers.BatchNormalization(
      axis=channel_axis, name='block13_sepconv2_bn')(x)

  x = layers.MaxPooling2D((3, 3),
                          strides=(2, 2),
                          padding='same',
                          name='block13_pool')(x)
  x = layers.add([x, residual])

  x = larq.layers.QuantSeparableConv2D(
      1536, (3, 3), padding='same',use_bias=False, name='block14_sepconv1')(x)
  x = layers.Activation('relu', name='block14_sepconv1_act')(x)
  x = layers.BatchNormalization(
      axis=channel_axis, name='block14_sepconv1_bn')(x)
  

  x = larq.layers.QuantSeparableConv2D(
      2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
  x = layers.Activation('relu', name='block14_sepconv2_act')(x)
  x = layers.BatchNormalization(
      axis=channel_axis, name='block14_sepconv2_bn')(x)
  

  if include_top:
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    imagenet_utils.validate_activation(classifier_activation, weights)
    x = layers.Dense(classes, activation=classifier_activation,
                     name='predictions')(x)
  else:
    if pooling == 'avg':
      x = layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
      x = layers.GlobalMaxPooling2D()(x)

  # Ensure that the model takes into account
  # any potential predecessors of `input_tensor`.
  if input_tensor is not None:
    inputs = layer_utils.get_source_inputs(input_tensor)
  else:
    inputs = img_input
  # Create model.
  model = training.Model(inputs, x, name='xception')

  # Load weights.
  if weights == 'imagenet':
    if include_top:
      weights_path = data_utils.get_file(
          'xception_weights_tf_dim_ordering_tf_kernels.h5',
          TF_WEIGHTS_PATH,
          cache_subdir='models',
          file_hash='0a58e3b7378bc2990ea3b43d5981f1f6')
    else:
      weights_path = data_utils.get_file(
          'xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
          TF_WEIGHTS_PATH_NO_TOP,
          cache_subdir='models',
          file_hash='b0042744bf5b25fce3cb969f33bebb97')
    model.load_weights(weights_path)
  elif weights is not None:
    model.load_weights(weights)

  return model


PATH= "/home/local/ASUAD/sbiookag/gender_data/Dataset_gender_12_02_2021"

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'val')
test_dir = os.path.join(PATH, 'test')

BATCH_SIZE = 32
IMG_SIZE = (120, 120)

train_dataset = image_dataset_from_directory(train_dir,
                                             shuffle=True,
                                             batch_size=BATCH_SIZE,
                                             image_size=IMG_SIZE)

validation_dataset = image_dataset_from_directory(validation_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)
test_dataset = image_dataset_from_directory(test_dir,
                                                  shuffle=True,
                                                  batch_size=BATCH_SIZE,
                                                  image_size=IMG_SIZE)

class_names = train_dataset.class_names
print ("class_names",class_names)


data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

preprocess_input = tf.keras.applications.xception.preprocess_input

# rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)


# Create the base model from the pre-trained model MobileNet V2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = Xception(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)

print(feature_batch.shape)

base_model.trainable = False
print (base_model.summary())

print('****************************************')
print(larq.models.summary(base_model))
from contextlib import redirect_stdout

with open('modelsummary.txt', 'w') as f:
    with redirect_stdout(f):
        larq.models.summary(base_model)



global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)
kwargs = dict(input_quantizer="ste_sign",
              kernel_quantizer="ste_sign",
              kernel_constraint="weight_clip")
prediction_layer = larq.layers.QuantDense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

inputs = tf.keras.Input(shape=(120, 120, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])
#loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
print (model.summary())
print (len(model.trainable_variables))
print(np.sum([np.prod(v.get_shape()) for v in model.trainable_weights]))
print('***********************************************************')

initial_epochs = 20

# loss0, accuracy0 = model.evaluate(validation_dataset)
# print("initial loss: {:.2f}".format(loss0))
# print("initial accuracy: {:.2f}".format(accuracy0))
early_stopping_monitor = EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=0,
    verbose=1,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)
checkpoint = ModelCheckpoint("weights/best_model_gender_larq_tf23_2610.h5", monitor='val_loss', verbose=1,
    save_best_only=True, mode='auto', period=1)

history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset,callbacks=[checkpoint])

# loss, accuracy = model.evaluate(test_dataset)
# print('Test accuracy :', accuracy)      

model.save("Xception_gender_120x120_04_05_2021.h5")



base_model.trainable = True

# print("Number of layers in the  model: ", len(model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 2

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False

base_learning_rate = 0.0001
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.Adam(lr=base_learning_rate/10),
              metrics=['accuracy'])

model.summary()       
print (len(model.trainable_variables))
checkpoint = ModelCheckpoint("weights/best_model_gender_larq_tf23_2610.h5", monitor='val_loss', verbose=1,
    save_best_only=True, mode='auto', period=1)

# initial_epochs= 50
fine_tune_epochs = 7
# total_epochs =  initial_epochs + fine_tune_epochs


history_fine = model.fit(train_dataset,
                         epochs=fine_tune_epochs, 
                         validation_data=validation_dataset,callbacks=[checkpoint])
#initial_epoch=history.epoch[-1], 

# loss, accuracy = model.evaluate(test_dataset)
# print('Test accuracy :', accuracy)

model.save("model_gender_larq_tf23_2610.h5",include_optimizer=False)
model.save("saved_model_gender_larq_tf23_2610")

with larq.context.quantized_scope(True):
    model.save("binary_model_model_gender_tf23_2610.h5")
    model.save('saved_binary_model_model_gender_tf23_2610')
    model.set_weights(model.get_weights())
    model.save('final_gender_tf23_2610')

