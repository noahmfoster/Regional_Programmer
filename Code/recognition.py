#This file will hold the character recognition portion of the code (CNN or Resnet with data)
import tensorflow as tf
from tensorflow import keras
import keras.applications
from keras.applications.resnet import preprocess_input

#from tensorflow.python.keras import layers
#from tensorflow.python.keras.layers import Dense, Flatten
#from tensorflow.python.keras.models import Sequential

def get_resnet():
    #Code source: https://chroniclesofai.com/transfer-learning-with-keras-resnet-50/
    #Changed format to match
    #https://www.tensorflow.org/guide/keras/transfer_learning
    #resnet_model = tf.keras.Sequential()



    #Notes from code source:

    #While importing the ResNet50 class, we mention include_top=False. 
    # This ensures that we can add our own custom input and output layers according to our data.
    # We mention the weights='imagenet'. 
    # This means that the Resnet50 model will use the weights it learnt while being trained on the imagenet data.
    # Finally, we mention layer.trainable= False in the pretrained model. 
    # This ensures that the model does not learn the weights again, saving us a lot of time and space complexity.
    
    data_augmentation = keras.Sequential(
    [tf.keras.layers.RandomFlip("horizontal"), tf.keras.layers.RandomRotation(0.1),]
    )
    
    pretrained_model= tf.keras.applications.ResNet50(include_top=False,
                    input_shape=(224,224,3),
                    pooling='avg',classes=6,
                    weights='imagenet')

    #for layer in pretrained_model.layers:
     #       layer.trainable=False
    pretrained_model.trainable = False

    inputs = keras.Input(shape=(224, 224, 3))
    x = data_augmentation(inputs)  # Apply random data augmentation
    # Pre-trained Xception weights requires that input be scaled
    # from (0, 255) to a range of (-1., +1.), the rescaling layer
    # outputs: `(inputs * scale) + offset`
    scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    x = scale_layer(x)
    x = pretrained_model(x, training = False)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(624, activation = 'relu')(x)
    outputs= tf.keras.layers.Dense(6, activation = 'softmax')(x)

  #  x = keras.layers.GlobalAveragePooling2D()(x)
    #x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
    #outputs = keras.layers.Dense(1)(x)
    #resnet_model.add(pretrained_model)

    #Add additional layers for further training on new data
    
    #resnet_model.add(tf.keras.layers.Flatten())
    #resnet_model.add(tf.keras.layers.Dense(512, activation='relu'))
    #resnet_model.add(tf.keras.layers.Dense(5, activation='softmax'))
    #x = data_augmentation(inputs)  # Apply random data augmentation

    #x = tf.keras.layers.Flatten()(x)
    #x = (tf.keras.layers.Dense(512, activation='relu'))(x)
    #outputs = (tf.keras.layers.Dense(1, activation='softmax'))(x)
    #outputs = (tf.keras.layers.Dense(5, activation='softmax'))(x)

    resnet_model = keras.Model(inputs, outputs)
    return resnet_model