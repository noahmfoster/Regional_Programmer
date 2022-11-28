#In this file, we will preprocess the image data in one function and preprocess the 
#language data in another

import tensorflow as tf
import numpy as np

#Not actually using this, probably delete from https://towardsdatascience.com/how-to-split-a-tensorflow-dataset-into-train-validation-and-test-sets-526c8dd29438
def get_dataset_partitions_tf(ds, ds_size, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1
    
    if shuffle:
        # Specify seed to always have the same split distribution between runs
        ds = ds.shuffle(shuffle_size, seed=12)
    
    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)
    
    train_ds = ds.take(train_size)    
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)
    
    return train_ds, val_ds, test_ds

def get_characters(file_path):
    """
    Loads The Office Character dataset

    :input file path with datapath for the Office Characters dataset

    :return X0: training images,
            Y0: training labels,
            X1: testing images,
            Y1: testing labels
            D0: TF Dataset training subset
            D1: TF Dataset testing subset
    """

    ## This process may take a bit to load the first time; should get much faster
    import tensorflow_datasets as tfds

    ## Overview of dataset downloading: https://www.kaggle.com/datasets/pathikghugare/the-office-characters
    #Used code below from https://www.tensorflow.org/datasets/api_docs/python/tfds/folder_dataset/ImageFolder
    builder = tfds.ImageFolder(file_path)
     # num examples, labels... are automatically calculated

    #ds = builder.as_dataset(as_supervised = True,shuffle_files=True)

    #ds = builder.as_dataset(as_supervised = True)

    train_ds = builder.as_dataset(as_supervised = True, split='train', shuffle_files=True)
    test_ds= builder.as_dataset(as_supervised=True, split='test', shuffle_files=True)
    #val_ds = train_ds.take(val_size)

    #train_ds, val_ds, test_ds = get_dataset_partitions_tf(ds, 2306, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True, shuffle_size=10000)
    tfds.show_examples(train_ds, builder.info)        


   # X0, X1 = [np.array([r[0] for r in tfds.as_numpy(D)]) for D in (D0, D1)]
    #Y0, Y1 = [np.array([r[1] for r in tfds.as_numpy(D)]) for D in (D0, D1)]

    #Resizing inputs
    size = (224, 224)

    train_ds = train_ds.map(lambda x, y: (tf.image.resize(x, size), y))
    #validation_ds = validation_ds.map(lambda x, y: (tf.image.resize(x, size), y))
    test_ds = test_ds.map(lambda x, y: (tf.image.resize(x, size), y))

    batch_size = 32

    train_ds = train_ds.cache().batch(batch_size).prefetch(buffer_size=10)
    #validation_ds = validation_ds.cache().batch(batch_size).prefetch(buffer_size=10)
    test_ds = test_ds.cache().batch(batch_size).prefetch(buffer_size=10)






    return  train_ds, test_ds #X0,Y0,X1,Y1,D0, D1


###############################################################################################
