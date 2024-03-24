import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Activation,Dense,Flatten,Conv2D,MaxPooling2D
from keras.layers.normalization.batch_normalization import BatchNormalization
from tensorflow.python.keras.optimizer_v1 import Adam
from tensorflow.python.keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore',category=FutureWarning)
# physical_devices=tf.config.experimental.list_physical_devices('GPU')
# print("NUm GPUs Available: ",len(physical_devices))
# tf.config.experimental.set_memory_growth(physical_devices[0],True)

cat_source_img=r'E:\_project work\git\Myprojects\Python\OpenCV\Cat-Dog-cnn\kagglecatsanddogs_3367a\PetImages\Cat'
dog_source_img=r'E:\_project work\git\Myprojects\Python\OpenCV\Cat-Dog-cnn\kagglecatsanddogs_3367a\PetImages\Dog'
cat_train_dir=r'E:\_project work\git\Myprojects\Python\OpenCV\Cat-Dog-cnn\train\Cat'
dog_train_dir=r'E:\_project work\git\Myprojects\Python\OpenCV\Cat-Dog-cnn\train\Dog'
cat_valid_dir=r'E:\_project work\git\Myprojects\Python\OpenCV\Cat-Dog-cnn\valid\Cat'
dog_valid_dir=r'E:\_project work\git\Myprojects\Python\OpenCV\Cat-Dog-cnn\valid\Dog'
cat_test_dir=r'E:\_project work\git\Myprojects\Python\OpenCV\Cat-Dog-cnn\test\Cat'
dog_test_dir=r'E:\_project work\git\Myprojects\Python\OpenCV\Cat-Dog-cnn\test\Dog'

if os.path.isdir(dog_train_dir) is False:
    os.makedirs(dog_train_dir)
    os.makedirs(cat_train_dir)
    os.makedirs(dog_valid_dir)
    os.makedirs(cat_valid_dir)
    os.makedirs(dog_test_dir)
    os.makedirs(cat_test_dir)   
    
    for c in random.sample(os.listdir(cat_source_img),500):
        shutil.move(os.path.join(cat_source_img,c),cat_train_dir)
    for c in random.sample(os.listdir(dog_source_img),500):
        shutil.move(os.path.join(dog_source_img,c),dog_train_dir)
    for c in random.sample(os.listdir(cat_source_img),100):
        shutil.move(os.path.join(cat_source_img,c),cat_valid_dir)
    for c in random.sample(os.listdir(dog_source_img),100):
        shutil.move(os.path.join(dog_source_img,c),dog_valid_dir)
    for c in random.sample(os.listdir(cat_source_img),50):
        shutil.move(os.path.join(cat_source_img,c),cat_test_dir)
    for c in random.sample(os.listdir(dog_source_img),50):
        shutil.move(os.path.join(dog_source_img,c),dog_test_dir)
        
train_dir=r'E:\_project work\git\Myprojects\Python\OpenCV\Cat-Dog-cnn\train'
valid_dir=r'E:\_project work\git\Myprojects\Python\OpenCV\Cat-Dog-cnn\valid'
test_dir=r'E:\_project work\git\Myprojects\Python\OpenCV\Cat-Dog-cnn\test'

train_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory=train_dir,target_size=(224,224),classes=['Cat','Dog'],batch_size=10)
valid_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory=valid_dir,target_size=(224,224),classes=['Cat','Dog'],batch_size=10)
test_batches=ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)\
    .flow_from_directory(directory=test_dir,target_size=(224,224),classes=['Cat','Dog'],batch_size=10,shuffle=False)

assert train_batches.n==1000
assert valid_batches.n==200
assert test_batches.n==100
assert train_batches.num_classes==valid_batches.num_classes==test_batches.num_classes==2

imgs,labels=next(train_batches)

def plotImages(images_arr):
    fig,axes=plt.subplots(1,10,figsize=(20,20))
    axes=axes.flatten()
    for img,ax in zip(images_arr,axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

#plotImages(imgs)
#print(labels)
model=Sequential([Conv2D(filters=32,kernel_size=(3,3),activation='relu',padding='same',input_shape=(224,224,3)),
                  MaxPooling2D(pool_size=(2,2),strides=2),
                  Conv2D(filters=64,kernel_size=(3,3),activation='relu',padding='same'),
                  MaxPooling2D(pool_size=(2,2),strides=2),
                  Flatten(),
                  Dense(units=2,activation='softmax')])

#model.summary()
model.compile(optimizer="Adam",loss="categorical_crossentropy",metrics=['accuracy'])
model.fit(x=train_batches,validation_data=valid_batches,epochs=10,verbose=1,steps_per_epoch=100,validation_steps=5)