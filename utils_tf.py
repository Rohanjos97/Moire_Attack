import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize = (5, 15))
    plt.imshow(np.transpose(npimg,(1,2,0)))
    plt.title(title)
    plt.show()
    
def image_folder_custom_label(root, transform, idx2label):
    data_gen = ImageDataGenerator(validation_split=0.0)
    old_data = data_gen.flow_from_directory(root, target_size=(224, 224), class_mode='sparse')
    old_classes = list(old_data.class_indices.keys())
    
    label2idx = {}
    
    for i, item in enumerate(idx2label) :
        label2idx[item] = i
    new_data = data_gen.flow_from_directory(root, target_size=(224, 224), 
                                 classes=old_classes, class_mode='sparse',
                                 interpolation='nearest', 
                                 shuffle=True, batch_size=32,
                                 save_to_dir=None, save_prefix='',
                                 save_format='png', follow_links=False,
                                 subset=None,
                                 seed=None, 
                                 )
    
#     new_data.class_indices = label2idx
#     new_data.classes = list(label2idx.keys())
    return new_data

def create_dir(dir, print_flag = False):
    if not os.path.exists(dir):
        os.makedirs(dir)
        if print_flag:
            print("Create dir {} successfully!".format(dir))
    elif print_flag:
        print("Directory {} is already existed. ".format(dir))

def save_img(input_img, target_save_path):
    if isinstance(input_img, np.ndarray):
        pil_img = Image.fromarray(input_img.astype(np.uint8))
    else:
        pil_img = input_img
    pil_img.save(target_save_path, "JPEG", quality = 100)

def adjust_contrast_and_brightness(input_img, beta = 30):
    input_img = tf.clip_by_value(input_img + beta, 0, 255)
    return input_img