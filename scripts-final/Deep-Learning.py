import tensorflow as tf
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dropout, Conv2DTranspose, concatenate

# Set paths
path = ''
image_path = os.path.join(path, '/...../CameraRGB/')
mask_path = os.path.join(path, '/....../CameraMask/')

# List and sort filenames
image_list = sorted(os.listdir(image_path), key=lambda x: int(x.split('.')[0]))
mask_list = sorted(os.listdir(mask_path), key=lambda x: int(x.split('.')[0]))

# Add full paths
image_list = [os.path.join(image_path, i) for i in image_list]
mask_list = [os.path.join(mask_path, i) for i in mask_list]

# Create dataset
image_filenames = tf.constant(image_list)
masks_filenames = tf.constant(mask_list)
dataset = tf.data.Dataset.from_tensor_slices((image_filenames, masks_filenames))

# Processing functions
multiplier = 2
def process_path(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    return img, mask

def preprocess(image, mask):
    input_image = tf.image.resize(image, (96*multiplier, 128*multiplier), method='nearest')
    input_mask = tf.image.resize(mask, (96*multiplier, 128*multiplier), method='nearest')
    input_image = input_image / 255.
    return input_image, input_mask

image_ds = dataset.map(process_path)
processed_image_ds = image_ds.map(preprocess)

# Model functions
def conv_block(inputs, n_filters=32, dropout_prob=0, max_pooling=True):
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    if dropout_prob > 0:
        conv = Dropout(dropout_prob)(conv)
    next_layer = MaxPooling2D(2, strides=2)(conv) if max_pooling else conv
    return next_layer, conv

def upsampling_block(expansive_input, contractive_input, n_filters=32):
    up = Conv2DTranspose(n_filters, 3, strides=2, padding='same')(expansive_input)
    merge = concatenate([up, contractive_input], axis=3)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge)
    conv = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv)
    return conv

def unet_model(input_size=(96*multiplier, 128*multiplier, 3), n_filters=32, n_classes=23):
    inputs = Input(input_size)
    cblock1 = conv_block(inputs, n_filters*1)
    cblock2 = conv_block(cblock1[0], n_filters*2)
    cblock3 = conv_block(cblock2[0], n_filters*4)
    cblock4 = conv_block(cblock3[0], n_filters*8, dropout_prob=0.3)
    cblock5 = conv_block(cblock4[0], n_filters*16, dropout_prob=0.3)
    cblock6 = conv_block(cblock5[0], n_filters*32, dropout_prob=0.3, max_pooling=False)
    
    ublock7 = upsampling_block(cblock6[0], cblock5[1], n_filters*16)
    ublock8 = upsampling_block(ublock7, cblock4[1], n_filters*8)
    ublock9 = upsampling_block(ublock8, cblock3[1], n_filters*4)
    ublock10 = upsampling_block(ublock9, cblock2[1], n_filters*2)
    ublock11 = upsampling_block(ublock10, cblock1[1], n_filters*1)
    
    conv12 = Conv2D(n_filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(ublock11)
    conv13 = Conv2D(n_classes, 1, padding='same')(conv12)
    
    model = tf.keras.Model(inputs=inputs, outputs=conv13)
    return model

# Initialize model
img_height = 96 * multiplier
img_width = 128 * multiplier
num_channels = 3
unet = unet_model((img_height, img_width, num_channels))
unet.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

# Training parameters
EPOCHS = 50
BUFFER_SIZE = 500
BATCH_SIZE = 32
train_dataset = processed_image_ds.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Train model
with tf.device('/GPU:0'):
    model_history = unet.fit(train_dataset, epochs=EPOCHS)

# Plot accuracy
accuracy = model_history.history["accuracy"]
plt.figure(figsize=(10, 5))
plt.plot(accuracy, marker='o', linestyle='-', linewidth=2, markersize=6)
plt.title("Training Accuracy Over Epochs", fontsize=14, fontweight='bold')
plt.xlabel("Epochs", fontsize=12)
plt.ylabel("Accuracy", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(range(0, len(accuracy), max(1, len(accuracy)//10)))
plt.show()
