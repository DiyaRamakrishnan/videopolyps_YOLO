import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model

img_length = 50
img_width = 50

class GradCAMModel(tf.keras.Model):
    def __init__(self, base_model, layer_name):
        super(GradCAMModel, self).__init__()
        self.base_model = base_model
        self.layer_name = layer_name
        self.grad_model = tf.keras.Model([base_model.inputs], [base_model.get_layer(layer_name).output, base_model.output])

    def call(self, inputs):
        features, output = self.grad_model(inputs)
        return features, output

def get_grad_cam(model, img_array, class_index, img_length, img_width):
    img_array = np.expand_dims(img_array, axis=0)
    
    with tf.GradientTape() as tape:
        features, output = model(img_array)
        grads = tape.gradient(output, features)
    
    if grads is None:
        print("Gradients are None. Check the model architecture.")
        return None
    
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, features), axis=-1)
    
    heatmap = np.maximum(heatmap, 0)

    max_heatmap = np.max(heatmap)
    
    if max_heatmap == 0:
        print("Warning: Maximum value in heatmap is zero. Setting a small value.")
        heatmap += 1e-10 
        max_heatmap = np.max(heatmap) 
    
    if max_heatmap == 0:
        print("Warning: Maximum value in heatmap is still zero. Check the model or input.")
        return None
    
    heatmap /= max_heatmap
    heatmap = heatmap[0]

    print("Heatmap:", heatmap)

    img = img_array[0]
    img = cv2.resize(img, (img_length, img_width))
    heatmap = cv2.resize(heatmap, (img_length, img_width))
    heatmap = np.uint8(255 * heatmap)

    img = img.astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET).astype(np.uint8)

    print("Max Heatmap Value:", np.max(heatmap))

    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_colored, 0.4, 0)

    return superimposed_img
