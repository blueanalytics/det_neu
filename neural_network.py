
# neural_network.py
import tensorflow as tf
from keras import backend as K
import cv2
import numpy as np
from PIL import Image

class NeuralNetwork:
    def __init__(self):
        tf.compat.v1.disable_eager_execution()
        tf.compat.v1.experimental.output_all_intermediates(True)

    def grad_cam(self, array):
        img = self.preprocess(array)
        model = tf.keras.models.load_model('WilhemNet_86.h5')
        preds = model.predict(img)
        argmax = np.argmax(preds[0])
        output = model.output[:, argmax]
        last_conv_layer = model.get_layer("conv10_thisone")
        grads = K.gradients(output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate(img)
        for filters in range(64):
            conv_layer_output_value[:, :, filters] *= pooled_grads_value[filters]
        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[2]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        img2 = cv2.resize(array, (512, 512))
        hif = 0.8
        transparency = heatmap * hif
        transparency = transparency.astype(np.uint8)
        superimposed_img = cv2.add(transparency, img2)
        superimposed_img = superimposed_img.astype(np.uint8)
        return superimposed_img[:, :, ::-1]

    def predict(self, array):
        batch_array_img = self.preprocess(array)
        model = tf.keras.models.load_model('WilhemNet_86.h5')
        prediction = np.argmax(model.predict(batch_array_img))
        proba = np.max(model.predict(batch_array_img)) * 100
        label = ""
        if prediction == 0:
            label = "bacteriana"
        if prediction == 1:
            label = "normal"
        if prediction == 2:
            label = "viral"
        heatmap = self.grad_cam(array)
        return label, proba, heatmap

    def preprocess(self, array):
        array = cv2.resize(array, (512, 512))
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        array = clahe.apply(array)
        array = array / 255
        array = np.expand_dims(array, axis=-1)
        array = np.expand_dims(array, axis=0)
        return array
