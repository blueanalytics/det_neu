import unittest
import numpy as np
from neural_network import NeuralNetwork

class TestNeuralNetwork(unittest.TestCase):
    def test_predict(self):
        neural_network = NeuralNetwork()
        input_data = (np.random.rand(256, 256, 3) * 255).astype(np.uint8)

        # Prueba la función predict
        result = neural_network.predict(input_data)

        # Asegura que el resultado es una tupla
        self.assertIsInstance(result, tuple)

        # Desempaqueta los valores de la tupla (label, proba, heatmap)
        label, proba, heatmap = result

        # Asegura que los valores son del tipo y rango esperados
        self.assertIsInstance(label, str)
        self.assertIsInstance(proba, float)
        self.assertIsInstance(heatmap, np.ndarray)

        # Ajusta las dimensiones según lo que devuelve tu función
        self.assertEqual(heatmap.shape, (512, 512, 3))
