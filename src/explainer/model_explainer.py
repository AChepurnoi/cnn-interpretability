from torch import Tensor
from typing import Type, Tuple
from src.explainer.misc_functions import apply_colormap_on_image, to_image, convert_to_grayscale
from src.explainer.functions import *
from .filter_visualisation import CNNLayerFilterVisualization
from .gradient_visualisation import generate_smooth_grad, VanillaBackprop
from .grad_cam import GradCam
from .guided_backprop import FilterOnImageVisualisation
import numpy as np


class ClassifierExplainer:
    def __init__(self, net):
        self.net = net

    def heatmap_visualisation(self, image: Tensor,
                              target_layer: int = 16,
                              target_class: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        gradcam = GradCam(self.net, target_layer=target_layer)
        cam = gradcam.generate_cam(image.unsqueeze(0), target_class=target_class)
        heatmap, heatmap_on_image = apply_colormap_on_image(to_image(image), cam, 'hsv')
        return np.asarray(heatmap), np.asarray(heatmap_on_image)

    def smooth_gradient_visualisation(self, image: Tensor,
                                      target_class: int = 1,
                                      iterations: int = 10,
                                      sigma: float = 1.5) -> np.ndarray:
        vb = VanillaBackprop(self.net)
        gradients = generate_smooth_grad(vb, image.unsqueeze(0), target_class, iterations, sigma)
        gradients = convert_to_grayscale(gradients)
        gradients = gradients - gradients.min()
        gradients /= gradients.max()
        return gradients

    def filter_pattern_visualisation(self, target_layer: int = 16, target_filter: int = 1) -> np.ndarray:
        cnn_vis = CNNLayerFilterVisualization(self.net.features, target_layer, target_filter)
        visualisation = cnn_vis.get_filter_visualisation()
        return visualisation

    def filter_activation_visualisation(self, image: Tensor,
                                        target_layer: int = 16,
                                        target_filter: int = 1) -> np.ndarray:
        fv = FilterOnImageVisualisation(self.net)
        grads = fv.generate_gradients(image.unsqueeze(0), target_layer, target_filter)
        gradient = convert_to_grayscale(grads)
        gradient = gradient - gradient.min()
        gradient /= gradient.max()
        return gradient

    def distributions_visualisation(self):
        show_distributions_of_layers(model=self.net)

    def max_activations_patches_visualisation(self,  layer: str, directory_name: str) -> None:
        show_max_activations_patches(self.net, layer, directory_name)

    def print_summary(self):
        print_summary(self.net)
