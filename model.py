#import tensorflow as tf
import numpy as np
from PIL import Image

class model:

    def __init__(self,labelpath=""):
        self.labels = open("labels.txt","r").read().split("\n")
        self.model_path="custom-model.tflite"

