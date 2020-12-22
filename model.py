import tensorflow as tf
import numpy as np
from PIL import Image

class model:
    def __init__(self,path_labels,path_model):
        
        self.labels = open(path_labels,"r").read().split('\n')
        self.output = []
        self.output_label = self.labeled(self.output,self.labels)

    def execute(self,image):
        interpreter = tf.lite.Interpreter(model_path="model2.tflite")
        interpreter.allocate_tensors()

        # Get input and output tensors.
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Test model on random input data.
        input_shape = input_details[0]['shape']
        input_data = np.array(image.resize(input_shape), dtype=np.uint8)
        print(input_shape)
        interpreter.set_tensor(input_details[0]['index'], input_data)

        interpreter.invoke()

        # The function `get_tensor()` returns a copy of the tensor data.
        # Use `tensor()` in order to get a pointer to the tensor.
        output_data = interpreter.get_tensor(output_details[0]['index'])
        print(len(output_data[0]))
        self.output = output_data[0]
        return output_data[0]

    def labeled(self,output,labels):
        for i in output:
            if i == 1:
                return labels[i]
            else:
                return "no match"



"""
interpreter = tf.lite.Interpreter(model_path="custom_model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data)
"""