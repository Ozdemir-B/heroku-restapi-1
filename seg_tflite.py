# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 16:23:16 2020

@author: hp
"""

import numpy as np
import tflite_runtime.interpreter as tflite
import cv2
import visualization_utils as vis_util

class seg:

    def __init__(self):
        self.img_path_adasdas = ""


    def create_category_index(self,label_path='coco_ssd_mobilenet/labelmap.txt'):
        """
        To create dictionary of label map

        Parameters
        ----------
        label_path : string, optional
            Path to labelmap.txt. The default is 'coco_ssd_mobilenet/labelmap.txt'.

        Returns
        -------
        category_index : dict
            nested dictionary of labels.

        """
        f = open(label_path)
        category_index = {}
        for i, val in enumerate(f):
            if i != 0:
                val = val[:-1]
                if val != '???':
                    category_index.update({(i-1): {'id': (i-1), 'name': val}})

        f.close()
        return category_index
    def get_output_dict(self,image, interpreter, output_details, nms=True, iou_thresh=0.5, score_thresh=0.6):
        """
        Function to make predictions and generate dictionary of output

        Parameters
        ----------
        image : Array of uint8
            Preprocessed Image to perform prediction on
        interpreter : tensorflow.lite.python.interpreter.Interpreter
            tflite model interpreter
        input_details : list
            input details of interpreter
        output_details : list
        nms : bool, optional
            To perform non-maximum suppression or not. The default is True.
        iou_thresh : int, optional
            Intersection Over Union Threshold. The default is 0.5.
        score_thresh : int, optional
            score above predicted class is accepted. The default is 0.6.

        Returns
        -------
        output_dict : dict
            Dictionary containing bounding boxes, classes and scores.

        """
        output_dict = {
                       'detection_boxes' : interpreter.get_tensor(output_details[0]['index'])[0],
                       'detection_classes' : interpreter.get_tensor(output_details[1]['index'])[0],
                       'detection_scores' : interpreter.get_tensor(output_details[2]['index'])[0],
                       'num_detections' : interpreter.get_tensor(output_details[3]['index'])[0]
                       }

        output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

        return output_dict



    def make_and_show_inference(self,img, interpreter, input_details, output_details, category_index, nms=True, score_thresh=0.6, iou_thresh=0.5):
        """
        Generate and draw inference on image

        Parameters
        ----------
        img : Array of uint8
            Original Image to find predictions on.
        interpreter : tensorflow.lite.python.interpreter.Interpreter
            tflite model interpreter
        input_details : list
            input details of interpreter
        output_details : list
            output details of interpreter
        category_index : dict
            dictionary of labels
        nms : bool, optional
            To perform non-maximum suppression or not. The default is True.
        score_thresh : int, optional
            score above predicted class is accepted. The default is 0.6.
        iou_thresh : int, optional
            Intersection Over Union Threshold. The default is 0.5.

        Returns
        -------
        NONE
        """
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb = cv2.resize(img_rgb, (300, 300), cv2.INTER_AREA)
        img_rgb = img_rgb.reshape([1, 300, 300, 3])

        interpreter.set_tensor(input_details[0]['index'], img_rgb)
        interpreter.invoke()

        output_dict = self.get_output_dict(img_rgb, interpreter, output_details, nms, iou_thresh, score_thresh)
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
        img,
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=score_thresh,
        line_thickness=3)

    # Load TFLite model and allocate tensors.
if __name__=="__main__":
    interpreter = tflite.Interpreter(model_path="coco_ssd_mobilenet/detect.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    seg_tf = seg()
    category_index = seg_tf.create_category_index()
    input_shape = input_details[0]['shape']
    cap = cv2.VideoCapture(0)

    while(True):
        ret, img = cap.read()
        if ret:
            seg_tf.make_and_show_inference(img, interpreter, input_details, output_details, category_index)
            cv2.imshow("image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
        
    cap.release()
    cv2.destroyAllWindows()
