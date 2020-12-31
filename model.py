import tflite_runtime.interpreter as tflite
import cv2
from PIL import Image
import seg_tflite

class Model:

    def __init__(self):
        self.interpreter = tflite.Interpreter(model_path="coco_ssd_mobilenet/detect.tflite")
        self.interpreter.allocate_tensors()

        # Get input and output tensors.
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.seg_tf = seg_tflite.seg()
        self.category_index = self.seg_tf.create_category_index()
        self.input_shape = self.input_details[0]['shape']

    def detect(self,im1):#takes a cv2 img as argument
        img=im1#cv2.imread(img_path,0)
        self.seg_tf.make_and_show_inference(img, self.interpreter, self.input_details, self.output_details, self.category_index)
        cv2.imshow("image",img)
        cv2.waitKey()

if __name__ == "__main__":
    interpreter = tflite.Interpreter(model_path="coco_ssd_mobilenet/detect.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    seg_tf = seg_tflite.seg()
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