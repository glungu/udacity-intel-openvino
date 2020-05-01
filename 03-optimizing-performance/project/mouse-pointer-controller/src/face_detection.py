import cv2
import numpy as np

from model_abstract import ModelAbstract


class ModelFaceDetection(ModelAbstract):
    """
    Class for the Face Detection Model.
    """

    def __init__(self, model_name, device='CPU', threshold=0.5, extension=None):
        """
        Use this to set your instance variables.
        """
        super().__init__('Face Detection', model_name, device, threshold, extension)

    def preprocess_input(self, image):
        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        """
        input_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        input_image = input_image.transpose((2, 0, 1))
        input_image = input_image.reshape(1, *input_image.shape)
        return {self.input_name: input_image}

    def preprocess_output(self, outputs, image):
        """
        Return tuple: p1, p2, image
        p1      - top left point of the detected face
        p2      - bottom right point of the detected face
        image   - cropped image of the face
        """
        output = outputs[self.output_name]
        w = image.shape[1]
        h = image.shape[0]

        probs = output[0, 0, :, 2]
        i = np.argmax(probs)
        if probs[i] > self.threshold:
            box = output[0, 0, i, 3:]
            p1 = (int(box[0] * w), int(box[1] * h))
            p2 = (int(box[2] * w), int(box[3] * h))
            return p1, p2, image[p1[1]:p2[1], p1[0]:p2[0]]
        else:
            # no face detected with enough confidence
            return None, None, None
