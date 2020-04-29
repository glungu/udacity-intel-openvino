import cv2

from model_abstract import ModelAbstract


class ModelFacialLandmarksDetection(ModelAbstract):
    """
    Class for the Face Detection Model.
    """

    def __init__(self, model_name, device='CPU', threshold=0.5, extension=None):
        """
        Use this to set your instance variables.
        """
        super().__init__('Facial Landmarks Detection', model_name, device, threshold, extension)

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
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """
        output = outputs[self.output_name]
        w = image.shape[1]
        h = image.shape[0]

        c1 = output[0, 0:2]
        c2 = output[0, 2:4]
        c1 = (int(c1[0] * w), int(c1[1] * h))
        c2 = (int(c2[0] * w), int(c2[1] * h))

        # eye width approx. 1/4 of face width
        # eye height approx 1/10 of face height
        eyew = int(w / (4*2))
        eyeh = int(h / (10*2))

        e1 = image[self.limit(c1[1] - eyeh, h):self.limit(c1[1] + eyeh, h),
                   self.limit(c1[0] - eyew, w):self.limit(c1[0] + eyew, w)]
        e2 = image[self.limit(c2[1] - eyeh, h):self.limit(c2[1] + eyeh, h),
                   self.limit(c2[0] - eyew, w):self.limit(c2[0] + eyew, w)]

        return c1, c2, e1, e2

    @staticmethod
    def limit(v, maxv):
        if v < 0:
            return 0
        if v > maxv:
            return maxv
        return v
