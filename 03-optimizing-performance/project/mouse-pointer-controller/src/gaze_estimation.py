import cv2

from model_abstract import ModelAbstract


class ModelGazeEstimation(ModelAbstract):
    """
    Class for the Gaze Estimation Model.
    """

    def __init__(self, model_name, device='CPU', threshold=0.5, extension=None):
        """
        Use this to set your instance variables.
        """
        super().__init__('Gaze Estimation', model_name, device, threshold, extension)

    def preprocess_input(self, image):
        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        """
        eyes, head_pose = image
        # eyes shape: 1x3x60x60
        # head pose: 1x3
        left_eye = self.transform_eye_image(eyes[0])
        right_eye = self.transform_eye_image(eyes[1])
        return {
            'left_eye_image': left_eye,
            'right_eye_image': right_eye,
            'head_pose_angles': [head_pose]
        }

    @staticmethod
    def transform_eye_image(image):
        img = cv2.resize(image, (60, 60))
        img = img.transpose((2, 0, 1))
        return img.reshape(1, *img.shape)

    def preprocess_output(self, outputs, image):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """
        # shape 1x3
        # Cartesian coordinates of gaze direction vector
        output = outputs[self.output_name]
        return output[0, :]
