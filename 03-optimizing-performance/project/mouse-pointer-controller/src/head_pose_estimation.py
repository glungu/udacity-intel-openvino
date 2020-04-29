import cv2
from model_abstract import ModelAbstract


class ModelHeadPoseEstimation(ModelAbstract):
    """
    Class for the Head Pose Estimation Model.
    """

    def __init__(self, model_name, device='CPU', threshold=0.5, extension=None):
        """
        Use this to set your instance variables.
        """
        super().__init__('Head Pose Estimation', model_name, device, threshold, extension)

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
        # print(f'[HeadPose] output: {type(outputs)}, {outputs}')
        yaw = outputs["angle_y_fc"][0, 0]
        pitch = outputs["angle_p_fc"][0, 0]
        roll = outputs["angle_r_fc"][0, 0]

        return [yaw, pitch, roll]
