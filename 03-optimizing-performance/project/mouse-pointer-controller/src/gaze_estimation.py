from openvino.inference_engine import IENetwork, IECore
import cv2
from util import check_layers_supported

"""
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
"""


class ModelGazeEstimation:
    """
    Class for the Face Detection Model.
    """

    def __init__(self, model_name, device='CPU', threshold=0.5, extension=None):
        """
        Use this to set your instance variables.
        """
        self.model_weights = model_name + '.bin'
        self.model_structure = model_name + '.xml'
        self.device = device
        self.threshold = threshold
        self.extension = extension

        self.core = IECore()
        self.model = self.core.read_network(self.model_structure, self.model_weights)

        self.input_name = next(iter(self.model.inputs))
        self.input_shape = self.model.inputs[self.input_name].shape
        self.output_name = next(iter(self.model.outputs))
        self.output_shape = self.model.outputs[self.output_name].shape
        self.network = None

    def load_model(self):
        """
        This method is for loading the model to the device specified by the user.
        If your model requires any Plugins, this is where you can load them.
        """
        print(f'[GazeEstimation] Loading...')

        if not self.check_model():
            self.core.add_extension(self.extension, self.device)
            self.check_model()

        self.network = self.core.load_network(
            network=self.model,
            device_name=self.device,
            num_requests=1)

    def predict(self, eyes, head_pose):
        """
        This method is meant for running predictions on the input image.
        """
        net_input = self.preprocess_input(eyes, head_pose)
        infer_request_handle = self.network.start_async(request_id=0, inputs=net_input)
        infer_request_handle.wait()
        net_output = infer_request_handle.outputs[self.output_name]
        boxes = self.preprocess_output(net_output)
        return boxes

    def check_model(self):
        return check_layers_supported(self.core, self.model, self.device)

    def preprocess_input(self, eyes, head_pose):
        """
        Before feeding the data into the model for inference,
        you might have to preprocess it. This function is where you can do that.
        """
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

    def preprocess_output(self, outputs):
        """
        Before feeding the output of this model to the next model,
        you might have to preprocess the output. This function is where you can do that.
        """
        # shape 1x3
        # Cartesian coordinates of gaze direction vector
        return outputs[0, :]
