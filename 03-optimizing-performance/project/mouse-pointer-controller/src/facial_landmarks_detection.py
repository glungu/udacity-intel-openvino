from openvino.inference_engine import IENetwork, IECore
import cv2
from util import check_layers_supported

"""
This is a sample class for a model. You may choose to use it as-is or make any changes to it.
This has been provided just to give you an idea of how to structure your model class.
"""


class ModelFacialLandmarksDetection:
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
        print(f'[FacialLandmarks] Loading...')

        if not self.check_model():
            self.core.add_extension(self.extension, self.device)
            self.check_model()

        self.network = self.core.load_network(
            network=self.model,
            device_name=self.device,
            num_requests=1)

    def predict(self, image):
        """
        This method is meant for running predictions on the input image.
        """
        net_input = self.preprocess_input(image)
        infer_request_handle = self.network.start_async(request_id=0, inputs=net_input)
        infer_request_handle.wait()
        net_output = infer_request_handle.outputs[self.output_name]
        return self.preprocess_output(net_output, image)

    def check_model(self):
        return check_layers_supported(self.core, self.model, self.device)

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
        w = image.shape[1]
        h = image.shape[0]

        c1 = outputs[0, 0:2]
        c2 = outputs[0, 2:4]
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
