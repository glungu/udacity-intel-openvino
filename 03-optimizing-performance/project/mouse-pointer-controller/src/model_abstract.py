"""
Class containing common model methods, to be inherited by other model classes.
"""
import time
import numpy as np
from util import check_layers_supported
from openvino.inference_engine import IENetwork, IECore


class ModelAbstract:
    """
    Init metrics
    """
    def __init__(self, name, model_name, device='CPU', threshold=0.5, extension=None):
        """
        Use this to set your instance variables.
        """
        self.name = name
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

        self.loading_time = -1.0
        self.inference_times = []
        self.processing_times = []

    def load_model(self):
        if not self.check_model():
            print(f'[{self.name}] Not all layers supported, adding extension...')
            self.core.add_extension(self.extension, self.device)
            self.check_model()
        else:
            print(f'[{self.name}] All layers supported')

        try:
            s = time.time()
            self.network = self.core.load_network(
                network=self.model,
                device_name=self.device,
                num_requests=1)
            self.loading_time = time.time() - s
        except Exception as e:
            print(f'[{self.name}] ERROR: Cannot load network: {e}')
            raise e

    def predict(self, image):
        s1 = time.time()
        net_input = self.preprocess_input(image)

        s2 = time.time()
        infer_request_handle = self.network.start_async(request_id=0, inputs=net_input)
        infer_request_handle.wait()
        self.inference_times.append(time.time() - s2)

        net_output = infer_request_handle.outputs
        output = self.preprocess_output(net_output, image)
        self.processing_times.append(time.time() - s1)
        return output

    def check_model(self):
        return check_layers_supported(self.core, self.model, self.device)

    def preprocess_input(self, image):
        raise NotImplementedError

    def preprocess_output(self, outputs, image):
        raise NotImplementedError

    def print_stats(self, header=True):
        avg_pt = np.average(np.array(self.processing_times, dtype=np.float))
        avg_it = np.average(np.array(self.inference_times, dtype=np.float))
        if header:
            print()
            print(f'| Name | Loading Time | Avg processing time | Avg inference time |')
            print(f'|------|--------------|---------------------|--------------------|')
        print(f'| {self.name} | {self.loading_time:.3} | {avg_pt:.3} | {avg_it:.3} |')
