from openvino.inference_engine import IENetwork, IECore

import numpy as np
import time

# Loading model
model_path='sep_cnn/sep_cnn'
model_weights=model_path+'.bin'
model_structure=model_path+'.xml'

# TODO: Load the model
model = IENetwork(model_structure, model_weights)
input_name=next(iter(model.inputs))

core = IECore()
exec_net = core.load_network(network=model, device_name='CPU', num_requests=1)


# Reading and Preprocessing Image
input_img=np.load('image.npy')
input_img=input_img.reshape(1, 28, 28)


# TODO: Using the input image, run inference on the model for 10 iterations
s = time.time()
for i in range(10):
    exec_net.infer(inputs={input_name: input_img})

# TODO: Finish the print statement
print("Time taken to run 10 iterations is: ", (time.time() - s))