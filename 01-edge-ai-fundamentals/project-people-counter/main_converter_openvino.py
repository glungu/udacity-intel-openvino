"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import os
import sys
import time
import socket
import json
import cv2
import numpy as np
import datetime

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.5,
                        help="Probability threshold for detections filtering"
                             "(0.5 by default)")
    return parser


def connect_mqtt():
    ### TODO: Connect to the MQTT client ###
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)
    return client


def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    infer_network = Network()
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    if 'ssd' in args.model:
        model_type = 'SSD'
    elif 'faster_rcnn' in args.model:
        model_type = 'Faster-RCNN'
    print('### Model type:', model_type)

    ### TODO: Load the model through `infer_network` ###
    infer_network.load_model(args.model)
    net_input_shape = infer_network.get_input_shape()
    print('### Network input shape:', net_input_shape)

    ### TODO: Handle the input stream ###
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)

    input_width = int(cap.get(3))
    input_height = int(cap.get(4))

    in_shape = net_input_shape['image_tensor']

    print('### Shapes:',
          (input_width, input_height),
          '->',
          (in_shape[3], in_shape[2]))

    print('### Threshold:', args.prob_threshold)

    # Create a video writer for the output video
    out = cv2.VideoWriter('out.mp4', 0x00000021, 30,
                          (input_width, input_height))

    t_start = datetime.datetime.now()
    t_infer = []
    print('### Start:', t_start)

    counter = 0
    duration = 0

    counter_prev = 0
    duration_prev = 0

    counter_total = 0
    counter_report = 0

    ### TODO: Loop until stream is over ###
    while cap.isOpened():
        ### TODO: Read from the video capture ###
        flag, frame = cap.read()
        if not flag:
            break

        ### TODO: Pre-process the image as needed ###
        net_image = cv2.resize(frame, (in_shape[3], in_shape[2]))
        # print('### Resize:', net_input.shape)
        net_image = net_image.transpose((2, 0, 1))
        # print('### Transpose:', net_input.shape)
        net_image = net_image.reshape(1, *net_image.shape)
        # print('### Reshape:', net_image.shape)

        ### TODO: Start asynchronous inference for specified request ###
        t_start_infer = datetime.datetime.now()

        if model_type == 'SSD':
            net_input = {
                'image_tensor': net_image
            }
        elif model_type == 'Faster-RCNN':
            net_input = {
                'image_tensor': net_image,
                'image_info': net_image.shape[1:]
            }

        duration_report = None

        infer_network.exec_net(net_input)

        ### TODO: Wait for the result ###
        if infer_network.wait() == 0:
            t_end_infer = datetime.datetime.now()
            t_infer.append((t_end_infer - t_start_infer) / datetime.timedelta(milliseconds=1))

            ### TODO: Get the results of the inference request ###
            net_output = infer_network.get_output()

            ### TODO: Extract any desired stats from the results ###
            # 1x1x100x7

            cnt = 0
            probs = net_output[0, 0, :, 2]
            for i, p in enumerate(probs):
                if p > args.prob_threshold:
                    cnt += 1
                    box = net_output[0, 0, i, 3:]
                    p1 = (int(box[0] * input_width), int(box[1] * input_height))
                    p2 = (int(box[2] * input_width), int(box[3] * input_height))
                    frame = cv2.rectangle(frame, p1, p2, (0, 0, 255), 3)

            print(cnt, end='', flush=True)

            if cnt != counter:
                counter_prev = counter
                counter = cnt
                if duration >= 3:
                    duration_prev = duration
                    duration = 0
                else:
                    duration = duration_prev + duration
                    duration_prev = 0  # unknown, not needed in this case
            else:
                duration += 1
                if duration >= 3:
                    counter_report = counter
                    if duration == 3 and counter > counter_prev:
                        print()
                        print('Enter:', counter_prev, '->', counter)
                        counter_total += counter - counter_prev
                    elif duration == 3 and counter < counter_prev:
                        print()
                        print('Exit:', counter_prev, '->', counter)
                        duration_report = int((duration_prev / 10.0) * 1000)

            ### TODO: Calculate and send relevant information on ###
            ### current_count, total_count and duration to the MQTT server ###
            ### Topic "person": keys of "count" and "total" ###
            ### Topic "person/duration": key of "duration" ###
            client.publish('person',
                           payload=json.dumps({
                               'count': counter_report, 'total': counter_total}),
                           qos=0, retain=False)
            if duration_report is not None:
                client.publish('person/duration',
                               payload=json.dumps({'duration': duration_report}),
                               qos=0, retain=False)
                print()
                print('Reporting duration:', duration_report, 'ms')

        ### TODO: Send the frame to the FFMPEG server ###
        out.write(frame)

        ### TODO: Write an output image if `single_image_mode` ###

    t_end = datetime.datetime.now()
    print()
    print('### Finished:', t_end)
    print('### Video processed in:', (t_end - t_start))
    print('### Total people counted:', counter_total)

    # save inference latencies
    np.savetxt("latency_openvino.csv",
               np.asarray(t_infer),
               delimiter=",")

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
