import os
import sys
import time
import socket
import json
import datetime

import cv2
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-f", "--file",
                        required=False,
                        default='',
                        type=str,
                        help="Path to CSV file(s)")
    args = parser.parse_args()
    return args


def visualize_compare_latencies(plot, title, file_csv_tensorflow, file_csv_openvino):
    # first latency value usually huge for TF, possibly warm-up, so exclude for proper scale
    array_tf = np.genfromtxt(file_csv_tensorflow, delimiter=',')[1:]
    array_ov = np.genfromtxt(file_csv_openvino, delimiter=',')[1:]

    avg_ov = np.average(array_ov)
    avg_tf = np.average(array_tf)
    print('### ', title,
          'average OpenVINO', avg_ov,
          'average Plain TF:', avg_tf,
          'Gain:', (avg_tf/avg_ov), 'times')

    label_ov, = plot.plot(array_ov, label='OpenVINO')
    label_tf, = plot.plot(array_tf, label='Plain TF')
    plot.legend(handles=[label_ov, label_tf], loc='upper left')
    plot.set_title(title)


def main():
    # Grab command line args
    args = parse_args()

    figure = plt.figure()

    subplot1, subplot2 = figure.subplots(1, 2, sharey=False)
    plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.3, hspace=None)

    # Visualize according to command line args
    visualize_compare_latencies(subplot1, 'ssd_inception_v2_coco',
                                'data/tf_latency_ssd.csv',
                                'data/openvino_latency_ssd.csv')
    visualize_compare_latencies(subplot2, 'faster_rcnn_inception_v2_coco',
                                'data/tf_latency_faster_rcnn.csv',
                                'data/openvino_latency_faster_rcnn.csv')
    plt.show()


if __name__ == '__main__':
    main()
