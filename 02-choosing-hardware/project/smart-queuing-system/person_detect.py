from datetime import datetime

import numpy as np
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IECore
import os
import cv2
import argparse
import traceback


class Queue:
    """
    Class for dealing with queues
    """
    def __init__(self):
        self.queues = []

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max = q
            frame = image[y_min:y_max, x_min:x_max]
            yield frame
    
    def check_coords(self, coords):
        d = {k + 1: 0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0] > q[0] and coord[2] < q[2]:
                    d[i + 1] += 1
        return d


class PersonDetect:
    """
    Class for the Person Detection Model.
    """

    def __init__(self, model_xml):
        # This method needs to be completed by you
        self.model_structure = model_xml
        self.model_weights = os.path.splitext(model_xml)[0] + ".bin"
        self.network = None
        self.input_shape = None
        self.input_key = None
        self.output_key = None

    def load_model(self, device):
        # This method needs to be completed by you
        model = IENetwork(self.model_structure, self.model_weights)
        print('Model loaded')
        core = IECore()
        print('Core created')
        self.network = core.load_network(network=model, device_name=device, num_requests=1)
        print('Network loaded')
        self.input_key = next(iter(self.network.inputs))
        self.output_key = next(iter(self.network.outputs))
        self.input_shape = self.network.inputs[self.input_key].shape
        print('Input key:', self.input_key, 'input shape:', self.input_shape)
        print('Output key:', self.output_key)

    def check_plugin(self, plugin):
        # This method needs to be completed by you
        raise NotImplementedError

    def predict(self, image, threshold, input_shape, queue):
        # This method needs to be completed by you
        net_input = self.preprocess_input(image)
        infer_request_handle = self.network.start_async(request_id=0, inputs=net_input)
        if infer_request_handle.wait() == 0:
            net_output = infer_request_handle.outputs[self.output_key]
            return self.preprocess_outputs(image, net_output, threshold, input_shape, queue)

    def preprocess_outputs(self, image, outputs, threshold, input_shape, queue):
        # This method needs to be completed by you
        for q in queue.queues:
            image = cv2.rectangle(image, (q[0], q[1]), (q[2], q[3]), (0, 255, 0), 5)

        # outputs.shape: (1, 1, 200, 7)
        boxes = []
        probs = outputs[0, 0, :, 2]
        for i, p in enumerate(probs):
            if p > threshold:
                box = outputs[0, 0, i, 3:]
                p1 = (int(box[0] * input_shape[0]), int(box[1] * input_shape[1]))
                p2 = (int(box[2] * input_shape[0]), int(box[3] * input_shape[1]))
                image = cv2.rectangle(image, p1, p2, (0, 0, 255), 3)
                boxes.append([p1[0], p1[1], p2[0], p2[1]])
        return boxes, image

    def preprocess_input(self, image):
        # This method needs to be completed by you
        input_image = cv2.resize(image, (self.input_shape[3], self.input_shape[2]))
        input_image = input_image.transpose((2, 0, 1))
        input_image = input_image.reshape(1, *input_image.shape)
        return {self.input_key: input_image}


def main(args):
    extensions = args.extensions
    model = args.model
    device = args.device
    visualise = args.visualise
    th = float(args.threshold)
    output_path = args.output_path

    start = datetime.now()
    pd = PersonDetect(model_xml=model)
    pd.load_model(device=device)
    print(f"Model loaded, loading time: {datetime.now()-start}")
    
    # Queue Parameters

    # For retail
    # queue.add_queue([620, 1, 915, 562])
    # queue.add_queue([1000, 1, 1264, 461])

    # For manufacturing
    # queue.add_queue([15, 180, 730, 780])
    # queue.add_queue([921, 144, 1424, 704])

    # For Transport
    # queue.add_queue([50, 90, 838, 794])
    # queue.add_queue([852, 74, 1430, 841])

    try:
        queue = Queue()
        queue.add_queue([15, 180, 900, 780])
        queue.add_queue([921, 144, 1600, 704])
        video_file = args.video
        cap = cv2.VideoCapture(video_file)
        cap.open(video_file)

        out_stats = open(output_path + '/stats.txt', 'w')
        out_stats.write('{}, {}, {}\n'.format('Frame', 'Total People', 'People in Queues'))

        if visualise:
            input_width = int(cap.get(3))
            input_height = int(cap.get(4))

            # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fourcc = cv2.VideoWriter_fourcc(*'h264')
            out = cv2.VideoWriter(output_path + '/out.mp4', fourcc, 25.0, (input_width, input_height))

        start = datetime.now()
        frame_counter = 0
        while cap.isOpened():
            ret, frame = cap.read()
            frame_counter += 1
            if not ret:
                break

            if visualise:
                coords, image = pd.predict(frame, th, (input_width, input_height), queue)
                queues_people = queue.check_coords(coords)

                # cv2.imshow("frame", image)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break

                # print(f"Total People in frame = {len(coords)}")
                # print(f"Number of people in queue = {queues_people}")

                out.write(frame)

                out_stats.write('{}, {}, {}\n'.format(frame_counter, len(coords), queues_people))

                # print('.', end='', flush=True)

                # if frame_counter < 10:
                #     cv2.imwrite("image_processed_{}.png".format(frame_counter), frame)

            else:
                coords = pd.predict(frame)
                print(coords)

        print(f'Total frames {frame_counter}, processing time: {datetime.now()-start}')

        if visualise:
            out.release()

        out_stats.close()
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference", e)
        traceback.print_exc()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--extensions', default=None)
    
    parser.add_argument('--visualise', action='store_true')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--max_people', default='To be given by you')
    parser.add_argument('--threshold', default='0.5')
    parser.add_argument('--output_path', default='To be given by you')
    
    args = parser.parse_args()

    main(args)
