import numpy as np
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IEPlugin
import os
import cv2
import argparse

class Queue:
    '''
    Class for dealing with queues
    '''
    def __init__(self):
        self.queues=[]

    def add_queue(self, points):
        self.queues.append(points)

    def get_queues(self, image):
        for q in self.queues:
            x_min, y_min, x_max, y_max=q
            frame=image[y_min:y_max, x_min:x_max]
            yield frame
    
    def check_coords(self, coords):
        d={k+1:0 for k in range(len(self.queues))}
        for coord in coords:
            for i, q in enumerate(self.queues):
                if coord[0]>q[0] and coord[2]<q[2]:
                    d[i+1]+=1
        return d

class PersonDetect:
    '''
    Class for the Person Detection Model.
    '''

    def __init__(self):
    '''
    TODO: This method needs to be completed by you
    '''
    raise NotImplementedError

    def load_model(self):
    '''
    TODO: This method needs to be completed by you
    '''
    raise NotImplementedError

    def check_plugin(self, plugin):
    '''
    TODO: This method needs to be completed by you
    '''
    raise NotImplementedError
        
    def predict(self, image):
    '''
    TODO: This method needs to be completed by you
    '''
    raise NotImplementedError

    def preprocess_outputs(self, outputs):
    '''
    TODO: This method needs to be completed by you
    '''
    raise NotImplementedError

    def preprocess_input(self, image):
    '''
    TODO: This method needs to be completed by you
    '''
    raise NotImplementedError

def main(args):
    extensions=args.extensions
    model=args.model
    device=args.device
    visualise=args.visualise

    start=time.time()
    pd=PersonDetect()
    pd.load_model()
    print(f"Time taken to load the model is: {time.time()-start}")
    
     #Queue Parameters
        # For retail
        #queue.add_queue([620, 1, 915, 562])
        #queue.add_queue([1000, 1, 1264, 461])
        # For manufacturing
        #queue.add_queue([15, 180, 730, 780])
        #queue.add_queue([921, 144, 1424, 704])
        # For Transport 
        #queue.add_queue([50, 90, 838, 794])
        #queue.add_queue([852, 74, 1430, 841])


    try:
        queue=Queue()
        queue.add_queue([620, 1, 915, 562])
        queue.add_queue([1000, 1, 1264, 461])
        video_file=args.video
        cap=cv2.VideoCapture(video_file)

        while cap.isOpened():
            ret, frame=cap.read()
            if not ret:
                continue

            print(f"Total People in frame = {len(coords)}")
            print(f"Number of people in queue = {num_people}")
        
            if visualise:
                coords, image=pd.predict(frame)
                num_people=queue.check_coords(coords)
                cv2.imshow("frame", image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                coords=pd.predict(frame)
                print(coords)

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print("Could not run Inference", e)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--extensions', default=None)
    
    parser.add_argument('--visualise', action='store_true')
    parser.add_argument('--video', default=None)
    parser.add_argument('--queue_param', default=None)
    parser.add_argument('--max_people', default='To be given by you')
    parser.add_argument('--threshold', default='To be given by you')
    
    args=parser.parse_args()

    main(args)

