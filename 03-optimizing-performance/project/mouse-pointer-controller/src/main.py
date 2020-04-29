from input_feeder import InputFeeder
from face_detection import ModelFaceDetection
from facial_landmarks_detection import ModelFacialLandmarksDetection
from head_pose_estimation import ModelHeadPoseEstimation
from gaze_estimation import ModelGazeEstimation
from mouse_controller import MouseController
import numpy as np
import cv2
import argparse


model_dir = '../models/intel/'
model_dir_face = {
    'FP32': model_dir + 'face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001',
    'FP16': model_dir + 'face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001',
    'INT8': model_dir + 'face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001'
}
model_dir_landmarks = {
    'FP32': model_dir + 'landmarks-regression-retail-0009/FP32/landmarks-regression-retail-0009',
    'FP16': model_dir + 'landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009',
    'INT8': model_dir + 'landmarks-regression-retail-0009/FP16-INT8/landmarks-regression-retail-0009'
}
model_dir_hpose = {
    'FP32': model_dir + 'head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001',
    'FP16': model_dir + 'head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001',
    'INT8': model_dir + 'head-pose-estimation-adas-0001/FP16-INT8/head-pose-estimation-adas-0001'
}
model_dir_gaze = {
    'FP32': model_dir + 'gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002',
    'FP16': model_dir + 'gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002',
    'INT8': model_dir + 'gaze-estimation-adas-0002/FP16-INT8/gaze-estimation-adas-0002'
}


def init(args):
    global model_face, model_landmarks, model_hpose, model_gaze_estimation, mouse_controller
    quantize = args.quantize

    model_face = ModelFaceDetection(model_dir_face[quantize])
    model_landmarks = ModelFacialLandmarksDetection(model_dir_landmarks[quantize])
    model_hpose = ModelHeadPoseEstimation(model_dir_hpose[quantize])
    model_gaze_estimation = ModelGazeEstimation(model_dir_gaze[quantize])

    model_face.load_model()
    model_landmarks.load_model()
    model_hpose.load_model()
    model_gaze_estimation.load_model()

    mouse_controller = None
    if args.precision in ['high', 'low', 'medium'] and args.speed in ['fast', 'slow', 'medium']:
        mouse_controller = MouseController(args.precision, args.speed)


def process_single_frame(image):
    model_face_detection_output = model_face.predict(image)
    model_landmarks_output = model_landmarks.predict(model_face_detection_output[2])
    model_head_pose_output = model_hpose.predict(model_face_detection_output[2])
    gaze_estimation_output = model_gaze_estimation.predict(
        ((model_landmarks_output[2], model_landmarks_output[3]), model_head_pose_output))

    # rotation vector
    rvec = np.array([0, 0, 0], np.float)
    # translation vector
    tvec = np.array([0, 0, 0], np.float)
    # camera matrix
    camera_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float)

    result, _ = cv2.projectPoints(gaze_estimation_output, rvec, tvec, camera_matrix, None)
    result = result[0][0]
    res = (int(result[0] * 100), int(result[1] * 100))

    e1 = (model_face_detection_output[0][0] + model_landmarks_output[0][0],
          model_face_detection_output[0][1] + model_landmarks_output[0][1])
    e2 = (model_face_detection_output[0][0] + model_landmarks_output[1][0],
          model_face_detection_output[0][1] + model_landmarks_output[1][1])
    cv2.arrowedLine(image, e1, (e1[0] - res[0], e1[1] + res[1]), (0, 0, 255), 5)
    cv2.arrowedLine(image, e2, (e2[0] - res[0], e2[1] + res[1]), (0, 0, 255), 5)

    print('.', end='', flush=True)
    return result, image


def process_image(file_path):
    feed = InputFeeder(input_type='image', input_file=file_path)
    feed.load_data()
    for batch in feed.next_batch():
        result, image = process_single_frame(batch)
        cv2.imshow('demo image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    feed.close()


def process_video(file_input, file_output):
    if file_input is None:
        feed = InputFeeder(input_type='cam')
    else:
        feed = InputFeeder(input_type='video', input_file=file_input)

    feed.load_data()

    w = int(feed.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(feed.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(feed.cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(file_output, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h), True)

    for batch in feed.next_batch():
        result, frame = process_single_frame(batch)
        out.write(frame)
        if mouse_controller is not None:
            mouse_controller.move(result[0], result[1])

    out.release()
    feed.close()


def main(args):
    init(args)

    if args.type == 'image':
        process_image(args.file)
    elif args.type == 'video':
        output = '../bin/output_{}.mp4'.format(args.quantize.lower())
        process_video(args.file, output)
    elif args.type == 'cam':
        process_video(None)

    model_face.print_stats(header=True)
    model_landmarks.print_stats(header=False)
    model_hpose.print_stats(header=False)
    model_gaze_estimation.print_stats(header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='video')
    parser.add_argument('--file', default='../bin/demo.mp4')
    parser.add_argument('--quantize', default='FP16')
    parser.add_argument('--precision', default='')
    parser.add_argument('--speed', default='')
    parsed_args = parser.parse_args()
    main(parsed_args)
