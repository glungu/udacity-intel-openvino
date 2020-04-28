from input_feeder import InputFeeder
from face_detection import ModelFaceDetection
from facial_landmarks_detection import ModelFacialLandmarksDetection
from head_pose_estimation import ModelHeadPoseEstimation
from gaze_estimation import ModelGazeEstimation
from mouse_controller import MouseController
import numpy as np
import cv2
import argparse


def init(args):
    global model_face_detection, model_landmarks, model_head_pose, model_gaze_estimation, mouse_controller
    model_face_detection = ModelFaceDetection(
        '../models/intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001')
    model_landmarks = ModelFacialLandmarksDetection(
        '../models/intel/landmarks-regression-retail-0009/FP16/landmarks-regression-retail-0009'
    )
    model_head_pose = ModelHeadPoseEstimation(
        '../models/intel/head-pose-estimation-adas-0001/FP16/head-pose-estimation-adas-0001'
    )
    model_gaze_estimation = ModelGazeEstimation(
        '../models/intel/gaze-estimation-adas-0002/FP16/gaze-estimation-adas-0002'
    )
    model_face_detection.load_model()
    model_landmarks.load_model()
    model_head_pose.load_model()
    model_gaze_estimation.load_model()
    mouse_controller = MouseController(args.precision, args.speed)


def process_single_frame(image):
    model_face_detection_output = model_face_detection.predict(image)
    model_landmarks_output = model_landmarks.predict(model_face_detection_output[2])
    model_head_pose_output = model_head_pose.predict(model_face_detection_output[2])
    gaze_estimation_output = model_gaze_estimation.predict(
        (model_landmarks_output[2], model_landmarks_output[3]), model_head_pose_output)

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
    cv2.arrowedLine(image, e1, (e1[0] - res[0], e1[1] - res[1]), (0, 0, 255), 5)
    cv2.arrowedLine(image, e2, (e2[0] - res[0], e2[1] - res[1]), (0, 0, 255), 5)

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


def process_video(file_path):
    feed = InputFeeder(input_type='video', input_file=file_path)
    feed.load_data()

    w = int(feed.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(feed.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    f = int(feed.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(feed.cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter('../bin/output_video.mp4', cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h), True)

    for batch in feed.next_batch():
        result, frame = process_single_frame(batch)
        out.write(frame)
        mouse_controller.move(result[0], result[1])

    out.release()
    feed.close()


def main(args):
    init(args)
    if args.type == 'image':
        process_image(args.file)
    elif args.type == 'video':
        process_video(args.file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='video')
    parser.add_argument('--file', default='../bin/demo.mp4')
    parser.add_argument('--precision', default='high')
    parser.add_argument('--speed', default='fast')
    parsed_args = parser.parse_args()
    main(parsed_args)
