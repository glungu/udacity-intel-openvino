from input_feeder import InputFeeder
from face_detection import ModelFaceDetection
from facial_landmarks_detection import ModelFacialLandmarksDetection
from head_pose_estimation import ModelHeadPoseEstimation
from gaze_estimation import ModelGazeEstimation
from mouse_controller import MouseController
import numpy as np
import cv2
import argparse
import logging


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


def init_logging(args):
    logging.basicConfig(
        filename='../bin/output_{}.log'.format(args.precision.lower()),
        format='%(levelname)s - %(message)s',
        level=args.log_level)


def init_models(args):
    global model_face, model_landmarks, model_hpose, model_gaze_estimation, mouse_controller
    precision = args.precision
    device = args.device
    threshold = args.threshold

    model_face = ModelFaceDetection(model_dir_face[precision], device, threshold)
    model_landmarks = ModelFacialLandmarksDetection(model_dir_landmarks[precision], device, threshold)
    model_hpose = ModelHeadPoseEstimation(model_dir_hpose[precision], device, threshold)
    model_gaze_estimation = ModelGazeEstimation(model_dir_gaze[precision], device, threshold)

    model_face.load_model()
    model_landmarks.load_model()
    model_hpose.load_model()
    model_gaze_estimation.load_model()

    mouse_controller = None
    if args.mouse_precision in ['high', 'low', 'medium'] and args.mouse_speed in ['fast', 'slow', 'medium']:
        mouse_controller = MouseController(args.mouse_precision, args.mouse_speed)


def process_single_frame(image, display_intermediate_output):
    model_face_output = model_face.predict(image)
    if model_face_output[2] is None:
        return 'No face detected', image

    model_landmarks_output = model_landmarks.predict(model_face_output[2])
    model_head_pose_output = model_hpose.predict(model_face_output[2])
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

    if display_intermediate_output:
        image = display_intermediate(image, model_face_output, model_landmarks_output, result)

    print('.', end='', flush=True)
    return result, image


def display_intermediate(image, model_face_output, model_landmarks_output, result):
    # face bounding box
    image = cv2.rectangle(image, model_face_output[0], model_face_output[1], (255, 0, 0), 2)

    # gaze vector projection
    res = (int(result[0] * 100), int(result[1] * 100))
    e1 = (model_face_output[0][0] + model_landmarks_output[0][0],
          model_face_output[0][1] + model_landmarks_output[0][1])
    e2 = (model_face_output[0][0] + model_landmarks_output[1][0],
          model_face_output[0][1] + model_landmarks_output[1][1])

    cv2.arrowedLine(image, e1, (e1[0] - res[0], e1[1] + res[1]), (0, 0, 255), 5)
    cv2.arrowedLine(image, e2, (e2[0] - res[0], e2[1] + res[1]), (0, 0, 255), 5)
    return image


def process_image(file_path, file_output, display_intermediate_output):
    feed = InputFeeder(input_type='image', input_file=file_path)
    feed.load_data()
    for batch in feed.next_batch():
        result, image = process_single_frame(batch, display_intermediate_output)
        # cv2.imshow('demo image', image)
        cv2.imwrite(file_output, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    feed.close()


def process_video(file_input, file_output, display_intermediate_output):
    if file_input is None:
        feed = InputFeeder(input_type='cam')
    else:
        feed = InputFeeder(input_type='video', input_file=file_input)

    feed.load_data()

    w = int(feed.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(feed.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(feed.cap.get(cv2.CAP_PROP_FPS))
    out = cv2.VideoWriter(file_output, cv2.VideoWriter_fourcc(*'avc1'), fps, (w, h), True)

    frame_counter = 0
    for batch in feed.next_batch():
        frame_counter += 1
        result, frame = process_single_frame(batch, display_intermediate_output)
        out.write(frame)

        logging.debug(f'Frame #{frame_counter} result: {result}')
        if type(result) == str and result == 'No face detected':
            logging.warning('Frame {}: No face detected', frame_counter)

        if mouse_controller is not None:
            mouse_controller.move(result[0], result[1])

    out.release()
    feed.close()


def main(args):
    init_logging(args)
    init_models(args)

    if args.type == 'image':
        output = '../bin/output_{}.png'.format(args.precision.lower())
        process_image(args.file, output, args.display_intermediate_output)
    elif args.type == 'video':
        output = '../bin/output_{}.mp4'.format(args.precision.lower())
        process_video(args.file, output, args.display_intermediate_output)
    elif args.type == 'cam':
        process_video(None)

    model_face.print_stats(header=True)
    model_landmarks.print_stats(header=False)
    model_hpose.print_stats(header=False)
    model_gaze_estimation.print_stats(header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', default='video',
                        help='Input type: image, video or cam.\n'
                             'Default: video')
    parser.add_argument('--file', default='../bin/demo.mp4',
                        help='Input image or video file.\n'
                             'Default: ../bin/demo.mp4')
    parser.add_argument('--precision', default='FP16',
                        help='Model precision parameter (FP32, FP16 or INT8). '
                             'Each model in the pipeline will be used with this precision if available.\n'
                             'Default: FP16')
    parser.add_argument('--device', default='CPU',
                        help='Device to run inference on (CPU, GPU, VPU, FPGA).\n'
                             'Default: CPU')
    parser.add_argument('--threshold', default=0.5,
                        help='Confidence threshold to use with Face Detection model.\n'
                             'Default: 0.5')
    parser.add_argument('--display_intermediate_output', default=True,
                        help='Whether to display intermediate output, '
                             'like face bounding box and gaze vector projection.\n'
                             'Default: True')
    parser.add_argument('--log_level', default='INFO',
                        help='Logging level to use.\n'
                             'Default: INFO')
    parser.add_argument('--mouse_precision', default='',
                        help='Mouse movement precision, optional')
    parser.add_argument('--mouse_speed', default='',
                        help='Mouse movement speed, optional')
    parsed_args = parser.parse_args()
    main(parsed_args)
