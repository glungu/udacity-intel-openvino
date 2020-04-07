import cv2

print('OpenCV version:', cv2.__version__)
print('OpenCV build:', cv2.getBuildInformation())

video_file = 'resources/manufacturing.mp4'
cap = cv2.VideoCapture(video_file)
cap.open(video_file)

input_width = int(cap.get(3))
input_height = int(cap.get(4))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('simple_out.mp4', fourcc, 25.0, (input_width, input_height))

# out = cv2.VideoWriter('resources/manufacturing_out.mp4', 0x00000021, 25, (input_width, input_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    out.write(frame)
    print('.', end='', flush=True)

out.release()
cap.release()
cv2.destroyAllWindows()





