import base64
import cv2
import zmq
import time 
context = zmq.Context()
footage_socket = context.socket(zmq.PUB)
footage_socket.connect('tcp://localhost:5505')

camera = cv2.VideoCapture("./test_video.mp4")  # init the camera

while True:
    try:
        grabbed, frame = camera.read()  # grab the current frame
        frame = cv2.resize(frame, (640, 480))  # resize the frame
        encoded, buffer = cv2.imencode('.jpg', frame)
        jpg_as_text = base64.b64encode(buffer)
        start = time.time()
        footage_socket.send(jpg_as_text)
        end = time.time()
        print("sent time: %f", 1/(end - start))

    except KeyboardInterrupt:
        camera.release()
        cv2.destroyAllWindows()
        break