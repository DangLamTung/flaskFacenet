import cv2
import numpy as np
import requests
import time
addr = 'http://localhost:5000'
test_url = addr + '/post'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}


camera = cv2.VideoCapture("./test_video.mp4")  # init the camera

while True:
    try:
        start = time.time()
        grabbed, frame = camera.read()  # grab the current frame
        img = cv2.resize(frame, (640, 480))  # resize the frame
        _, img_encoded = cv2.imencode('.jpg', img)

        response = requests.post(test_url, data=img_encoded.tostring(), headers=headers)
        # footage_socket.send(jpg_as_text)
        end = time.time()
        print("inference time: ", 1/(end - start))

    except KeyboardInterrupt:
        camera.release()
        cv2.destroyAllWindows()
        break