from flask import Flask,request, jsonify, Response,send_file
import numpy as np
import cv2
import base64
import time
import zmq
from threading import Thread


app = Flask(__name__)


@app.route('/')
def hello():
    return "Hello World!"
@app.route('/post', methods=['POST','GET'])
def test():
    # #read image file string data
    # filestr = request.files['file'].read()
    # #convert string data to numpy array
    # npimg = np.fromstring(filestr, np.uint8)
    # # convert numpy array to image
    # img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
    r = request
    # convert string of image data to uint8
    nparr = np.fromstring(r.data, np.uint8)
    # decode image
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  
    return ""

class Worker(Thread):
    def __init__(self):
        Thread.__init__(self)
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self.active = True

    def run(self):
        self._socket.bind('tcp://*:5505')
        self._socket.setsockopt_string(zmq.SUBSCRIBE, np.unicode(''))
        while self.active:
            start = time.time()
            frame = self._socket.recv_string()
            img = base64.b64decode(frame)
            npimg = np.fromstring(img, dtype=np.uint8)
            source = cv2.imdecode(npimg, 1)
            end = time.time()
            print("receive time: %f", (end - start))
            cv2.imshow("Stream", source)

            cv2.waitKey(1)


if __name__ == '__main__':
    worker = Worker()
    worker.start()
    app.run(host='0.0.0.0', port=5000)