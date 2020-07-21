import cv2 as cv
vcap = cv.VideoCapture("rtsp://admin:Redbean2018@192.168.1.212:554")
while(1):
    ret, frame = vcap.read()
    cv.imshow('VIDEO', frame)
    cv.waitKey(1)
