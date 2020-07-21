import socket
import time
import sys
import cv2

UDP_IP = "127.0.0.1"
UDP_PORT = 5005
buf = 100000000
file_name = sys.argv[1]

img = cv2.imread(file_name)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.sendto(img, (UDP_IP, UDP_PORT))
print ("Sending %s ..." % file_name)

f = open(file_name, "r")
data = f.read(buf)
while(data):
    if(sock.sendto(data, (UDP_IP, UDP_PORT))):
        data = f.read(buf)
        time.sleep(0.02) # Give receiver a bit time to save

sock.close()
f.close()