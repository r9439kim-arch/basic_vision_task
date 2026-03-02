import cv2
import numpy as np

def main():
    cam=cv2.VideoCapture(0)

    while True:
        ret,frame=cam.read()

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        blur=cv2.GaussianBlur(gray,(5,5),0)
        binary=cv2.Canny(blur,50,150)

        contours,hierarchy=cv2.findContours(binary,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)

        cv2.drawContours(frame,contours,-1,(0,200,0),1)

        cv2.imshow('contours',frame)

        if cv2.waitKey(1)==113:
            break

if __name__=='__main__':
    main()