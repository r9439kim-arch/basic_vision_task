import cv2
from ultralytics import YOLO

model=YOLO("/home/urim/Downloads/best.pt")

def main():
    cap=cv2.VideoCapture(0)

    while True:
        ret,frame=cap.read()

        if not ret:
            print("not frmae")
            break

        results=model(frame,conf=0.5)

        annotated_frame=results[0].plot()

        cv2.imshow("task6",annotated_frame)

        if cv2.waitKey(1)==113:
            break
    
if __name__ == "__main__":
    main()