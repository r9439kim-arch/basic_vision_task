import cv2
import torch

def main():
    model=torch.hub.load('ultralytics/yolov5','yolov5s')

    im=cv2.VideoCapture(0)

    while True:
        ret,frame=im.read()
        results=model(frame)
        print(type(results))
        results.print()

        frame=results.render()[0]

        cv2.imshow("Object detection",frame)
        if cv2.waitKey(1)==113:
            break

if __name__ == "__main__":
    main()

