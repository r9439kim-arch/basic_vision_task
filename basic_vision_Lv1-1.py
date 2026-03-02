import cv2

def main():
    cap=cv2.VideoCapture(0)

    while True:
        ret,frame=cap.read()

        if not ret:
            print("not frame")
            break
        
        cv2.imshow("Streaming",frame)

        if cv2.waitKey(1)==113:
            break

if __name__ == "__main__":
    main()