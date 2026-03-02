import cv2
import numpy as np

def main():
    image=cv2.imread("/home/urim/vision_task/image.jpg")
    
    RGB=image[:,:,::-1]

    print("RGB type: ",type(RGB))
    print("RGB shape: ",RGB.shape)
    print("\n")

    cv2.imshow("RGB",RGB)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()
 