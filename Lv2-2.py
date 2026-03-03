import cv2
import numpy as np
import os
import glob
import pickle

def calibrate_camera():
    CHECKERBOARD = (6,8) 
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    objpoints = []
    imgpoints = [] 

    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    
    images = glob.glob('/home/urim/vision_task/checkerboards/*.png')
    
    print(len(images))

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ret, corners = cv2.findChessboardCorners(gray,
                                               CHECKERBOARD,
                                               cv2.CALIB_CB_ADAPTIVE_THRESH +
                                               cv2.CALIB_CB_FAST_CHECK +
                                               cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        if ret == True:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners2)
            
            img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                      gray.shape[::-1], None, None)
    
    print("Camera matrix : \n")
    print(mtx)
    print("\ndist : \n")
    print(dist)
    print("\nrvecs : \n")
    print(rvecs)
    print("\ntvecs : \n")
    print(tvecs)
    
    calibration_data = {
        'camera_matrix': mtx,
        'dist_coeffs': dist,
        'rvecs': rvecs,
        'tvecs': tvecs
    }
    
    with open('camera_calibration.pkl', 'wb') as f:
        pickle.dump(calibration_data, f)
    
    return calibration_data

def live_video_correction(calibration_data):
    mtx = calibration_data['camera_matrix']
    dist = calibration_data['dist_coeffs']
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        h, w = frame.shape[:2]
        
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
        
        dst = cv2.undistort(frame, mtx, dist, None, newcameramtx)
        
        x, y, w, h = roi
        if all(v > 0 for v in [x, y, w, h]):
            dst = dst[y:y+h, x:x+w]
        
        original = cv2.resize(frame, (640, 480))
        corrected = cv2.resize(dst, (640, 480))
        combined = np.hstack((original, corrected))
        
        cv2.imshow('Corrected', combined)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if os.path.exists('camera_calibration.pkl'):
        print("Loading existing calibration data...")
        with open('camera_calibration.pkl', 'rb') as f:
            calibration_data = pickle.load(f)
    else:
        print("Performing new camera calibration...")
        calibration_data = calibrate_camera()
    
    print("Starting live video correction...")
    live_video_correction(calibration_data)