# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 13:38:24 2019

@author: wattai
"""

import numpy as np
import cv2


cap = cv2.VideoCapture(0)

if cap.isOpened() is False:
    raise("IO Error")

scale = 0.02308  # 3.5 cm size of square
n_crosspoint = (9, 6)  # number of crosspoint on each axis

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((np.prod(n_crosspoint), 3), np.float32)
objp[:, :2] = scale * np.mgrid[:n_crosspoint[0], :n_crosspoint[1]].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

imgInd = 0
while True:
    ret, img = cap.read()
    if ret is False:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.putText(img, 'Number of capture: ' + str(imgInd),
                (30, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    cv2.putText(img, 'c: Capture the image', (30, 40),
                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    cv2.putText(img,
                'q: Finish capturing and calcurate the camera matrix and distortion',
                (30, 60), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0))
    cv2.imshow("Image", img)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('c'):
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, n_crosspoint, None)
        # If found, add object points, image points (after refining them)
        if ret is True:
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners,
                                        (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, n_crosspoint, corners2, ret)
            cv2.imshow('Image', img)
            cv2.waitKey(500)

            imgInd += 1

    if k == ord('q'):
        break

# Calc urate the camera matrix
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                   gray.shape[::-1],
                                                   None, None)

# Save the csv file
np.save('mtx.npy', mtx)
np.save('dist.npy', dist)
#np.savetxt("mtx.csv", mtx, delimiter=",")
#np.savetxt("dist.csv", dist, delimiter=",")

cap.release()
cv2.destroyAllWindows()


total_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i],
                                      mtx, dist)
    error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    total_error += error
print("total error: ", total_error/len(objpoints))
