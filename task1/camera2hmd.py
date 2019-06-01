# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 13:06:22 2019

@author: wattai
"""

import cv2
import numpy as np

aruco = cv2.aruco

# WEBカメラ
cap = cv2.VideoCapture(0)

# dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
parameters =  aruco.DetectorParameters_create()

# CORNER_REFINE_NONE, no refinement. CORNER_REFINE_SUBPIX,
# do subpixel refinement. CORNER_REFINE_CONTOUR use contour-Points

parameters.cornerRefinementMethod = aruco.CORNER_REFINE_CONTOUR

"""
cameraMatrix = np.array( [[1.42068235e+03,0.00000000e+00,9.49208512e+02],
    [0.00000000e+00,1.37416685e+03,5.39622051e+02],
    [0.00000000e+00,0.00000000e+00,1.00000000e+00]] )
distCoeffs = np.array( [1.69926613e-01,-7.40003491e-01,-7.45655262e-03,
                        -1.79442353e-03, 2.46650225e+00] )
"""

cameraMatrix = np.loadtxt("mtx.csv", delimiter=",")
distCoeffs = np.loadtxt("dist.csv", delimiter=",")

cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

markerLength = 10.0


markers_x = 5
markers_y = 7
marker_length = 1.6
marker_separation = 0.4
grid_board = aruco.GridBoard_create(markers_x, markers_y, marker_length,
                                    marker_separation, dictionary)


def main():

    ret, frame = cap.read()

    # 変換処理ループ
    while ret is True:
        corners, ids, rejectedImgPoints = aruco.detectMarkers(
                frame, dictionary, parameters=parameters)

        aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))

        for i, corner in enumerate(corners):
            points = corner[0].astype(np.int32)
            cv2.polylines(frame, [points], True, (0, 255, 255))
            cv2.putText(frame, str(ids[i][0]), tuple(points[0]),
                        cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1)

        # for multiple markers. --------------------------
        retval, rvec, tvec = aruco.estimatePoseBoard(corners, ids,
                                                     grid_board,
                                                     cameraMatrix,
                                                     distCoeffs)
        if retval != 0:
            R = cv2.Rodrigues(rvec)[0]
            d_tvec = np.array([(markers_x//2) * (
                    marker_length + marker_separation) + marker_length/2.,
                               (markers_y//2) * (
                    marker_length + marker_separation) + marker_length/2.,
                               0])
            d_tvec_R = np.dot(R, d_tvec).reshape(3, 1)
            tvec = tvec + d_tvec_R

            tvec_2d = np.dot(cameraMatrix, tvec)
            tvec_2d = tvec_2d[:2] / tvec_2d[2]
            aruco.drawAxis(frame, cameraMatrix, distCoeffs,
                           rvec, tvec, 2*marker_length)
        # --------------------------------------------------
        """
        # for single marker. -------------------------------
        rvecs, tvecs, _objPoints = aruco.estimatePoseSingleMarkers(
                corners, marker_length, cameraMatrix, distCoeffs)
        if ids is not None:
            for i in range( ids.size ):
                # print( 'rvec {}, tvec {}'.format( rvecs[i], tvecs[i] ))
                # print( 'rvecs[{}] {}'.format( i, rvecs[i] ))
                # print( 'tvecs[{}] {}'.format( i, tvecs[i] ))
                aruco.drawAxis(frame, cameraMatrix, distCoeffs,
                               rvecs[i], tvecs[i], 2*markerLength)
        # --------------------------------------------------
        """
        cv2.imshow('org', frame)

        # Escキーで終了
        key = cv2.waitKey(50)
        if key == 27:  # ESC
            break

        # 次のフレーム読み込み
        ret, frame = cap.read()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
