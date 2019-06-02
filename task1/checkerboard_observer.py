# -*- coding: utf-8 -*-
"""
Created on Mon May 27 21:59:55 2019

@author: wattai
"""

import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

WIDTH = 640
HEIGHT = 480
FPS = 30


def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple(imgpts[2].ravel()), (0, 0, 255), 5)
    return img


class ExtrinsicParamsEstimator:

    def __init__(self, mtx, dist, n_crosspoint=(7, 5), length_square_side=1.0):
        """
        Parameter
        mtx: Camera Matrix consist of intrinsic parameters.
        dist: distortion parameters.
        n_crosspoint: the size of the checkerboard.
        length_square_size: the length of a side of a square.
        """
        self.K = mtx
        self.dist = dist
        self.n_crosspoint = n_crosspoint

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                         30, 0.001)
        self.objp = np.zeros((np.prod(n_crosspoint), 3), np.float32)
        self.objp[:, :2] = length_square_side * np.mgrid[
                            0:n_crosspoint[0], 0:n_crosspoint[1]
                            ].T.reshape(-1, 2)

        self.axis = np.float32([[3, 0, 0],
                                [0, 3, 0],
                                [0, 0, -3]]).reshape(-1, 3)*length_square_side
        self.R = None
        self.t = None

    def estimate_pose(self, img):
        """
        Paramter
        img: img that has a checkerboard.
        """
        img = img.astype('uint8')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray,
                                                 self.n_crosspoint, None)

        if ret is True:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11),
                                        (-1, -1), self.criteria)

            # Find the rotation and translation vectors.
            # When you chose Ransac, you need to add inliers for output.
            ret, rvecs, tvecs, inliner = cv2.solvePnPRansac(
                    self.objp, corners2,
                    self.K, self.dist, flags=cv2.SOLVEPNP_ITERATIVE)

            # project 3D points to image plane
            imgpts, jac = cv2.projectPoints(self.axis,
                                            rvecs, tvecs,
                                            self.K, self.dist)
            img = draw(img, corners2, imgpts)

            self.R = cv2.Rodrigues(rvecs)[0]
            self.t = np.copy(tvecs)

        cv2.imshow('img', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            None

        return ret, self.R, self.t

    def world2camera(self, x):
        x = x.reshape(3, -1)
        return (self.R @ x) + self.t

    def camera2world(self, x):
        x = x.reshape(3, -1)
        return (self.R.T @ x) - (self.R.T @ self.t)


if __name__ == '__main__':

    K = np.load('mtx.npy')
    dist = np.load('dist.npy')
    # K = np.loadtxt('mtx.csv', delimiter=',')
    # dist = np.loadtxt('dist.csv', delimiter=',')

    """
    # online estimation.
    estimator = ExtrinsicParamsEstimator(K, dist, n_crosspoint=(9, 6),
                                         length_square_side=0.02308)
    cap = cv2.VideoCapture(0)
    cap.set(3, width)
    cap.set(4, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if cap.isOpened() is False:
        raise("IO Error")
    while True:
        ret, img = cap.read()
        if ret is False:
            continue
        ret_, R, t = estimator.estimate_pose(img)
        if ret_:
            campos = estimator.camera2world(x=np.zeros([3, 1])).squeeze()
            print('x: %0.3f, y: %0.3f, z: %0.3f' % (campos[0],
                                                    campos[1],
                                                    campos[2]))
    cap.release()
    cv2.destroyAllWindows()
    """

    """
    # offline estimation.
    estimator = ExtrinsicParamsEstimator(K, dist, n_crosspoint=(7, 6),
                                         length_square_side=0.035)
    for fname in sorted(glob.glob('./samples/data/left*.jpg')):
        img = cv2.imread(fname, 1)
        R, t = estimator.estimate_pose(img)
        print(R, t)
    """

    # offline estimation from video file.
    RR = []
    tt = []
    ckb = ExtrinsicParamsEstimator(K, dist, n_crosspoint=(9, 6),
                                   length_square_side=0.02308)
    cap = cv2.VideoCapture('view_around.avi')
    for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if ret:
            ret, R, t = ckb.estimate_pose(frame)
            if ret:
                campos = ckb.camera2world(x=np.zeros([3, 1]))
                RR.append(R)
                tt.append(campos)
                # print(R, t)
                print(t[2])

    cap.release()
    cv2.destroyAllWindows()

    RR = np.array(RR)
    tt = np.array(tt)

    N = len(tt)
    y = np.linspace(-1, 1, N)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax = Axes3D(fig)
    ax.scatter(tt[:, 0, 0],
               tt[:, 1, 0],
               tt[:, 2, 0],
               "o",
               s=100,
               c=y,
               cmap=cm.seismic
               )
    ax.scatter(ckb.objp[0:, 0],
               ckb.objp[0:, 1],
               ckb.objp[0:, 2],
               "O",
               color="#ff0000",
               s=50,
               )
    # ax.set_title("Scatter Plot")
    plt.show()
