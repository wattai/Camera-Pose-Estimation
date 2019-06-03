# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:18:52 2019

@author: wattai
"""

import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

import checkerboard_observer as ckb

# parameter for opencv webcam capture instance.
WIDTH = 640
HEIGHT = 480
FPS = 30

# paramter for choosing the combination of images to extract featurepoints.
SHIFT_WIDTH = 15
ROLL_WIDTH = 15

# parameter for down sampling to choose good featurepoitns.
FEATURE_DROP_RATIO = 0.75


def outer(a):
    return np.array([[0, -a[2], a[1]],
                     [a[2], 0, -a[0]],
                     [-a[1], a[0], 0]])


class CameraPoseEstimatorBasedOnPointCould:
    def __init__(self, mtx, dist, X_3d, X_pts, X_desc,
                 board_observer, feature_detector, feature_matcher,
                 feature_drop_ratio=0.75):
        self.K = mtx
        self.dist = dist
        self.X_3d = X_3d
        self.X_pts = X_pts
        self.X_desc = X_desc
        self.ckb = board_observer
        self.detector = feature_detector
        self.matcher = feature_matcher
        self.feature_drop_ratio = feature_drop_ratio

    def estimate_pose(self, img):
        R_ckb, t_ckb, R_ptc, t_ptc = None, None, None, None
        ret_match, self.matches = self.extract_match_featpoints(img)
        if ret_match:
            if len(self.matches) < 6:
                raise(ValueError(
                        'the number of matched points is less than 6.'))

            X_3d_matched = np.array(
                    [self.X_3d[m[0].queryIdx] for m in self.matches])
            X_2d_matched = np.array(
                    [self.kp[m[0].trainIdx].pt for m in self.matches])

            ret_ckb, R_ckb_, t_ckb_ = self.ckb.estimate_pose(img)
            if ret_ckb:
                # Find the rotation and translation vectors.
                # When you chose Ransac, you need to add inliers for output.
                ret_PnP, rvecs, tvecs, inliner = cv2.solvePnPRansac(
                        X_3d_matched, X_2d_matched, self.K, self.dist,
                        # rvec=cv2.Rodrigues(R_ckb_)[0], # tvec=t_ckb_,
                        useExtrinsicGuess=False, flags=cv2.SOLVEPNP_ITERATIVE)
                if ret_PnP:
                    R_ptc_ = cv2.Rodrigues(rvecs)[0]
                    t_ptc_ = tvecs.copy()
                    R_ptc = R_ptc_.T
                    t_ptc = np.dot(-R_ptc_.T, t_ptc_)
                    assert(R_ptc.shape == (3, 3))
                    assert(t_ptc.shape == (3, 1))
                    R_ckb = R_ckb_.T
                    t_ckb = np.dot(-R_ckb_.T, t_ckb_)
        ret = ret_match and ret_ckb
        return ret, R_ckb, t_ckb, R_ptc, t_ptc

    def extract_match_featpoints(self, img):
        # initialize variables.
        ret = False  # succecss flag.
        good_featpoints = []  # list for good featurepoints.

        # undistortion
        img = cv2.undistort(img, self.K, self.dist)

        # detect featurepoints and compute feature vector.
        self.kp, self.desc = self.detector.detectAndCompute(img, None)
        if self.desc is not None:
            ret = True

        if ret:
            # match featurepoints.
            matches = self.matcher.knnMatch(self.X_desc.astype('uint8'),
                                            self.desc.astype('uint8'),
                                            k=2)

            # down sample from matched points.
            for m, n in matches:
                if m.distance < self.feature_drop_ratio * n.distance:
                    good_featpoints.append([m])

        return ret, good_featpoints.copy()


class TriangulatorBasedOnCheckerBoard:
    def __init__(self, mtx, dist, board_observer,
                 feature_detector, feature_matcher,
                 feature_drop_ratio=0.75):
        self.K = mtx
        self.dist = dist
        self.ckb = board_observer
        self.detector = feature_detector
        self.matcher = feature_matcher
        self.feature_drop_ratio = feature_drop_ratio

    def reconstruct3d(self, imgs):
        self.matches = self.extract_match_featpoints(imgs)
        if len(self.matches) < 8:
            raise(ValueError('the number of matched points is less than 8.'))

        pts1 = np.array([self.kp1[m[0].queryIdx].pt for m in self.matches])
        pts2 = np.array([self.kp2[m[0].trainIdx].pt for m in self.matches])
        desc1 = np.array([self.desc1[m[0].queryIdx] for m in self.matches])
        desc2 = np.array([self.desc1[m[0].queryIdx] for m in self.matches])

        pts1 = np.float32(pts1).copy()
        pts2 = np.float32(pts2).copy()

        # find Fundamental Matrix
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)
        # remove outlier.
        pts1 = pts1[mask.ravel() == 1]
        pts2 = pts2[mask.ravel() == 1]
        desc1 = desc1[mask.ravel() == 1]
        desc2 = desc2[mask.ravel() == 1]

        x1 = pts1[:, 1].astype('int')
        y1 = pts1[:, 0].astype('int')
        x2 = pts2[:, 1].astype('int')
        y2 = pts2[:, 0].astype('int')
        self.featcolor1 = imgs[0][x1, y1, ::-1]
        self.featcolor2 = imgs[1][x2, y2, ::-1]

        ret1, R1, t1 = self.ckb.estimate_pose(imgs[0])
        ret2, R2, t2 = self.ckb.estimate_pose(imgs[1])
        print(ret1, ret2)
        ret = (ret1 and ret2)
        X_3d = None
        pts1_matched = None
        desc1_matched = None
        if ret:
            M1 = np.hstack((R1, t1))
            M2 = np.hstack((R2, t2))

            self.P1 = np.dot(self.K,  M1)
            self.P2 = np.dot(self.K,  M2)

            X_4d = cv2.triangulatePoints(self.P1, self.P2, pts1.T, pts2.T)
            X_3d = (X_4d[:3, :] / X_4d[-1, :]).T

            pts1_matched = pts1.copy()
            desc1_matched = desc1.copy()
        return ret, X_3d, pts1_matched, desc1_matched

    def extract_match_featpoints(self, imgs):
        # undistortion
        img1 = cv2.undistort(imgs[0], self.K, self.dist)
        img2 = cv2.undistort(imgs[1], self.K, self.dist)

        # detect featurepoints and compute feature vector.
        self.kp1, self.desc1 = self.detector.detectAndCompute(img1, None)
        self.kp2, self.desc2 = self.detector.detectAndCompute(img2, None)

        # match featurepoints.
        matches = self.matcher.knnMatch(self.desc1, self.desc2, k=2)

        # down sample from matched points.
        good_featpoints = []
        for m, n in matches:
            if m.distance < self.feature_drop_ratio * n.distance:
                good_featpoints.append([m])
        """
        # 特徴量をマッチング状況に応じてソートする
        # good = sorted(matches, key=lambda x: x[1].distance)

        # 対応する特徴点同士を描画
        img3 = cv2.drawMatchesKnn(img1, self.kp1, img2, self.kp2,
                                  good[:30], None, flags=2)

        # 画像表示
        cv2.imshow('img', img3)

        # キー押下で終了
        cv2.waitKey(1)
        # cv2.destroyAllWindows()
        """
        return good_featpoints.copy()


if __name__ == "__main__":
    # load camera intrinsic params.
    K = np.load('../params/mtx.npy')
    dist = np.load('../params/dist.npy')

    # make the instance of the checker board observer.
    board_observer = ckb.ExtrinsicParamsEstimator(
            K, dist, n_crosspoint=(9, 6), length_square_side=0.02308)

    # make feature-point detector.
    feature_detector = cv2.AKAZE_create(threshold=0.001)

    # make feature-point matcher.
    feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    """
    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    FLANN_INDEX_LSH = 1
    index_params= dict(algorithm = FLANN_INDEX_KDTREE,
                       table_number = 12, # 12
                       key_size = 20,     # 20
                       multi_probe_level = 2) #2
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(self.desc1.astype('float32'),
                             self.desc2.astype('float32'), k=2)
    """

    # make pointcloud from video file.
    triangulator = TriangulatorBasedOnCheckerBoard(
            K, dist, board_observer,
            feature_detector, feature_matcher,
            feature_drop_ratio=FEATURE_DROP_RATIO)

    # state variables.
    X_3d = np.zeros([0, 3])
    X_pts = np.zeros([0, 2])
    X_desc = None
    colors = np.zeros([0, 3])
    X_3d_cam = np.zeros([0, 3])
    Viewvecs_cam = np.zeros([0, 3])

    # load captured video.
    cap = cv2.VideoCapture('../video/output.avi')
    frames = []
    for i in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    frames = np.array(frames)
    cap.release()
    cv2.destroyAllWindows()

    # train from captured video.
    for f_p, f_n in zip(frames[::SHIFT_WIDTH],
                        np.roll(frames, ROLL_WIDTH)[::SHIFT_WIDTH]):
        ret, x_3d, pts1, desc1 = triangulator.reconstruct3d(
                imgs=[f_p, f_n])
        if ret:
            X_3d = np.concatenate((X_3d, x_3d), axis=0)
            # store the 2D-keypoints and the descriptors.
            X_pts = np.concatenate((X_pts, pts1), axis=0)
            if X_desc is None:
                X_desc = np.zeros([0, desc1.shape[1]])
            X_desc = np.concatenate((X_desc, desc1), axis=0)

            x_3d_cam = triangulator.ckb.camera2world(x=np.zeros([3, 1]))
            X_3d_cam = np.concatenate((X_3d_cam, x_3d_cam.T), axis=0)
            viewvec_cam = triangulator.ckb.camera2world(
                    x=np.array([0, 0, 0.1]))
            Viewvecs_cam = np.concatenate((Viewvecs_cam, viewvec_cam.T),
                                          axis=0)
            colors = np.concatenate((colors, triangulator.featcolor1), axis=0)
            """
            x_2d = reconstructor.project(x_3d, estimator.P1)
            plt.figure()
            plt.imshow(f_p[:, :, ::-1])
            plt.scatter(x_2d[:, 0], x_2d[:, 1], c=estimator.featcolor1/255)
            plt.xlim(0, f_p.shape[1])
            plt.ylim(0, f_p.shape[0])
            plt.show()
            """

    # visualizaiton of 3D pointcloud.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_3d[0:, 0],
               X_3d[0:, 1],
               X_3d[0:, 2],
               "o",
               s=5,
               c=colors/255,
               )

    ax.plot(X_3d_cam[0:, 0],
            X_3d_cam[0:, 1],
            X_3d_cam[0:, 2],
            "o-",
            color="#00aa00",
            markersize=10,
            linewidth=3,
            )

    for i in range(len(X_3d_cam)):
        ax.plot(np.array([X_3d_cam[i, 0], Viewvecs_cam[i, 0]]),
                np.array([X_3d_cam[i, 1], Viewvecs_cam[i, 1]]),
                np.array([X_3d_cam[i, 2], Viewvecs_cam[i, 2]]),
                "-",
                color="#aa0000",
                markersize=10,
                linewidth=3,
                alpha=0.5,
                )

    ax.scatter(triangulator.ckb.objp[0:, 0],
               triangulator.ckb.objp[0:, 1],
               triangulator.ckb.objp[0:, 2],
               "O",
               color="#ff0000",
               s=100,
               )
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_title("3D Point Cloud")
    plt.tight_layout()
    plt.show()

    # make pose estimator.
    estimator = CameraPoseEstimatorBasedOnPointCould(
            K, dist, X_3d, X_pts, X_desc, board_observer,
            feature_detector, feature_matcher,
            feature_drop_ratio=FEATURE_DROP_RATIO)

    # make opencv webcam capture instance.
    cap = cv2.VideoCapture(0)
    cap.set(3, WIDTH)
    cap.set(4, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    # start realtime camera pose estimation.
    while cap.isOpened():
        ret_cap, frame = cap.read()
        if ret_cap:
            ret_est, R_ckb, t_ckb, R_ptc, t_ptc = estimator.estimate_pose(
                    img=frame)
            if ret_est:
                print('distance_R: %0.4f, distance_t: %0.4f' % (
                        np.linalg.norm(R_ckb - R_ptc),
                        np.linalg.norm(t_ckb - t_ptc)))

    cap.release()
    cv2.destroyAllWindows()
