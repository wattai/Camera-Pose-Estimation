# -*- coding: utf-8 -*-
"""
Created on Tue May 28 21:18:52 2019

@author: wattai
"""

import glob
import numpy as np
import cv2
import triangulation as reconstructor
import projection_matrix as P_estimator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

import extrinsic_params_estimator as camera

def out(a):
    return np.array([[0, -a[2], a[1]],
                     [a[2], 0, -a[0]],
                     [-a[1], a[0], 0]])
    
class CameraPoseEstimatorBasedOnPointCould:
    def __init__(self, mtx, dist, X_3d, X_pts, X_desc):
        self.K = mtx
        self.dist = dist
        self.X_3d = X_3d
        self.X_pts = X_pts
        self.X_desc = X_desc
        self.ckb = camera.ExtrinsicParamsEstimator(self.K, self.dist,
                                                   n_crosspoint=(9, 6),
                                                   length_square_side=0.02308)

    def estimate_pose(self, img):
        R_ckb, t_ckb, R_ptc, t_ptc = None, None, None, None
        self.matches = self.extract_match_featpoints(img)
        if len(self.matches) < 6:
            raise(ValueError('the number of matched points is less than 6.'))

        X_3d_matched = np.array([self.X_3d[m[0].queryIdx] for m in self.matches])
        X_2d_matched = np.array([self.kp[m[0].trainIdx].pt for m in self.matches])
        
        ret_ckb, R_ckb_, t_ckb_ = self.ckb.estimate_pose(img)
        if ret_ckb:
            # Find the rotation and translation vectors.
            # When you chose Ransac, you need to add inliers for output.
            ret_PnP, rvecs, tvecs, inliner = cv2.solvePnPRansac(X_3d_matched,
                                                                X_2d_matched,
                                                                self.K,
                                                                self.dist,
                                                                #rvec=cv2.Rodrigues(R_ckb_)[0],
                                                                #tvec=t_ckb_,
                                                                useExtrinsicGuess=False,
                                                                flags=cv2.SOLVEPNP_ITERATIVE)
            if ret_PnP:
                R_ptc_ = cv2.Rodrigues(rvecs)[0]
                t_ptc_ = tvecs.copy()
                R_ptc = R_ptc_.T
                t_ptc = np.dot(-R_ptc_.T, t_ptc_)
                """
                dlt = P_estimator.ProjectionMatrixDLT()
                P_ptc = dlt.projection_matrix(X_3d_matched, X_2d_matched)
                #X_2d_dlt = project(X_3d, P_dlt)
                M_ptc = np.linalg.inv(self.K) @ P_ptc
                R_ptc = M_ptc[:, :3]
                t_ptc = M_ptc[:, 3].reshape(3, 1)
                """
                #assert(P_ptc.shape == (3, 4))
                #assert(M_ptc.shape == (3, 4))
                assert(R_ptc.shape == (3, 3))
                assert(t_ptc.shape == (3, 1))
                R_ckb = R_ckb_.T
                t_ckb = np.dot(-R_ckb_.T, t_ckb_)
        return ret_ckb, R_ckb, t_ckb, R_ptc, t_ptc

    def extract_match_featpoints(self, img):
        # undistortion
        img = cv2.undistort(img, self.K, self.dist)

        # A-KAZE検出器の生成
        detector = cv2.AKAZE_create(threshold=0.001)
        # detector = cv2.AgastFeatureDetector_create()
        # detector = cv2.ORB_create()
        # detector = cv2.xfeatures2d.SIFT_create()
        # detector = cv2.xfeatures2d.SURF_create()

        # 特徴量の検出と特徴量ベクトルの計算
        self.kp, self.desc = detector.detectAndCompute(img, None)

        # Brute-Force Matcher生成
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # 特徴量ベクトル同士をBrute-Force＆KNNでマッチング
        matches = bf.knnMatch(self.X_desc.astype('uint8'), self.desc, k=2)

        # データを間引きする
        ratio = 0.5
        good = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good.append([m])

        featpoints = matches #good.copy()
        return featpoints

class StereoscopicCameraPoseEstimator:

    def __init__(self, mtx, dist):
        self.K = mtx
        self.dist = dist
        self.ckb = camera.ExtrinsicParamsEstimator(self.K, self.dist,
                                                   n_crosspoint=(9, 6),
                                                   length_square_side=0.02308)

    def estimate_pose(self, imgs):
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
        # 外れ値を取り除きます
        pts1 = pts1[mask.ravel()==True]
        pts2 = pts2[mask.ravel()==True]
        desc1 = desc1[mask.ravel()==True]
        desc2 = desc2[mask.ravel()==True]

        x1 = pts1[:, 1].astype('int')
        y1 = pts1[:, 0].astype('int')
        x2 = pts2[:, 1].astype('int')
        y2 = pts2[:, 0].astype('int')
        band = 10
        self.featcolor1 = imgs[0][x1, y1, ::-1]
        self.featcolor2 = imgs[1][x2, y2, ::-1]

        #self.featcolor1_hex = ["#%x%x%x" % (r, g, b) for b, g, r in self.featcolor1]
        #self.featcolor2_hex = ["#%x%x%x" % (r, g, b) for b, g, r in self.featcolor2]

        """
        # find Fundamental Matrix
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

        # 外れ値を取り除きます
        pts1 = pts1[mask.ravel()==1]
        pts2 = pts2[mask.ravel()==1]
        """

        """
        #https://stackoverflow.com/questions/33906111/how-do-i-estimate-positions-of-two-cameras-in-opencv
        
        # Normalize for Esential Matrix calaculation
        self.pts1_norm = cv2.undistortPoints(pts1.reshape(-1, 1, 2),
                                        cameraMatrix=self.K,
                                        distCoeffs=self.dist).squeeze()
        self.pts2_norm = cv2.undistortPoints(pts2.reshape(-1, 1, 2),
                                        cameraMatrix=self.K,
                                        distCoeffs=self.dist).squeeze()

        print(self.pts1_norm.shape)
        #self.pts1_norm = pts1
        #self.pts2_norm = pts2
        self.E, mask = cv2.findEssentialMat(self.pts1_norm, self.pts2_norm,
                                       focal=1.0,
                                       pp=(0., 0.),
                                       method=cv2.RANSAC,
                                       prob=0.999, threshold=3.0)
        points, R, t, mask = cv2.recoverPose(self.E,
                                             self.pts1_norm, self.pts2_norm)


        pts1_ = np.c_[self.pts1_norm, np.ones(len(self.pts1_norm))]
        pts2_ = np.c_[self.pts2_norm, np.ones(len(self.pts2_norm))]
        z = 0.
        for pt1, pt2 in zip(pts1_, pts2_):
            z += np.linalg.det(np.r_[t.T, pt1[None, :], (R@pt2)[None, :]])
        if np.sign(z) < 0.:
            t *= -1

        M_l = np.hstack((np.eye(3, 3), np.zeros((3, 1))))
        M_r = np.hstack((R, t))

        P_l = np.dot(self.K,  M_l)
        P_r = np.dot(self.K,  M_r)
        """

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
            """
            X_2d = np.concatenate((pts1[None, :], pts2[None, :]), axis=0)
            X_P = np.concatenate((P1[None, :], P2[None, :]), axis=0)
            dlt = reconstructor.Reconstruction3dPoseDLT()
            self.X_3d = dlt.reconst(X_2d, X_P)
            """
            X_4d = cv2.triangulatePoints(self.P1, self.P2, pts1.T, pts2.T)
            X_3d = (X_4d[:3, :] / X_4d[-1, :]).T

            pts1_matched = pts1.copy()
            desc1_matched = desc1.copy()
        return ret, X_3d, pts1_matched, desc1_matched

    def extract_match_featpoints(self, imgs):
        # undistortion
        img1 = cv2.undistort(imgs[0], self.K, self.dist)
        img2 = cv2.undistort(imgs[1], self.K, self.dist)

        # A-KAZE検出器の生成
        detector = cv2.AKAZE_create(threshold=0.001)
        # detector = cv2.AgastFeatureDetector_create()
        # detector = cv2.ORB_create()
        # detector = cv2.xfeatures2d.SIFT_create()
        # detector = cv2.xfeatures2d.SURF_create()

        # 特徴量の検出と特徴量ベクトルの計算
        self.kp1, self.desc1 = detector.detectAndCompute(img1, None)
        self.kp2, self.desc2 = detector.detectAndCompute(img2, None)

        # Brute-Force Matcher生成
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        # 特徴量ベクトル同士をBrute-Force＆KNNでマッチング
        matches = bf.knnMatch(self.desc1, self.desc2, k=2)

        # データを間引きする
        ratio = 0.5
        good = []
        for m, n in matches:
            if m.distance < ratio * n.distance:
                good.append([m])

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

        featpoints = good.copy()
        return featpoints


if __name__ == "__main__":
    K = np.load('mtx.npy')
    dist = np.load('dist.npy')
    # K = np.loadtxt('mtx.csv', delimiter=',')
    # dist = np.loadtxt('dist.csv', delimiter=',')
    # make instance of Stereoscopic Pose Estimator.
    # estimator = StereoscopicCameraPoseEstimator(K, dist)

    """
    fnames_left = glob.glob('./samples/data/left0*.jpg')
    fnames_right = glob.glob('./samples/data/right0*.jpg')
    K = np.array([5.3591573396163199e+02, 0., 3.4228315473308373e+02, 0.,
                  5.3591573396163199e+02, 2.3557082909788173e+02,
                  0., 0., 1.]).reshape(3, 3)
    dist = np.array([-2.6637260909660682e-01, -3.8588898922304653e-02,
                     1.7831947042852964e-03, -2.8122100441115472e-04,
                     2.3839153080878486e-01])
    for fname_left, fname_right in zip(fnames_left, fnames_right):
        print(fname_left)
        print(fname_right)
        img_left = cv2.imread(fname_left, 0)
        img_right = cv2.imread(fname_right, 0)
        X_3d, R, t = estimator.estimate_pose([img_left, img_right])
        print(R)
        print(t)
    """

    # estimation from video file.
    estimator = StereoscopicCameraPoseEstimator(K, dist)
    RR = []
    tt = []
    R_track = np.eye(3)
    t_track = np.zeros([3, 1])
    X_3d = np.zeros([0, 3])
    X_pts = np.zeros([0, 2])
    X_desc = np.zeros([0, 61])
    colors = np.zeros([0, 3])
    X_3d_cam = np.zeros([0, 3])
    Viewvecs_cam = np.zeros([0, 3])


    cap = cv2.VideoCapture('output.avi')
    frames = []
    for i in range(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    frames = np.array(frames)
    cap.release()
    cv2.destroyAllWindows()

    for f_p, f_n in zip(frames[::10], np.roll(frames, 60)[::10]):
        ret, x_3d, pts1, desc1 = estimator.estimate_pose([f_p, f_n])
        if ret:
            X_3d = np.concatenate((X_3d, x_3d), axis=0)
            # store the 2D-keypoints and the descriptors.
            X_pts = np.concatenate((X_pts, pts1), axis=0)
            X_desc = np.concatenate((X_desc, desc1), axis=0)

            x_3d_cam = estimator.ckb.camera2world(x=np.zeros([3, 1]))
            X_3d_cam = np.concatenate((X_3d_cam, x_3d_cam.T), axis=0)
            viewvec_cam = estimator.ckb.camera2world(x=np.array([0, 0, 0.1]))
            Viewvecs_cam = np.concatenate((Viewvecs_cam, viewvec_cam.T), axis=0)
            colors = np.concatenate((colors, estimator.featcolor1), axis=0)

            x_2d = reconstructor.project(x_3d, estimator.P1)
            plt.figure()
            plt.imshow(f_p[:, :, ::-1])
            plt.scatter(x_2d[:, 0], x_2d[:, 1], c=estimator.featcolor1/255)
            plt.xlim(0, f_p.shape[1])
            plt.ylim(0, f_p.shape[0])
            plt.show()



    N = len(X_3d[0:])
    y = np.linspace(-1, 1, N)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X_3d[0:, 0],
               X_3d[0:, 1],
               X_3d[0:, 2],
               "o",
               #color="#00aa00",
               s=5,
               #ms=4,
               #mew=0.5,
               #c=np.linspace(0, 1, 200),
               #c=y,
               c=colors/255, #estimator.featcolor1/255,#a * N,
               #cmap=cm.seismic
               )

    ax.plot(X_3d_cam[0:, 0],
               X_3d_cam[0:, 1],
               X_3d_cam[0:, 2],
               "o-",
               color="#00aa00",
               markersize=10,
               linewidth=3,
               #s=200,
               #ms=4,
               #mew=0.5,
               #cmap=cm.seismic,
               #c=y,
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
                   #s=200,
                   #ms=4,
                   #mew=0.5,
                   #cmap=cm.seismic,
                   #c=y,
                )

    ax.scatter(estimator.ckb.objp[0:, 0],
               estimator.ckb.objp[0:, 1],
               estimator.ckb.objp[0:, 2],
               "O",
               color="#ff0000",
               s=100,
               #ms=4,
               #mew=0.5,
               #c=np.linspace(0, 1, 200),
               #c=y,
               #c=colors/255, #estimator.featcolor1/255,#a * N,
               #cmap=cm.seismic
               )
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    # ax.set_title("Scatter Plot")
    plt.tight_layout()
    plt.show()

    camest = CameraPoseEstimatorBasedOnPointCould(K, dist, X_3d, X_pts, X_desc)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret_cap, frame = cap.read()
        if ret_cap:
            ret_est, R_ckb, t_ckb, R_ptc, t_ptc = camest.estimate_pose(img=frame)
            if ret_est:
                #print(R_ckb)
                #print('ckb', t_ckb.T)
                #print(R_ptc)
                #print('ptc', t_ptc.T)
                print('distance_R: %0.4f, distance_t: %0.4f' % (
                        np.linalg.norm(R_ckb - R_ptc),
                        np.linalg.norm(t_ckb - t_ptc)))

    cap.release()
    cv2.destroyAllWindows()
    
