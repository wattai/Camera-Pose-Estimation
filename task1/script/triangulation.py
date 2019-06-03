# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 00:51:21 2019

@author: wattai
"""

import numpy as np
np.random.seed(7)


def project(X_3d, P):
    N = X_3d.shape[0]
    X_2d = np.dot(np.hstack((X_3d, np.ones([N, 1]))), P.T)
    return X_2d[:, :2] / X_2d[:, 2][:, None]


class Reconstruction3dPoseDLT:
    def __init__(self):
        self.N, self.M = None, None
        self.A = None
        self.U, self.S, self.V = None, None, None

    def a(self, x_2d, P):
        return np.array([[P[0, 0]-P[2, 0]*x_2d[0],
                          P[0, 1]-P[2, 1]*x_2d[0],
                          P[0, 2]-P[2, 2]*x_2d[0],
                          P[0, 3]-P[2, 3]*x_2d[0]],
                         [P[1, 0]-P[2, 0]*x_2d[1],
                          P[1, 1]-P[2, 1]*x_2d[1],
                          P[1, 2]-P[2, 2]*x_2d[1],
                          P[1, 3]-P[2, 3]*x_2d[1]]])

    def reconst(self, X_2d, X_P):
        self.N_cam = X_2d.shape[0]
        self.N_pair = X_2d.shape[1]
        self.M = 4
        self.X_3d_reconst = np.zeros([self.N_pair, self.M])

        for i in range(0, self.N_pair):
            self.A = np.zeros([0, self.M])
            for j in range(0, self.N_cam):
                self.A = np.concatenate((self.A, self.a(X_2d[j][i], X_P[j])),
                                        axis=0)
            self.U, self.S, self.V = np.linalg.svd(self.A, full_matrices=False)
            self.X_3d_reconst[i, :] = self.V[-1]
            normalizer = self.X_3d_reconst[:, 3].reshape(self.N_pair, 1)
        return self.X_3d_reconst[:, :3] / normalizer


if __name__ is '__main__':

    P1 = np.hstack((np.eye(3), np.ones([3, 1])))
    P2 = np.array([[-1.16947620e+02, -5.58083756e+02,  5.18178288e+02,
                    2.09444515e+04],
                   [4.73859362e+02,  2.27654161e+02,  4.56142144e+02,
                    4.98118994e+03],
                   [-5.32882591e-01,  2.99798655e-01,  7.91300771e-01,
                    5.92931028e+01]])

    N = 6
    X_3d = 10*np.random.rand(N, 3)
    x_2d_P1 = project(X_3d, P1)
    x_2d_P2 = project(X_3d, P2)
    X_2d = np.concatenate((x_2d_P1[None, :], x_2d_P2[None, :]), axis=0)
    X_P = np.concatenate((P1[None, :], P2[None, :]), axis=0)

    dlt = Reconstruction3dPoseDLT()
    X_3d_dlt = dlt.reconst(X_2d, X_P)

    print('reconstruction error: %.5f' % np.linalg.norm(X_3d - X_3d_dlt))
