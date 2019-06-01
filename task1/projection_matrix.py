# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 21:49:13 2019

@author: wattai
"""


import numpy as np
np.random.seed(7)


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def project(X_3d, P):
    N = X_3d.shape[0]
    X_2d = np.dot(np.hstack((X_3d, np.ones([N, 1]))), P.T)
    return X_2d[:, :2] / X_2d[:, 2][:, None]


class ProjectionMatrixDLT:
    def __init__(self):
        self.N, self.M = None, None
        self.A = None
        self.U, self.S, self.V = None, None, None

    def A_x(self, x_3d, x_2d):
        return np.array([x_3d[0], x_3d[1], x_3d[2], 1.,
                         0., 0., 0., 0.,
                         -x_3d[0]*x_2d[0], -x_3d[1]*x_2d[0],
                         -x_3d[2]*x_2d[0], -x_2d[0]])

    def A_y(self, x_3d, x_2d):
        return np.array([0., 0., 0., 0.,
                         x_3d[0], x_3d[1], x_3d[2], 1.,
                         -x_3d[0]*x_2d[1], -x_3d[1]*x_2d[1],
                         -x_3d[2]*x_2d[1], -x_2d[1]])

    def projection_matrix(self, X_3d, X_2d):
        if X_3d.shape[0] != X_2d.shape[0]:
            raise(ValueError('The shapes of each inputs are different.'))
        elif X_3d.shape[0] < 6:
            raise(ValueError('The number of data-points is not enough.'))

        self.N = X_3d.shape[0]
        self.M = 12

        self.A = np.zeros([2*self.N, self.M])
        for i in range(0, self.N):
            self.A[2*i, :] = self.A_x(X_3d[i], X_2d[i])
            self.A[2*i+1, :] = self.A_y(X_3d[i], X_2d[i])

        self.U, self.S, self.V = np.linalg.svd(self.A, full_matrices=False)
        return self.V[-1].reshape(3, 4)


if __name__ is '__main__':

    P = np.array([[-1.16947620e+02, -5.58083756e+02,  5.18178288e+02,
                   2.09444515e+04],
                  [4.73859362e+02,  2.27654161e+02,  4.56142144e+02,
                   4.98118994e+03],
                  [-5.32882591e-01,  2.99798655e-01,  7.91300771e-01,
                   5.92931028e+01]])

    N = 6
    X_3d = 10*np.random.rand(N, 3)
    X_2d = project(X_3d, P)

    dlt = ProjectionMatrixDLT()
    P_dlt = dlt.projection_matrix(X_3d, X_2d)
    X_2d_dlt = project(X_3d, P_dlt)

    print('reprojection error: %.5f' % np.linalg.norm(X_2d - X_2d_dlt))
    print('cos similality:  %.5f' % cos_sim(P.reshape(-1), P_dlt.reshape(-1)))
