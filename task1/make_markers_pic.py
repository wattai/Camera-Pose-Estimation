# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 11:55:21 2019

@author: wattai
"""

import numpy as np
import cv2
# from cv2 import aruco
aruco = cv2.aruco


markers_x = 3
markers_y = 5
marker_length = 3
marker_separation = 1


# DICT_4X4_50は4ｘ4の格子でマーカ作成、ID50個
# drawMarker(dictionary, marker ID, marker_size[pix])
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
grid_board = aruco.GridBoard_create(markers_x, markers_y, marker_length,
                                    marker_separation, dictionary)

marker = aruco.drawMarker(dictionary, 4, 100)

cv2.imwrite('ar_marker.png', marker)

margins = 40
img_width  = markers_x * (marker_length + marker_separation
                          ) - marker_separation + 2 * margins
img_height = markers_y * (marker_length + marker_separation
                          ) - marker_separation + 2 * margins

grid_board.draw(img)