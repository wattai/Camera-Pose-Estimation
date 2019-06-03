# -*- coding: utf-8 -*-
"""
Created on Wed May 29 20:59:58 2019

@author: wattai
"""

import numpy as np
import cv2

WIDTH = 640
HEIGHT = 480
FPS = 30

cap = cv2.VideoCapture(0)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)
cap.set(cv2.CAP_PROP_FPS, FPS)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('../video/output.avi', fourcc, FPS, (WIDTH, HEIGHT))

while cap.isOpened():
    ret, frame = cap.read()
    if ret is False:
        continue

    # write the flipped frame
    out.write(frame)

    cv2.imshow("Image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
