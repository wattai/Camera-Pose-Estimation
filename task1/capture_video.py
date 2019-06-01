# -*- coding: utf-8 -*-
"""
Created on Wed May 29 20:59:58 2019

@author: imd
"""

import numpy as np
import cv2

width = 640
height = 480
fps = 30

cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)
cap.set(cv2.CAP_PROP_FPS, fps)

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('wattai.avi', fourcc, fps, (width, height))

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
