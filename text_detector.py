from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import numpy as np 
import argparse
import imutils
import time
import cv2
def call():
	image=cv2.imread("test_4.jpg")
	gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	cv2.imshow("grey",gray)
	_,thresh=cv2.threshold(gray,140,255,cv2.THRESH_BINARY_INV)
	kernel=cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	dilated=cv2.dilate(thresh,kernel,iterations=1)
	cv2.imshow("in",dilated)
	#contours, hierarchy=cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
	cnts = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	contours = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
	#screenCnt = None
	for contour in contours:
		[x,y,w,h]=cv2.boundingRect(contour)
		if h>400 or w>400:
			continue
		if h<30 or w<30:
			continue
		cv2.rectangle(image,(x,y),(x+w,y+h),(250,0,10),2)
	cv2.imshow("out",image)
	cv2.waitKey(0)
call()