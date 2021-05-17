import sys

import numpy as np
import cv2

def getPos(box) :
    x = round(box[0] / 24)
    y = round(box[1] / 50)
    return 25 * y + x

im = cv2.imread('train.png')
im3 = im.copy()

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

#################      Now finding Contours         ###################

contours,hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 50 and cv2.contourArea(c) < 2000]
print(boxes)
boxes.sort(key=getPos)

samples =  np.empty((0,100))
responses = []
keys = [i for i in range(48,58)]

for box in boxes:
    [x,y,w,h] = box
    if h > 28 :
        if  h>28:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            cv2.imshow('norm',im)
            key = cv2.waitKey(0)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                keychr = int(chr(key))
                responses.append(keychr)
                print(keychr)
                pos = getPos([x, y])
                print([x, y])
                print("pos", pos)
                sample = roismall.reshape((1,100))
                samples = np.append(samples,sample,0)

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print ("training complete")

np.savetxt('pysamples.data',samples)
np.savetxt('pyresponses.data',responses)