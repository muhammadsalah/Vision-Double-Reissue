## Jul,17th, 20017.
## Author: M.Salah
## Team: Vision-Double (IBM Egypt - CIC Summer Internship)
## Title: Enchanced Face Detection Python Module
## Description:
## 	The Module aims to provide a multi threaded Face Detection.
##	The Module needs to be extended to pass remodified parameters
## ------------------------------------------------------------------------------------------------------
import cv2
import thread


class FaceDetector ():
	def __init__(self,scaleFactor,minNeighbors,minSize,CascPath):
		self.scaleFactor = scaleFactor
		self.minNeighbors= minNeighbors
		self.minSize=minSize
		self.FaceCascade =cv2.CascadeClassifier(CascPath)
	
	def DetectFace(self,Image,imgno):
		#Prepare the Image
		GrayImage=cv2.cvtColor(Image,cv2.COLOR_BGR2GRAY)
		faces=self.FaceCascade.detectMultiScale(
		GrayImage,
		scaleFactor= self.scaleFactor,
		minNeighbors=self.minNeighbors,
		minSize=self.minSize,
		flags = cv2.cv.CV_HAAR_SCALE_IMAGE
		)
		print('Found faces: '+str(len(faces))+' in frame '+str(imgno))
		for (x, y, w, h) in faces:
    			cv2.rectangle(Image, (x, y), (x+w, y+h), (0, 0, 255), 2)
		if len(faces) > 0:
			cv2.imwrite('./app/data/snapshots/'+'Faces'+str(len(faces))+'frame'+str(imgno)+'.png',Image)
			print('Saved the captured frame: Faces'+str(len(faces))+'frame'+str(imgno)+'.png')

