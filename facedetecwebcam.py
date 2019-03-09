from imageai.Detection import ObjectDetection
import os
import cv2
import matplotlib
import numpy as np
import sys

face_cascade = cv2.CascadeClassifier('D:/Python/Python36/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
#------------------------------------------------
execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel(detection_speed = 'fastest')
#---------------------
video_capture = cv2.VideoCapture(0)
#------------------------------------------------

while True:

	ret, frame = video_capture.read()

	video_path = detector.detectObjectsFromVideo(input_file_path=os.path.join(execution_path, "traffic.mp4"),
                                output_file_path=os.path.join(execution_path, "traffic_detected")
                                , frames_per_second=20, log_progress=True)
	#print(detections)
	#returned_image=cv2.cvtColor(returned_image, cv2.COLOR_RGB2BGR)
	#cv2.imshow('result',returned_image)
	#cv2.imshow('crop',extracted_objects)
	#cv2.imwrite('result.jpg',returned_image)
	a = 0
	for i in detections:
		#txt = 'Image Crop'+str(a)
		#txt1 = 'Face Detec'+str(a)
		#txt2 = 'Face'+str(a)
		#wr = txt + ".jpg"
		if i['name'] == 'person':
			#print(i['name'])
			#cv2.imshow(txt,extracted_objects[a])
			img1=cv2.cvtColor(extracted_objects[a],cv2.COLOR_RGB2BGR)
			#cv2.imshow('hinh 1',img1)
			#cv2.imwrite(wr,img1)
			gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
			faces = face_cascade.detectMultiScale(gray, 1.3, 5)
			for (x,y,w,h) in faces:
    				cv2.rectangle(img1,(x,y),(x+w,y+h),(0,0,255),1)
    				img2 = img1[y:y+h,x:x+w]
			#cv2.imshow(txt1,img1)
			#cv2.imshow(txt2,img2)
			#cv2.imwrite(wr,img2)
		a=a+1

	cv2.imshow('Video', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()