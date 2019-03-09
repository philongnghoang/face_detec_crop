from imageai.Detection import ObjectDetection
import os
import shutil
import cv2
import time
execution_path = os.getcwd()
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel(detection_speed="faster")
face_cascade = cv2.CascadeClassifier('D:/Python/Python36/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

capture = cv2.VideoCapture(0)
while capture.isOpened():
	ret, frame = capture.read()
	img_crop=frame[50:550,100:600]
	img=cv2.cvtColor(img_crop,cv2.COLOR_BGR2HSV)
	img=cv2.cvtColor(img,cv2.COLOR_HSV2RGB)
	cv2.rectangle(frame,(50,100),(550,600),(0,255,0),1)
	return_img,detections,extracted_ob = detector.detectObjectsFromImage(input_type='array',input_image=img,output_type="array",minimum_percentage_probability=30,extract_detected_objects=True) #extract_detected_objects=True
	#detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "source_1.png"), output_image_path=os.path.join(execution_path , "imagenew.png"))
	print('result:')
	a=0
	for eachObject in detections:
		print('eachObject:',eachObject)
		txt = 'Face Detec'+str(a)
		wr =  'Image Crop'+ str(a) + '.jpg'
		#cv2.imwrite(wr,extracted_ob[a])
		if eachObject['name']=='person':
			myimg_person= extracted_ob[a]
			#print("a:",a)
			#time.sleep(0.5)
			#print('type',type(myimg_person))
			#print('size:',myimg_person.shape)
			if myimg_person.shape[1] != 0 and myimg_person.shape[0] != 0 :
				gray = cv2.cvtColor(myimg_person, cv2.COLOR_BGR2GRAY)
				faces = face_cascade.detectMultiScale(gray, 1.3, 5)
				for (x,y,w,h) in faces:
					cv2.rectangle(img_crop,(x,y),(x+w,y+h),(0,0,255),1)
					img2 = myimg_person[y:y+h,x:x+w]
					img2=cv2.cvtColor(img2,cv2.COLOR_RGB2BGR)
					#cv2.imshow(txt,img2)
					cv2.imwrite(wr,img2)
		a=a+1
	cv2.imshow('input:',frame)
	#return_img=cv2.cvtColor(return_img,cv2.COLOR_RGB2BGR)
	#cv2.imshow('result:',return_img)
	time.sleep(0.1)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
capture.release()
cv2. destroyAllWindows()


