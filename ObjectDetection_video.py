from imageai.Detection import VideoObjectDetection
import os
import shutil
import cv2

#hsv_img= cv2.cvtColor(img,cv2.COLOR_BGR2GRB)
execution_path = os.getcwd()

camera = cv2.VideoCapture(0)

detector = VideoObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
detector.loadModel(detection_speed = 'fastest')

video_path = detector.detectObjectsFromVideo(camera_input=camera,
    output_file_path=os.path.join(execution_path, "camera_detected_video")
    , frames_per_second=20, log_progress=True, minimum_percentage_probability=30)

print(video_path)
#detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "source_1.png"), output_image_path=os.path.join(execution_path , "imagenew.png"))
#print('bla bla')
#print('detections:',detections)
#print('detections 0:',detections[0])
#print('type of detections:',detections.type())
#print('size of detection:',detections.shape())
#print('result:',eachObject)
#cv2.waitKey(0)
#cv2.destroyAllWindows() 