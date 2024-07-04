from ultralytics import YOLO
import cv2
import os
from sort.sort import *
from utils import *
import matplotlib.pyplot as plt
import numpy as np




sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel("models/ESPCN_x2.pb")  # Ensure you have the ESPCN model file
sr.setModel("espcn", 2)  # Set the model and scale





results={}
mot_tracker = Sort()

coco_model = YOLO('yolov8n.pt')

license_plate_detector = YOLO('license_plate_model_5.pt')


input_video_path = 'demo6.mp4'
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)


cap = cv2.VideoCapture(input_video_path)

if not cap.isOpened():
    print(f"Error: Could not open video '{input_video_path}'")
    exit()
    
frame_nmr=-1
vehicals = [2,3,5,7]
ret = True
while ret:
    frame_nmr+=1
    ret,frame = cap.read()
    
    if ret and frame_nmr<10:
        results[frame_nmr] = {}
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicals:
                detections_.append([x1, y1, x2, y2, score])
                
        trac_ids = mot_tracker.update(np.asarray(detections_))
        
        
        license_plates = license_plate_detector(frame)[0]
        license_ = []
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, trac_ids)
            
            if car_id != -1:
                license_plate_crop = frame[int(y1):int(y2),int(x1):int(x2)]
                
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop,cv2.COLOR_BGR2GRAY)
                
                upscaled = sr.upsample(license_plate_crop_gray)
                
                blurred = cv2.GaussianBlur(upscaled, (5, 5), 0)
                
                denoised = cv2.fastNlMeansDenoising(upscaled, None, 3, 5, 15)
                
                sharpening_kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]]) 
                sharpened = cv2.filter2D(denoised, -1, kernel=sharpening_kernel)
                
                _, license_plate_crop_thresh = cv2.threshold(sharpened,78,255,cv2.THRESH_BINARY_INV)
                
                # cv2.imshow('license denoised',denoised)
                # cv2.imshow('license upscales',upscaled)
                # cv2.imshow('license sharpened',sharpened)
                # cv2.imshow('license threshold',license_plate_crop_thresh)
                # cv2.waitKey(0)
                
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
                
                if license_plate_text is not None:
                    results[frame_nmr][int(car_id)] = {'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                                                'license_plate': {'bbox': [x1, y1, x2, y2],
                                                                    'text': license_plate_text,
                                                                    'bbox_score': score,
                                                                    'text_score': license_plate_text_score}}
            
        
        


write_csv(results,'./test.csv')