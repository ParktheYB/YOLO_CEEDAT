# -*- coding: utf-8 -*-
"""
Created on Sun Jan 14 00:53:39 2024

@author: 82103
"""

#https://docs.ultralytics.com/tasks/pose/

from ultralytics import YOLO
import cv2
import torch
import matplotlib.pyplot as plt

# 모델 로드
model = YOLO('yolov8n-pose.pt')  # 사전 훈련된 YOLOv8n 모델

# 이미지 리스트에 대한 배치 추론 실행
source = cv2.imread('Rune.jpg')
results = model(source)  # Results 객체의 생성자 반환

# 결과 생성자 처리
for result in results:
    boxes = result.boxes  # bbox 출력을 위한 Boxes 객체
    masks = result.masks  # 세그멘테이션 마스크 출력을 위한 Masks 객체
    keypoints = result.keypoints  # 자세 출력을 위한 Keypoints 객체
    probs = result.probs  # 분류 출력을 위한 Probs 객체
    
for box in boxes.xyxy.tolist():
    k=0
    xmin, ymin, xmax, ymax = round(box[0]),round(box[1]),round(box[2]),round(box[3])
    cv2.rectangle(source, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
    cv2.putText(source, f"{k}", (xmin + 5, ymin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)   
    k=+1
for keypoint in keypoints.xy.tolist():
    break
        
cv2.imshow('1', source)
plt.imshow(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
plt.show()
#%%
from ultralytics import YOLO
import cv2
import torch
import matplotlib.pyplot as plt

# 모델 로드
model = YOLO('yolov8n-pose.pt')  # 사전 훈련된 YOLOv8n 모델

# 이미지 리스트에 대한 배치 추론 실행
source = cv2.imread('Rune.jpg')
results = model(source)  # Results 객체의 생성자 반환
annotated_frame = results[0].plot()

# Display the annotated frame
cv2.imshow("YOLOv8 Inference", annotated_frame)
plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
plt.show()
#%%
import cv2
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8n-pose.pt')

# 동영상 파일 사용시
# video_path = "path/to/your/video/file.mp4"
# cap = cv2.VideoCapture(video_path)

# webcam 사용시
cap = cv2.VideoCapture(0)

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Display the annotated frame
        cv2.imshow("YOLOv8 Inference", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()