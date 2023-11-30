import cv2
import mediapipe as mp
import time
import numpy as np
import logging
from util.eyedetection import eyeWarningSystem
from util.headPoseEst import headPoseEstimation, checkDistractedDriving
from util.talkingDetection import obtainMouthStatus
from playsound import playsound
from helper.helper import preProcessImage, getPixelCoordMesh
import os
from UI.design import createCustomUI

# Set up logging
logging.basicConfig(filename='driver_monitor.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


faceDetector = mp.solutions.face_detection
drawing = mp.solutions.drawing_utils

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)


# cap = cv2.VideoCapture("videos/video_2.hevc")
cap = cv2.VideoCapture(0)

# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]
LEFT_EYELIDS = [414,286,258,257,259,260,467,359,249,390,373,374,380,381,382,362]

# right eyes indices  
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

head_pos = ""
STATUS = {'HEAD STATE': None, 'RIGHT EYE': None, 'LEFT EYE': None, 'DROWSY STATE': None, 'YAWNING STATE': None, 'TALKING': None}


face_detection = faceDetector.FaceDetection(min_detection_confidence=0.7)

left_eye_mesh = []
right_eye_mesh = []
lip_mesh = []


def detect_face(img):
	global STATUS, left_eye_mesh, right_eye_mesh, lip_mesh
	image = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

	image.flags.writeable = False

	results = face_mesh.process(image)

	image.flags.writeable = True

	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	# if results:
	image, face_mesh_win, position, head_orientation = headPoseEstimation(image, results, head_pos)
	STATUS = checkDistractedDriving(position, STATUS)
	mesh_coord = getPixelCoordMesh(image, results,LEFT_EYE, RIGHT_EYE)
	if(len(mesh_coord) == 468):
		STATUS= eyeWarningSystem(mesh_coord,LEFT_EYE, RIGHT_EYE, image, STATUS, position)
		STATUS, lip_mesh = obtainMouthStatus(image, mesh_coord, STATUS)

		# Check if the DROWSY state is detected in STATUS
		if STATUS['DROWSY STATE'] == 'DRIVER DROWSY':
			playsound('alarm.mp3')


		left_eye_mesh = np.array([mesh_coord[p] for p in LEFT_EYE ], dtype=np.int32)
		right_eye_mesh = np.array([mesh_coord[i] for i in RIGHT_EYE], dtype=np.int32)
		logging.info(f"STATUS: {STATUS}")
	else:
		print("[!] CANNOT DETECT FACE")
		logging.error("[!] CANNOT DETECT FACE")

		
	final_image = createCustomUI(face_mesh_win, image, mesh_coord, left_eye_mesh, right_eye_mesh, lip_mesh, STATUS, head_pos, head_orientation)

	return final_image


