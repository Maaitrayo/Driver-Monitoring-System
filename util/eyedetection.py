import cv2
import numpy as np
import math

from UI.design import showText

from helper.helper import euclaideanDistance
# from util.headPoseEst import COUNTER


# variables 
CEF_COUNTER =0
TOTAL_BLINKS =0
# constants
CLOSED_EYES_FRAME =3
DROWSY_FRAMES = 15
EAR_THRESHOLD = 3.6

# individual eye counters
COUNT_LE = 0 # LE --> LEFT EYE
COUNT_RE = 0 # LE --> RIGHTT EYE
CLOSE_THRESHOLD = 3 # --> THRESHOLD 
check = {'LEFT': 'OPEN', 'RIGHT': 'OPEN'}


def getLeftEAR(image, mesh_coord, LEFT_EYE):
	if len(LEFT_EYE) == 0 or len(mesh_coord) == 0:
		return 0, image
	# LEFT_EYE 
	# horizontal line 
	else:
		if mesh_coord[LEFT_EYE[0]] and mesh_coord[LEFT_EYE[8]] and mesh_coord[LEFT_EYE[12]] and mesh_coord[LEFT_EYE[4]]:
			lh_right = mesh_coord[LEFT_EYE[0]]
			lh_left = mesh_coord[LEFT_EYE[8]]

			# vertical line 
			lv_top = mesh_coord[LEFT_EYE[12]]
			lv_bottom = mesh_coord[LEFT_EYE[4]]

			lvDistance = euclaideanDistance(lv_top, lv_bottom)
			lhDistance = euclaideanDistance(lh_right, lh_left)

			leRatio = lhDistance/lvDistance
      
			# cv2.polylines(image,  [np.array([mesh_coord[p] for p in LEFT_EYE ], dtype=np.int32)], True, (0,255,0), 1, cv2.LINE_AA)
 
	
	
	return leRatio, image

def getRightEAR(image, mesh_coord, RIGHT_EYE):
	if len(RIGHT_EYE) == 0 or len(mesh_coord) == 0:
			return 0, image
	else:
		# Right eyes 
		# horizontal line 
		rh_right = mesh_coord[RIGHT_EYE[0]]
		rh_left = mesh_coord[RIGHT_EYE[8]]
		# vertical line 
		rv_top = mesh_coord[RIGHT_EYE[12]]
		rv_bottom = mesh_coord[RIGHT_EYE[4]]

		rhDistance = euclaideanDistance(rh_right, rh_left)
		rvDistance = euclaideanDistance(rv_top, rv_bottom)

		reRatio = rhDistance/rvDistance
		# cv2.polylines(image,  [np.array([mesh_coord[i] for i in RIGHT_EYE ], dtype=np.int32)], True, (0,255,0), 1, cv2.LINE_AA)
		return reRatio, image 

def blinkCounter(EAR, image, eye):
    global CEF_COUNTER, TOTAL_BLINKS, CLOSED_EYES_FRAME, EAR_THRESHOLD
    # print(EAR)

    if EAR >EAR_THRESHOLD:
        CEF_COUNTER +=1
    
    else:
        if CEF_COUNTER>CLOSED_EYES_FRAME and eye=="BOTH":
            TOTAL_BLINKS +=1
            CEF_COUNTER =0
    
    # print("TOTAL BLINKS : ", TOTAL_BLINKS)
    showText(image,  f'Total Blinks: {TOTAL_BLINKS}', cv2.FONT_HERSHEY_COMPLEX,  0.7, (30,150),2)

def eyeClosing(left_EAR, right_EAR, image, STATUS):
  global COUNT_LE, COUNT_RE, check, EAR_THRESHOLD
  STATUS['RIGHT EYE'] = "RIGHT EYE OPEN"
  STATUS['LEFT EYE'] = "LEFT EYE OPEN"
  if left_EAR >EAR_THRESHOLD:
    COUNT_LE +=1
    if COUNT_LE > 3:
      # AS the image is flipped left eye on image is actually right eye
      STATUS['RIGHT EYE'] = "RIGHT EYE CLOSING"
      showText(image,  f'RIGHT EYE CLOSING', cv2.FONT_HERSHEY_COMPLEX, 1, (int(image.shape[0]/2), 100), 2, (0,255,255), pad_x=6, pad_y=6)
      check['RIGHT'] = 'CLOSE'
      COUNT_LE=0
    
  elif right_EAR >EAR_THRESHOLD:
    COUNT_RE +=1
    if COUNT_RE > 3:
      # AS the image is flipped right eye on image is actually left eye
      STATUS['LEFT EYE'] = "LEFT EYE CLOSING"
      showText(image,  f'LEFT EYE CLOSING', cv2.FONT_HERSHEY_COMPLEX, 1, (int(image.shape[0]/2), 100), 2, (0,255,255), pad_x=6, pad_y=6)
      check['LEFT'] = 'CLOSE'
      COUNT_RE = 0

  return STATUS, check 

def drowsynessDetection(EAR, image, STATUS):
  global CEF_COUNTER, DROWSY_FRAMES, EAR_THRESHOLD
  # print(EAR)
  STATUS["DROWSY STATE"] = "DRIVER NOT DROWSY"

  if EAR >EAR_THRESHOLD:
      CEF_COUNTER +=1
      if CEF_COUNTER>DROWSY_FRAMES :
        STATUS["DROWSY STATE"] = "DRIVER DROWSY" 
        showText(image,  f'DRIVER DROWSY', cv2.FONT_HERSHEY_COMPLEX,  0.7, (30,350),2)
        CEF_COUNTER =0
  
  
  return STATUS
  



def eyeWarningSystem(mesh_coord,LEFT_EYE, RIGHT_EYE, image, STATUS, position):
  if position == 'Forward':
    # Getting the left EYE ASPECT RATIO(EAR)
    left_EAR, image = getLeftEAR(image, mesh_coord, LEFT_EYE)

    # Getting the right EYE ASPECT RATIO(EAR)
    right_EAR, image = getRightEAR(image, mesh_coord, RIGHT_EYE)

    STATUS, check = eyeClosing(left_EAR, right_EAR, image, STATUS)
    EAR = (left_EAR + right_EAR)/2
    STATUS = drowsynessDetection(EAR, image, STATUS)

    # left_eye_mesh = np.array([mesh_coord[p] for p in LEFT_EYE ], dtype=np.int32)
    # right_eye_mesh = np.array([mesh_coord[i] for i in RIGHT_EYE], dtype=np.int32)

    if STATUS["DROWSY STATE"] == "DRIVER NOT DROWSY":
      # whenever the driver was detected as drowsy the blim=nk counter used to rapidly incerease, thus this ckeck is applied
      blinkCounter(EAR, image, eye="BOTH")
  return STATUS




