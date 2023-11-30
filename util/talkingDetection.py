import cv2 
import numpy as np

from helper.helper import euclaideanDistance

YAWN_COUNTER = 0
YAWN_DIST_THRESHOLD = 16
YAWN_FRAME_THRESHOLD = 10

TALKING_THRESHOLD = 10
TALKING_FRAME_THRESHOLD = 10
TALKING_COUNTER = 0
MOUTH_BLINKS = 0
THRESHOLD_BLINKS_MOUTH = 10
mouth_loc = [183,42,81,82, 13, 312, 311,310,272,407,324,318,402,317,14,87,178,88,95]
def checkYawning(lip_distance, STATUS):
    global YAWN_COUNTER, YAWN_FRAME_THRESHOLD, YAWN_DIST_THRESHOLD
    if lip_distance > YAWN_DIST_THRESHOLD:
        YAWN_COUNTER+=1
        if YAWN_COUNTER > YAWN_FRAME_THRESHOLD:
            STATUS['YAWNING STATE'] = "DRIVER YAWNING"
            # print("[!] YAWNING")
            YAWN_COUNTER=0
    else:
        STATUS['YAWNING STATE'] = "NOT YAWNING"
    
    return STATUS

def checkTalking(lip_distance, STATUS):
    global TALKING_THRESHOLD, TALKING_COUNTER, MOUTH_BLINKS, THRESHOLD_BLINKS_MOUTH
    STATUS["TALKING"] = "NOT TALKING"
    if lip_distance > TALKING_THRESHOLD:
        TALKING_COUNTER+=1
        
    else:
        if TALKING_COUNTER > TALKING_FRAME_THRESHOLD:
            MOUTH_BLINKS+=1
            TALKING_COUNTER=0
            if MOUTH_BLINKS > THRESHOLD_BLINKS_MOUTH:
                STATUS["TALKING"] = "DRIVER TALKING"
                # print("[!] DRIVER TALKING ")
                MOUTH_BLINKS = 0

    return STATUS


def obtainMouthStatus(image,mesh_coord, STATUS):
    global YAWN_COUNTER, TALKING_THRESHOLD, mouth_loc
    if len(mesh_coord) == 0:
        return 0, image
        
    point_up = mesh_coord[13]
    point_down = mesh_coord[14]

    lip_mesh = np.array([mesh_coord[p] for p in mouth_loc ], dtype=np.int32)

    lip_distance = euclaideanDistance(point_up, point_down)
    # cv2.polylines(image,  [np.array([point_up, point_down], dtype=np.int32)], True, (0,255,0), 1, cv2.LINE_AA)
    # print(lip_mesh)
    STATUS = checkYawning(lip_distance, STATUS)
    STATUS = checkTalking(lip_distance, STATUS)
    return STATUS, lip_mesh
   