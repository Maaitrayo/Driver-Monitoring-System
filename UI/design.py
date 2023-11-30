from turtle import left
import cv2
import numpy as np

WHITE = (255,255,255)

def showText(img, text, font, fontScale, textPos, textThickness=1,textColor=(0,255,0), bgColor=(0,0,0), pad_x=3, pad_y=3):
   
    (t_w, t_h), _= cv2.getTextSize(text, font, fontScale, textThickness) # getting the text size
    x, y = textPos
    cv2.rectangle(img, (x-pad_x, y+ pad_y), (x+t_w+pad_x, y-t_h-pad_y), bgColor,-1) # draw rectangle 
    cv2.putText(img,text, textPos,font, fontScale, textColor,textThickness ) # draw in text

    return img

def imgResize(image, width, height):
	resized_img = cv2.resize(image, (height, width))
	return resized_img

def drawFaceMesh(window, mesh_coordinates):
    '''
    Taking the list of coordinates of each detected points and plotting them one by one using opencv
    to shift the entire mask of the face I am just adding some padding to its pixel coordinates
    '''
    for pos in mesh_coordinates:
        cv2.circle(window, (pos[0]+50, pos[1]-150), 1,WHITE, 1)

    return window


def drawEyeMesh(blank_win, left_eye_mesh, right_eye_mesh):
    left_mesh = []
    right_mesh = []
    y_pad_eye = 180
    for left_pos in left_eye_mesh:
        left_mesh.append([left_pos[0]-230, left_pos[1]+y_pad_eye])
    # print(left_mesh,"\n")
    left_mesh = np.array(left_mesh, np.int32)
    left_mesh = left_mesh.reshape((-1,1,2))
    cv2.polylines(blank_win,  [left_mesh], True, (0,255,0), 1, cv2.LINE_AA)
    
    for right_pos in right_eye_mesh:
        right_mesh.append([right_pos[0]+50, right_pos[1]+y_pad_eye])
    # print(right_mesh,"\n")
    right_mesh = np.array(right_mesh, np.int32)
    right_mesh = right_mesh.reshape((-1,1,2))
    cv2.polylines(blank_win,  [right_mesh], True, (0,255,0), 1, cv2.LINE_AA)

    return blank_win


def drawLipMesh(left_win, lip_mesh):
    lip_mesh_ui = []
    y_pad_lip = 220
    x_pad_lip = 50

    for lip_pos in lip_mesh:
        lip_mesh_ui.append([lip_pos[0]+x_pad_lip, lip_pos[1]+y_pad_lip]) 

    lip_mesh_ui = np.array(lip_mesh_ui, np.int32)
    lip_mesh_ui = lip_mesh_ui.reshape((-1,1,2))
    cv2.polylines(left_win,  [lip_mesh_ui], True, (0,255,0), 1, cv2.LINE_AA)
    return left_win


def warningSignals(left_win, STATUS):
    RED = (0,0,255)
    GREEN = (0,255,0)
    if STATUS['HEAD STATE'] == "DRIVER DISTRACTED":
        left_win = cv2.rectangle(left_win, (10, 10), (170,50),RED,-1)
        cv2.putText(left_win, 'DISTRACTED', (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    else:
        left_win = cv2.rectangle(left_win, (10, 10), (170,50),GREEN,-1)
        cv2.putText(left_win, 'ATTENTIVE', (50, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

    if STATUS["DROWSY STATE"] == "DRIVER DROWSY":
        left_win = cv2.rectangle(left_win, (10, 60), (170,100),RED,-1)
        cv2.putText(left_win, 'DROWSY', (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    else:    
        left_win = cv2.rectangle(left_win, (10, 60), (170,100),GREEN,-1)
        cv2.putText(left_win, 'NOT DROWSY', (50, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    
    if STATUS['YAWNING STATE'] == "DRIVER YAWNING":
        left_win = cv2.rectangle(left_win, (10, 110), (170,150),RED,-1)
        cv2.putText(left_win, 'YAWNING', (50, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    else:
        left_win = cv2.rectangle(left_win, (10, 110), (170,150),GREEN,-1)
        cv2.putText(left_win, 'NOT YAWNING', (50, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

    if STATUS["TALKING"] == "DRIVER TALKING":
        left_win = cv2.rectangle(left_win, (10, 160), (170,200),RED,-1)
        cv2.putText(left_win, 'TALKING', (50, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)
    else:
        left_win = cv2.rectangle(left_win, (10, 160), (170,200),GREEN,-1)
        cv2.putText(left_win, 'NOT TALKING', (50, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

    return left_win

def displayHeadStats(left_win,head_pos, head_orientation):
    cv2.putText(left_win, head_pos, (50, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)


def drawBoundaries(left_win):
    CENTER_LINE = 250 # Y coordinate where the center line should be placed
    left_win = cv2.line(left_win,(0, CENTER_LINE),(left_win.shape[0],CENTER_LINE), WHITE,2) # first line horizontal
    left_win = cv2.line(left_win,(180, 0),(180,CENTER_LINE), WHITE,2) # first verticle line
    left_win = cv2.line(left_win,(0, CENTER_LINE+50),(500,CENTER_LINE+50), WHITE,2) # second horiziontal line, just below first horizontal line
    left_win = cv2.line(left_win,(0, CENTER_LINE+100),(500,CENTER_LINE+100), WHITE,2) # third horiziontal line, just below second horizontal lineD
    left_win = cv2.line(left_win,(0, CENTER_LINE+170),(left_win.shape[0]//2,CENTER_LINE+170), WHITE,2) # fourth horiziontal line, part 1
    left_win = cv2.line(left_win,(left_win.shape[0]//2, CENTER_LINE+170),(left_win.shape[0],CENTER_LINE+170), WHITE,2) # fourth horiziontal line, part 2
    left_win = cv2.line(left_win,(left_win.shape[0]//2, CENTER_LINE+100),(left_win.shape[0]//2,left_win.shape[1]), WHITE,2) # second verticle line
    left_win = cv2.line(left_win,(0, CENTER_LINE+200),(500,CENTER_LINE+200), WHITE,2) # fifth horiziontal line, just below second fourth line
    # print(left_win.shape)
    return left_win

def createCustomUI(blank_win, image, mesh_coord, left_eye_mesh, right_eye_mesh, lip_mesh, STATUS, head_pos, head_orientation):
    blank_win = imgResize(blank_win, 500, 500)
    image = imgResize(image, 500, 500)
    
    left_win = drawFaceMesh(blank_win, mesh_coord)
    left_win = drawBoundaries(left_win)
    left_win = drawEyeMesh(left_win, left_eye_mesh, right_eye_mesh)
    left_win = drawLipMesh(left_win, lip_mesh)
    left_win = warningSignals(left_win, STATUS)
    # left_win = displayHeadStats(left_win, head_pos, head_orientation )
    
    print(STATUS)

    final_image = cv2.hconcat([left_win, image])
    return final_image
