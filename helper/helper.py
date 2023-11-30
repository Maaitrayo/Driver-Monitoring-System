import cv2
import math

def preProcessImage(image):
  image_ROI = image[100:image.shape[0], 250:image.shape[1]]
  image = cv2.resize(image, (0,0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
  # image = cv2.cvtColor(image_ROI, cv2.COLOR_BGR2RGB)
  return image


def getPixelCoordMesh(image, results,LEFT_EYE, RIGHT_EYE):
  img_h, img_w, img_c = image.shape
  mesh_coord = []

  if results.multi_face_landmarks:
    mesh_coord = [(int(point.x * img_w), int(point.y * img_h)) for point in results.multi_face_landmarks[0].landmark]
  # print(mesh_coord,"\n")
  return mesh_coord


def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance