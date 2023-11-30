import cv2
import mediapipe as mp

faceDetector = mp.solutions.face_detection
drawing = mp.solutions.drawing_utils
# STATUS = {} 

def faceDetection(image, face_detection, STATUS):
    # global STATUS
    results = face_detection.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.detections:
        STATUS['FACE'] = "TRUE"
        for id, detection in enumerate(results.detections):
            drawing.draw_detection(image, detection)
            bBox = detection.location_data.relative_bounding_box

            h, w, c = image.shape
            boundingBox = int(bBox.xmin * w), int(bBox.ymin * h), int(bBox.width * w), int(bBox.height * h)
        
    else:
        STATUS['FACE'] = "FALSE"
    
    return image, STATUS