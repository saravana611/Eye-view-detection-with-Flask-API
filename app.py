from flask import Flask, render_template, Response
import cv2
import numpy as np
import time
import json

from turtle import pos
import cv2 as cv
import cv2
import mediapipe as mp
import utils, math


app=Flask(__name__)
camera = cv2.VideoCapture(0)

# variables 
global CEF_COUNTER
CEF_COUNTER =0
TOTAL_BLINKS =0
# constants
CLOSED_EYES_FRAME =3
FONTS =cv.FONT_HERSHEY_COMPLEX
position = False
# face bounder indices 
FACE_OVAL=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

# lips indices for Landmarks
LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]
LOWER_LIPS =[61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
UPPER_LIPS=[ 185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78] 
# Left eyes indices 
LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]
LEFT_EYEBROW =[ 336, 296, 334, 293, 300, 276, 283, 282, 295, 285 ]

# right eyes indices
RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  
RIGHT_EYEBROW=[ 70, 63, 105, 66, 107, 55, 65, 52, 53, 46 ]

map_face_mesh = mp.solutions.face_mesh
# camera object 

# landmark detection function 
def landmarksDetection(img, results, draw=False):
    img_height, img_width= img.shape[:2]
    # list[(x,y), (x,y)....]
    mesh_coord = [(int(point.x * img_width), int(point.y * img_height)) for point in results.multi_face_landmarks[0].landmark]
    if draw :
        [cv.circle(img, p, 2, (0,255,0), -1) for p in mesh_coord]

    # returning the list of tuples for each landmarks 
    return mesh_coord

# Euclaidean distance 
def euclaideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x)**2 + (y1 - y)**2)
    return distance

# Blinking Ratio
def blinkRatio(img, landmarks, right_indices, left_indices):
    # Right eyes 
    # horizontal line 
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line 
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]
    # draw lines on right eyes 
    # cv.line(img, rh_right, rh_left, utils.GREEN, 2)
    # cv.line(img, rv_top, rv_bottom, utils.WHITE, 2)

    # LEFT_EYE 
    # horizontal line 
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]

    # vertical line 
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclaideanDistance(rh_right, rh_left)
    rvDistance = euclaideanDistance(rv_top, rv_bottom)

    lvDistance = euclaideanDistance(lv_top, lv_bottom)
    lhDistance = euclaideanDistance(lh_right, lh_left)

    reRatio = rhDistance/rvDistance
    leRatio = lhDistance/lvDistance

    ratio = (reRatio+leRatio)/2
    return ratio 

# Eyes Extrctor function,
def eyesExtractor(img, right_eye_coords, left_eye_coords):
    # converting color image to  scale image 
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # getting the dimension of image 
    dim = gray.shape

    # creating mask from gray scale dim
    mask = np.zeros(dim, dtype=np.uint8)

    # drawing Eyes Shape on mask with white color 
    cv.fillPoly(mask, [np.array(right_eye_coords, dtype=np.int32)], 255)
    cv.fillPoly(mask, [np.array(left_eye_coords, dtype=np.int32)], 255)

    # showing the mask 
    # cv.imshow('mask', mask)
    
    # draw eyes image on mask, where white shape is 
    eyes = cv.bitwise_and(gray, gray, mask=mask)
    # change black color to gray other than eys 
    # cv.imshow('eyes draw', eyes)
    eyes[mask==0]=155
    
    # getting minium and maximum x and y  for right and left eyes 
    # For Right Eye 
    r_max_x = (max(right_eye_coords, key=lambda item: item[0]))[0]
    r_min_x = (min(right_eye_coords, key=lambda item: item[0]))[0]
    r_max_y = (max(right_eye_coords, key=lambda item : item[1]))[1]
    r_min_y = (min(right_eye_coords, key=lambda item: item[1]))[1]

    # For LEFT Eye
    l_max_x = (max(left_eye_coords, key=lambda item: item[0]))[0]
    l_min_x = (min(left_eye_coords, key=lambda item: item[0]))[0]
    l_max_y = (max(left_eye_coords, key=lambda item : item[1]))[1]
    l_min_y = (min(left_eye_coords, key=lambda item: item[1]))[1]

    # croping the eyes from mask 
    cropped_right = eyes[r_min_y: r_max_y, r_min_x: r_max_x]
    cropped_left = eyes[l_min_y: l_max_y, l_min_x: l_max_x]

    # returning the cropped eyes 
    return cropped_right, cropped_left

# Eyes Postion Estimator 
def positionEstimator(cropped_eye):
    # getting height and width of eye 
    h, w =cropped_eye.shape
    
    # remove the noise from images
    gaussain_blur = cv.GaussianBlur(cropped_eye, (9,9),0)
    median_blur = cv.medianBlur(gaussain_blur, 3)

    # applying thrsholding to convert binary_image
    ret, threshed_eye = cv.threshold(median_blur, 130, 255, cv.THRESH_BINARY)

    # create fixd part for eye with 
    piece = int(w/3) 

    # slicing the eyes into three parts 
    right_piece = threshed_eye[0:h, 0:piece]
    center_piece = threshed_eye[0:h, piece: piece+piece]
    left_piece = threshed_eye[0:h, piece +piece:w]
    
    # calling pixel counter function
    eye_position, color = pixelCounter(right_piece, center_piece, left_piece)

    return eye_position, color 

# creating pixel counter function 
def pixelCounter(first_piece, second_piece, third_piece):
    # counting black pixel in each part 
    right_part = np.sum(first_piece==0)
    center_part = np.sum(second_piece==0)
    left_part = np.sum(third_piece==0)
    # creating list of these values
    eye_parts = [right_part, center_part, left_part]

    # getting the index of max values in the list 
    max_index = eye_parts.index(max(eye_parts))
    pos_eye ='' 
    if max_index==0:
        pos_eye="RIGHT"
        color=[utils.BLACK, utils.GREEN]
    elif max_index==1:
        pos_eye = 'CENTER'
        color = [utils.YELLOW, utils.PINK]
    elif max_index ==2:
        pos_eye = 'LEFT'
        color = [utils.GRAY, utils.YELLOW]
    else:
        pos_eye="Closed"
        color = [utils.GRAY, utils.YELLOW]
    return pos_eye, color 


def gen_frames(CEF_COUNTER):
        
    with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:
        start_time = time.time()
        while True:
            success, frame = camera.read()  # read the camera frame
            if not success:
                break
            else: 

                ret, frame = camera.read() 
                frame = cv2.flip(frame,1)

                frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
                frame_height, frame_width= frame.shape[:2]
                rgb_frame = frame[:, :, ::-1]
                results  = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    mesh_coords = landmarksDetection(frame, results, False)
                    ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
                    

                    if ratio >5.5:
                        CEF_COUNTER +=1
                        
                        utils.colorBackgroundText(frame,  f'Blink', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )

                    else:
                        if CEF_COUNTER>CLOSED_EYES_FRAME:                                           
                            #TOTAL_BLINKS +=1
                            CEF_COUNTER =0
                    
                
                    cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
                    cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)

                    # Blink Detector Counter Completed
                    right_coords = [mesh_coords[p] for p in RIGHT_EYE]
                    left_coords = [mesh_coords[p] for p in LEFT_EYE]
                    crop_right, crop_left = eyesExtractor(rgb_frame, right_coords, left_coords)

 
                    eye_position, color = positionEstimator(crop_right)
                    
                    eye_position_left, color = positionEstimator(crop_left)
                    if eye_position != "CENTER" and eye_position_left != "CENTER":                    
                        position = True
                    else:
                        position = False   
                    utils.colorBackgroundText(frame, f'LookedSide: {position}', FONTS, 1.0, (40, 320), 2, color[0], color[1], 8, 8)
                                                           
                    #json.dump("no",position)
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def gen_json(CEF_COUNTER):
        
    with map_face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5) as face_mesh:
        start_time = time.time()
        while True:
            success, frame = camera.read()  # read the camera frame
            if not success:
                break
            else: 

                ret, frame = camera.read() 
                frame = cv2.flip(frame,1)

                frame = cv.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv.INTER_CUBIC)
                frame_height, frame_width= frame.shape[:2]
                rgb_frame = frame[:, :, ::-1]
                results  = face_mesh.process(rgb_frame)

                if results.multi_face_landmarks:
                    mesh_coords = landmarksDetection(frame, results, False)
                    ratio = blinkRatio(frame, mesh_coords, RIGHT_EYE, LEFT_EYE)
                    

                    if ratio >5.5:
                        CEF_COUNTER +=1
                        
                        utils.colorBackgroundText(frame,  f'Blink', FONTS, 1.7, (int(frame_height/2), 100), 2, utils.YELLOW, pad_x=6, pad_y=6, )

                    else:
                        if CEF_COUNTER>CLOSED_EYES_FRAME:                                           
                            #TOTAL_BLINKS +=1
                            CEF_COUNTER =0
                    
                
                    cv.polylines(frame,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)
                    cv.polylines(frame,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, utils.GREEN, 1, cv.LINE_AA)

                    # Blink Detector Counter Completed
                    right_coords = [mesh_coords[p] for p in RIGHT_EYE]
                    left_coords = [mesh_coords[p] for p in LEFT_EYE]
                    crop_right, crop_left = eyesExtractor(rgb_frame, right_coords, left_coords)

 
                    eye_position, color = positionEstimator(crop_right)
                    
                    eye_position_left, color = positionEstimator(crop_left)
                    if eye_position != "CENTER" and eye_position_left != "CENTER":                    
                        position = True
                    else:
                        position = False   
                    utils.colorBackgroundText(frame, f'LookedSide: {position}', FONTS, 1.0, (40, 320), 2, color[0], color[1], 8, 8)
                                                           
                    #json.dump("no",position)
                return position

                
                

def main_frame():  # generate frame by frame from camera
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/main_frame')
def main():
    """Video streaming home page."""
    return Response(main_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/', methods=["GET"])
def index():
    return render_template('index.html')
@app.route('/video_feed', methods=["GET"])
def video_feed():
    return Response(gen_frames(CEF_COUNTER), mimetype='multipart/x-mixed-replace; boundary=frame') 

@app.route('/video_json', methods=["GET"])
def video_feed_json():
            ret = gen_json(CEF_COUNTER) 
            resp_data = {"LookedSide": str(ret)} # convert ret (numpy._bool) to bool for json.dumps
            return json.dumps(resp_data)


     

if __name__=='__main__':
    app.run(debug=True)

    