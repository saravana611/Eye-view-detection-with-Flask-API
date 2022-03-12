# Eye-view-detection-with-Flask-API
This is Eyes 👁️ 👁️ Tracking Project, here we will use computer vision techniques to extracting the eyes, Mediapipe python modules will provide the face landmarks and a person is looking screen or not with Flask API.

## Specifications
Pass a video stream object, identify whether the detected person is looking left or right apart from the screen. Json response must send response as {“LookedSide”:”true”}

**API Endpoints:**
* main_frame : retuns the live video streaming using API
* video_feed : return the live video feed with predictionn i.e person is looking screen or somewhere else
* video_json : return json response of live feed prediction.

# Installation

 Run: pip install -r requirements.txt

# How to Run

              
## Run API Server
python app.py

## Run API client - Web
Simply open a web browser and enter:

http://127.0.0.1:5001/main_frame

http://127.0.0.1:5001/video_feed

http://127.0.0.1:5001/video_json

 and get results.
