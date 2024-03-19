import cv2
import mediapipe as mp
import numpy as np
from Focuser import Focuser

focuser = None
def focusing(val):
	# value = (val << 4) & 0x3ff0
	# data1 = (value >> 8) & 0x3f
	# data2 = value & 0xf0
	# os.system("i2cset -y 6 0x0c %d %d" % (data1,data2))
    focuser.set(Focuser.OPT_FOCUS, val)

def gstreamer_pipeline (capture_width=1280, capture_height=720, display_width=640, display_height=360, framerate=60, flip_method=0) :   
    return ('nvarguscamerasrc ! ' 
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))

## initialize pose estimator
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

# Set the camera frame size
camera_frame_width = 1920
camera_frame_height = 1080

cap.set(3, camera_frame_width)
cap.set(4, camera_frame_height)

real_width = 100
camera_pix = 720
pixel_to_cm = real_width/camera_pix

cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Frame', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.isOpened():
     # Read frame
     _, frame = cap.read()

     # Resize and convert to RGB
     frame = cv2.resize(frame, (1920, 1080))
     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

     # Pose detection
     results = pose.process(frame_rgb)

     if results.pose_landmarks:
          # Drawing landmarks on the frame
          mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
          
          landmark_index = 12 # Change this to your landmark of interest
          landmark = results.pose_landmarks.landmark[landmark_index]
          x1, y1 = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])

          landmark_index = 11 # Change this to your landmark of interest
          landmark = results.pose_landmarks.landmark[landmark_index]
          x2, y2 = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])

          distance = np.sqrt(((x1-x2)**2) + ((y1-y2)**2)) # in pixel

          measured_dist = round(distance * pixel_to_cm, 2)

          text_x = 10
          text_y1 = 30
          text_y2 = 60
          text_dist = 90

          # Display coordinates on the frame
          cv2.putText(frame, f'X1: {x1}, Y1: {y1}', (text_x, text_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
          cv2.putText(frame, f'X2: {x2}, Y2: {y2}', (text_x, text_y2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
          cv2.putText(frame, f'Distance: {measured_dist} cm', (text_x, text_dist), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

     # Display the frame
     cv2.imshow('Frame', frame)

     if cv2.waitKey(10) & 0xFF == ord('q'):
          break

cap.release()
cv2.destroyAllWindows()