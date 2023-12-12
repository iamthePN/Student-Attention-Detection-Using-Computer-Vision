# Import necessary libraries
import cv2
import dlib
from scipy.spatial import distance as dist
import numpy as np

# Load pre-trained face cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load pre-trained facial landmarks predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Function to calculate eye aspect ratio
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Set constants for eye aspect ratio threshold, consecutive frames, head rotation threshold, and attention probability
EAR_THRESHOLD = 0.2
CONSECUTIVE_FRAMES_THRESHOLD = 3  
HEAD_ROTATION_THRESHOLD = 5

STATIC_ATTENTION_ANGLE_THRESHOLD = 0 
MAX_ATTENTION_PROBABILITY = 1.0  
MIN_ATTENTION_PROBABILITY = 0.85  

# Initialize counters and variables
frame_counter = 0
blink_counter = 0
head_rotation = "Forward"  
attention_probability = MAX_ATTENTION_PROBABILITY  

# Open video capture from file (in this case 'cam1.mp4')
cap = cv2.VideoCapture('cam1.mp4')

# Start the main loop for video processing
while True:
    # Read a frame from the video capture
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame using the face cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Check if no faces are detected
    if len(faces) == 0:
        # Display a warning message on the frame
        cv2.putText(frame, 'CRITICAL WARNING: No Face Detected!', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Iterate through detected faces
    for (x, y, w, h) in faces:
        # Extract region of interest (ROI) from the grayscale frame
        roi_gray = gray[y:y + h, x:x + w]
        
        # Use the facial landmarks predictor on the ROI
        landmarks = predictor(roi_gray, dlib.rectangle(0, 0, w, h))
        
        # Draw a rectangle around the face in the original frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Set offsets for drawing landmarks on the original frame
        x_offset = x  
        y_offset = y 

        # Calculate eye centers, nose tip, and distances for head rotation analysis
        left_eye_center = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)], axis=0)
        right_eye_center = np.mean([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)], axis=0)
        nose_tip = (landmarks.part(30).x, landmarks.part(30).y)
        left_eye_to_nose = dist.euclidean(left_eye_center, nose_tip)
        right_eye_to_nose = dist.euclidean(right_eye_center, nose_tip)

        # Determine head rotation based on eye-to-nose distances
        if left_eye_to_nose > right_eye_to_nose + HEAD_ROTATION_THRESHOLD:
            head_rotation = "Left"
        elif right_eye_to_nose > left_eye_to_nose + HEAD_ROTATION_THRESHOLD:
            head_rotation = "Right"
        else:
            head_rotation = "Forward"

        # Calculate head rotation angle
        head_rotation_angle = left_eye_to_nose - right_eye_to_nose

        # Calculate attention probability based on head rotation angle
        if abs(head_rotation_angle) <= STATIC_ATTENTION_ANGLE_THRESHOLD:
            attention_probability = MAX_ATTENTION_PROBABILITY
        else:
            attention_probability = MAX_ATTENTION_PROBABILITY - (abs(head_rotation_angle) - STATIC_ATTENTION_ANGLE_THRESHOLD) / (180 - STATIC_ATTENTION_ANGLE_THRESHOLD)

        # Draw facial landmarks on the original frame
        for i in range(68):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            cv2.circle(frame, (x + x_offset, y + y_offset), 2, (0, 0, 255), -1)

        # Extract eye regions for eye aspect ratio calculation
        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

        # Calculate eye aspect ratios
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        # Calculate average eye aspect ratio
        avg_ear = (left_ear + right_ear) / 2.0

        # Update blink and frame counters based on eye aspect ratio
        if avg_ear < EAR_THRESHOLD:
            frame_counter += 1
        else:
            if frame_counter >= CONSECUTIVE_FRAMES_THRESHOLD:
                blink_counter += 1
            frame_counter = 0

        # Display warning message if attention probability is low
        if attention_probability < MIN_ATTENTION_PROBABILITY:
            cv2.putText(frame, 'WARNING: Low Attention!', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display blink count, head rotation, and attention probability on the frame
    cv2.putText(frame, f'Eye Countdown: {blink_counter}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f'Head Rotation: {head_rotation}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f'Attention Probability: {attention_probability:.2f}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display the frame with annotations
    cv2.imshow('Attention Detection', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
