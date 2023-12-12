# Project Title

The provided code is a Python script for real-time attention detection using facial landmarks and eye blink analysis. It utilizes the OpenCV library for video capture and image processing, Dlib for facial landmark detection, and Scipy for distance calculations. The code detects faces in a video stream, analyzes the orientation of the head based on the positions of facial landmarks, calculates the eye aspect ratio to detect blinks, and estimates the user's attention level based on head rotation. Warning messages are displayed for low attention levels or when no face is detected. The application counts blinks, monitors head rotation, and provides an attention probability, offering a simple system for assessing and alerting potential lapses in attention.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Installation](#installation)


## Overview

The provided code implements an attention detection system using facial landmarks and eye movement analysis. The project uses computer vision techniques, leveraging the OpenCV and dlib libraries, to process video frames from a source (in this case, 'cam1.mp4'). The key features of the system include:

Face Detection: Utilizing the Haar Cascade classifier, the system detects faces within each video frame.

Facial Landmarks: The dlib library is employed to predict 68 facial landmarks on the detected face, aiding in the analysis of eye movement and head rotation.

Eye Aspect Ratio (EAR): The code calculates the EAR for each eye based on the spatial relationships between specific facial landmarks. This ratio is used to determine blink events.

Head Rotation Analysis: The system assesses the rotation of the user's head by comparing the distances from the eyes to the nose tip, identifying if the head is turned left, right, or forward.

Attention Probability: The code computes an attention probability based on the angle of head rotation. Low attention probability triggers warnings in the frame.

Blink Detection: The system counts the number of blinks by monitoring consecutive frames where the average EAR falls below a predefined threshold.

Visual Feedback: The processed frames are annotated with facial landmarks, head rotation status, blink count, and attention probability, providing a visual representation of the attention-related metrics.

## Requirements

Run "pip install -r requirements.txt" to install the Prerequisites Libraries

1) Python >= 3.6
2) ibraries (Mentioned in "requirements.txt")


## Installation

Run Student Attention Detection: Navigate to the project folder in your terminal or command prompt. Execute the attention.py script to start the application using "python attention.py".

    To Detect only face:
        Run Face Detection: Navigate to the project folder in your terminal or command prompt. Execute the face_detection.py script to  
                            start the application using "python face_detection.py".


