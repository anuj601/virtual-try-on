import cv2
import mediapipe as mp

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Start webcam capture
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

try:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert the BGR image to RGB and process it with MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        # Draw the pose annotations on the image.
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Display the resulting image
        cv2.imshow('Virtual Try-On - Pose Detection', image)
        
        # Wait for 1 millisecond for a key press and check if it's 'q'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): # Press 'q' to quit
            print("Quitting...")
            break
        elif key == 27: # Also quit on 'ESC' key
            print("Quitting...")
            break

finally:
    # This block runs no matter how the loop ends (error or quit)
    print("Releasing resources...")
    cap.release()
    cv2.destroyAllWindows()
    # Sometimes needed on Windows to fully close the window
    for i in range(5):
        cv2.waitKey(1)