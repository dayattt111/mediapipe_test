import cv2
import numpy as np
import mediapipe as mp

# Initialize drawing and hands solutions
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Define drawing specifications for hand landmarks and connections
landmark_drawing_spec = mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
connection_drawing_spec = mp_drawing.DrawingSpec(thickness=2, color=(0, 0, 255))

# Configure hand detection and tracking parameters
min_detection_confidence = 0.8  # Minimum confidence for hand detection
min_tracking_confidence = 0.5  # Minimum confidence for hand tracking

# Capture video from webcam
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    min_detection_confidence=min_detection_confidence,
    min_tracking_confidence=min_tracking_confidence
) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to capture frame from the webcam.")
            break

        # Convert BGR frame to RGB for MediaPipe processing
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Flip the image horizontally for a more natural view (optional)
        # image = cv2.flip(image, 1)

        # Set image flags for efficient processing
        image.flags.writeable = False

        # Detect hand landmarks in the image
        results = hands.process(image)

        # Convert image back to BGR for OpenCV display
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw hand landmarks and connections if results are found
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS, connection_drawing_spec, landmark_drawing_spec
                )

                # Extract fingertip landmarks for detection
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                # Calculate the distance between the palm base and each fingertip
                # (replace with your preferred distance calculation method)
                palm_base = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                thumb_distance = np.linalg.norm(np.array([thumb_tip.x, thumb_tip.y]) - np.array([palm_base.x, palm_base.y]))
                index_distance = np.linalg.norm(np.array([index_tip.x, index_tip.y]) - np.array([palm_base.x, palm_base.y]))
                middle_distance = np.linalg.norm(np.array([middle_tip.x, middle_tip.y]) - np.array([palm_base.x, palm_base.y]))
                ring_distance = np.linalg.norm(np.array([ring_tip.x, ring_tip.y]) - np.array([palm_base.x, palm_base.y]))
                pinky_distance = np.linalg.norm(np.array([pinky_tip.x, pinky_tip.y]) - np.array([palm_base.x, palm_base.y]))

                # Define thresholds for finger extension (adjust as needed)
                thumb_threshold = 50  # Adjust for thumb extension sensitivity
                finger_threshold = 30  # Adjust for other finger extension sensitivity

                # Detect extended fingers based on distance thresholds
                extended_fingers = []
                if thumb_distance > thumb_threshold:
                    extended_fingers.append("Thumb")
                if index_distance > finger_threshold:
                    extended_fingers.append("Index")
                if middle_distance > finger_threshold:
                    extended_