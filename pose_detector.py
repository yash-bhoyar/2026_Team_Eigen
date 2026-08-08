import cv2
import mediapipe as mp


class PoseDetector:
    """Handles MediaPipe Pose detection and skeleton landmark rendering."""

    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def process_frame(self, frame):
        """
        Takes an OpenCV BGR frame, detects pose landmarks, 
        draws full-body skeleton overlay, and returns the modified frame.
        """
        # Convert frame color space from BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process pose detection
        results = self.pose.process(rgb_frame)

        # Draw pose skeleton landmarks and connections if detected
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )

        return frame, results.pose_landmarks
