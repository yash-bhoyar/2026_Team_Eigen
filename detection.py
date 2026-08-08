import time
import cv2
import numpy as np

# Configurable Constants
BENDING_ANGLE_THRESHOLD = 90.0      # Degrees (angles below this indicate stooping/bending)
BENDING_DURATION_THRESHOLD = 2.0    # Continuous seconds before triggering UNSAFE_POSTURE
RESTRICTED_ZONE = (100, 100, 350, 350)  # Rectangular box (x_min, y_min, x_max, y_max) in pixels

# MediaPipe Pose Landmark Indices
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26


def calculate_angle(a, b, c):
    """
    Calculates interior angle (in degrees) at vertex b between line segments ba and bc.
    a: Shoulder coordinates [x, y]
    b: Hip coordinates [x, y]
    c: Knee coordinates [x, y]
    """
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return angle


class SafetyDetector:
    """Handles unsafe posture and restricted zone entry detection logic."""

    def __init__(self):
        self.bend_start_time = None

    def draw_restricted_zone(self, frame):
        """Draws the restricted zone rectangle outline and text label on the frame."""
        x1, y1, x2, y2 = RESTRICTED_ZONE
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            frame,
            "RESTRICTED ZONE",
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )
        return frame

    def detect(self, pose_landmarks, frame_width, frame_height):
        """
        Evaluates pose landmarks against safety rules.
        Returns 'RESTRICTED_ZONE_ENTRY', 'UNSAFE_POSTURE', or 'SAFE'.
        """
        if not pose_landmarks:
            self.bend_start_time = None
            return "SAFE"

        landmarks = pose_landmarks.landmark

        # Calculate torso center coordinates (average of left & right body landmarks) in pixels
        shoulder_x = (landmarks[LEFT_SHOULDER].x + landmarks[RIGHT_SHOULDER].x) / 2 * frame_width
        shoulder_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2 * frame_height

        hip_x = (landmarks[LEFT_HIP].x + landmarks[RIGHT_HIP].x) / 2 * frame_width
        hip_y = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2 * frame_height

        knee_x = (landmarks[LEFT_KNEE].x + landmarks[RIGHT_KNEE].x) / 2 * frame_width
        knee_y = (landmarks[LEFT_KNEE].y + landmarks[RIGHT_KNEE].y) / 2 * frame_height

        shoulder = [shoulder_x, shoulder_y]
        hip = [hip_x, hip_y]
        knee = [knee_x, knee_y]

        # 1. Check Restricted Zone Entry (using Hip center point)
        x1, y1, x2, y2 = RESTRICTED_ZONE
        if x1 <= hip_x <= x2 and y1 <= hip_y <= y2:
            return "RESTRICTED_ZONE_ENTRY"

        # 2. Check Bad Bending Posture (using torso/back angle at hip)
        back_angle = calculate_angle(shoulder, hip, knee)

        if back_angle < BENDING_ANGLE_THRESHOLD:
            if self.bend_start_time is None:
                self.bend_start_time = time.time()
            elif time.time() - self.bend_start_time >= BENDING_DURATION_THRESHOLD:
                return "UNSAFE_POSTURE"
        else:
            self.bend_start_time = None

        return "SAFE"
