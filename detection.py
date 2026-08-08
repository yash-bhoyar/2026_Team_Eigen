import time
import cv2
import numpy as np

# Configurable Constants
BENDING_ANGLE_THRESHOLD = 90.0        # Degrees (angles below this indicate stooping/bending)
WARNING_DURATION_THRESHOLD = 5.0      # Continuous seconds for POSTURE_WARNING (Moderate Risk)
HIGH_RISK_DURATION_THRESHOLD = 15.0   # Continuous seconds for POSTURE_HIGH_RISK (Severe Risk)
RESTRICTED_ZONE = (100, 100, 350, 350)  # Rectangular box (x_min, y_min, x_max, y_max) in pixels

# Repetitive Motion Detection Constants
STANDING_ANGLE_THRESHOLD = 130.0      # Degrees (angles above this indicate standing/upright)
BENDING_REPETITIVE_THRESHOLD = 110.0  # Degrees (angles below this indicate bent torso for cycle)
REPETITIVE_CYCLE_THRESHOLD = 10       # Complete bend-rise cycles required for risk flag
REPETITIVE_WINDOW_SECONDS = 60.0      # Rolling time window in seconds


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
    """Handles unsafe posture severity tiers and restricted zone entry detection logic."""

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
        Returns:
          - 'RESTRICTED_ZONE_ENTRY': Core body landmark inside restricted box
          - 'POSTURE_HIGH_RISK': Sustained stooping >= 15.0 seconds
          - 'POSTURE_WARNING': Stooping >= 5.0 seconds but < 15.0 seconds
          - 'SAFE': Normal posture or brief bending (< 5.0 seconds)
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

        # 1. Check Restricted Zone Entry (Priority: Zone breach takes immediate precedence)
        x1, y1, x2, y2 = RESTRICTED_ZONE
        if x1 <= hip_x <= x2 and y1 <= hip_y <= y2:
            return "RESTRICTED_ZONE_ENTRY"

        # 2. Check Bad Bending Posture Severity Tiers
        back_angle = calculate_angle(shoulder, hip, knee)

        if back_angle < BENDING_ANGLE_THRESHOLD:
            if self.bend_start_time is None:
                self.bend_start_time = time.time()
            
            elapsed = time.time() - self.bend_start_time
            
            if elapsed >= HIGH_RISK_DURATION_THRESHOLD:
                return "POSTURE_HIGH_RISK"
            elif elapsed >= WARNING_DURATION_THRESHOLD:
                return "POSTURE_WARNING"
            else:
                # Brief bend (< 5 seconds) is considered normal movement / safe
                return "SAFE"
        else:
            self.bend_start_time = None
            return "SAFE"


def detect_repetitive_motion(
    angle_history,
    current_angle,
    current_time=None,
    standing_threshold=STANDING_ANGLE_THRESHOLD,
    bending_threshold=BENDING_REPETITIVE_THRESHOLD,
    cycle_threshold=REPETITIVE_CYCLE_THRESHOLD,
    window_seconds=REPETITIVE_WINDOW_SECONDS
):
    """
    Tracks rolling history of torso angles over a 60-second window and detects bend-rise cycles.
    A bend-rise cycle is a transition from standing (> 130°) to bent (< 110°) and back to standing (> 130°).
    
    Returns:
      - status: 'REPETITIVE_MOTION_RISK' if cycle count >= cycle_threshold else None
      - cycle_count: int count of completed bend-rise cycles within window_seconds
      - angle_history: updated list of (timestamp, angle) pairs within window_seconds
    """
    if current_time is None:
        current_time = time.time()

    if current_angle is not None:
        angle_history.append((current_time, current_angle))

    # Auto-discard entries older than 60 seconds
    angle_history = [entry for entry in angle_history if current_time - entry[0] <= window_seconds]

    # Count completed bend-rise cycles
    cycle_count = 0
    state = "STANDING"

    for _, angle in angle_history:
        if state == "STANDING":
            if angle < bending_threshold:
                state = "BENT"
        elif state == "BENT":
            if angle > standing_threshold:
                state = "STANDING"
                cycle_count += 1

    status = "REPETITIVE_MOTION_RISK" if cycle_count >= cycle_threshold else None
    return status, cycle_count, angle_history

