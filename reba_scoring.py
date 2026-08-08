"""
reba_scoring.py
================
Rapid Entire Body Assessment (REBA) Risk Scoring Module for SafeGuard.

Official Reference:
Hignett, S., & McAtamney, L. (2000). Rapid Entire Body Assessment (REBA).
Applied Ergonomics, 31(2), 201-205.

This module evaluates trunk and neck flexion angles derived from MediaPipe pose
landmarks and computes REBA-inspired risk scores and action levels.
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional
import numpy as np

from detection import calculate_angle

# ==============================================================================
# REBA OFFICIAL BODY-PART SCORING TABLE CONSTANTS
# Citing: REBA Ergonomic Assessment Standard (Hignett & McAtamney, 2000)
# ==============================================================================

# TRUNK FLEXION SCORING TABLE
# Flexion 0° - 20°   -> Score 1
# Flexion 20° - 60°  -> Score 2
# Flexion > 60°      -> Score 3
REBA_TRUNK_FLEXION_LEVEL_1_MAX_DEG = 20.0
REBA_TRUNK_FLEXION_LEVEL_2_MAX_DEG = 60.0

REBA_TRUNK_BASE_SCORE_LEVEL_1 = 1
REBA_TRUNK_BASE_SCORE_LEVEL_2 = 2
REBA_TRUNK_BASE_SCORE_LEVEL_3 = 3

# NECK FLEXION SCORING TABLE
# Flexion 0° - 20°   -> Score 1
# Flexion > 20°      -> Score 2
REBA_NECK_FLEXION_LEVEL_1_MAX_DEG = 20.0

REBA_NECK_BASE_SCORE_LEVEL_1 = 1
REBA_NECK_BASE_SCORE_LEVEL_2 = 2

# REPETITIVE MOTION ACTIVITY BONUS
# Activity score bonus (+1) when high-frequency repetitive motion is present
REBA_REPETITIVE_MOTION_ACTIVITY_BONUS = 1

# MEDIAPIPE POSE LANDMARK INDICES
LEFT_EAR = 7
RIGHT_EAR = 8
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
LEFT_KNEE = 25
RIGHT_KNEE = 26


@dataclass
class REBAScoreResult:
    """Structured result holding REBA scoring breakdown and overall action level."""
    trunk_flexion_angle: float
    neck_flexion_angle: float
    trunk_score: int
    neck_score: int
    repetitive_bonus: int
    combined_score: int
    action_level: int
    action_label: str

    def to_dict(self) -> Dict[str, Any]:
        """Returns the result as a standard Python dictionary."""
        return asdict(self)


def calculate_trunk_score(trunk_flexion_angle: float) -> int:
    """
    Computes REBA Trunk Score based on trunk flexion angle (in degrees).
    Official REBA Table:
      - 0° to 20° flexion  = 1
      - 20° to 60° flexion = 2
      - > 60° flexion      = 3
    Note: Trunk twist/side-bend adjustment (+1) is not yet measured (requires 3D landmarks).
    """
    flexion = abs(trunk_flexion_angle)
    if flexion <= REBA_TRUNK_FLEXION_LEVEL_1_MAX_DEG:
        return REBA_TRUNK_BASE_SCORE_LEVEL_1
    elif flexion <= REBA_TRUNK_FLEXION_LEVEL_2_MAX_DEG:
        return REBA_TRUNK_BASE_SCORE_LEVEL_2
    else:
        return REBA_TRUNK_BASE_SCORE_LEVEL_3


def calculate_neck_score(neck_flexion_angle: float) -> int:
    """
    Computes REBA Neck Score based on neck flexion angle (in degrees).
    Official REBA Table:
      - 0° to 20° flexion = 1
      - > 20° flexion     = 2
    Note: Neck twist/side-bend adjustment (+1) is not yet measured (requires 3D landmarks).
    """
    flexion = abs(neck_flexion_angle)
    if flexion <= REBA_NECK_FLEXION_LEVEL_1_MAX_DEG:
        return REBA_NECK_BASE_SCORE_LEVEL_1
    else:
        return REBA_NECK_BASE_SCORE_LEVEL_2


def map_reba_action_level(combined_score: int) -> tuple:
    """
    Maps combined REBA score to REBA Action Levels and risk descriptions:
      - Score 1-2: Action Level 0 -> 'Negligible risk'
      - Score 3-4: Action Level 1 -> 'Low risk, may need action'
      - Score 5-6: Action Level 2 -> 'Medium risk, further investigation needed'
      - Score 7+:  Action Level 3 -> 'High risk, investigate and implement change soon'
    """
    if combined_score <= 2:
        return 0, "Negligible risk"
    elif combined_score <= 4:
        return 1, "Low risk, may need action"
    elif combined_score <= 6:
        return 2, "Medium risk, further investigation needed"
    else:
        return 3, "High risk, investigate and implement change soon"


def evaluate_reba_score(
    pose_landmarks,
    frame_width: int,
    frame_height: int,
    is_repetitive_risk: bool = False
) -> Optional[REBAScoreResult]:
    """
    Calculates trunk flexion angle, neck flexion angle, individual REBA scores,
    and combined action level from MediaPipe pose landmarks.
    
    Args:
        pose_landmarks: MediaPipe Pose landmarks object.
        frame_width: Video frame width in pixels.
        frame_height: Video frame height in pixels.
        is_repetitive_risk: Boolean flag indicating if REPETITIVE_MOTION_RISK is active (+1 bonus).
        
    Returns:
        REBAScoreResult dataclass instance or None if pose_landmarks is None.
    """
    if not pose_landmarks:
        return None

    landmarks = pose_landmarks.landmark

    # 1. Ear center coordinates (average of left & right ear)
    ear_x = (landmarks[LEFT_EAR].x + landmarks[RIGHT_EAR].x) / 2 * frame_width
    ear_y = (landmarks[LEFT_EAR].y + landmarks[RIGHT_EAR].y) / 2 * frame_height

    # 2. Shoulder center coordinates
    shoulder_x = (landmarks[LEFT_SHOULDER].x + landmarks[RIGHT_SHOULDER].x) / 2 * frame_width
    shoulder_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2 * frame_height

    # 3. Hip center coordinates
    hip_x = (landmarks[LEFT_HIP].x + landmarks[RIGHT_HIP].x) / 2 * frame_width
    hip_y = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2 * frame_height

    # 4. Knee center coordinates
    knee_x = (landmarks[LEFT_KNEE].x + landmarks[RIGHT_KNEE].x) / 2 * frame_width
    knee_y = (landmarks[LEFT_KNEE].y + landmarks[RIGHT_KNEE].y) / 2 * frame_height

    ear = [ear_x, ear_y]
    shoulder = [shoulder_x, shoulder_y]
    hip = [hip_x, hip_y]
    knee = [knee_x, knee_y]

    # Calculate Trunk interior angle (shoulder-hip-knee)
    trunk_interior_angle = calculate_angle(shoulder, hip, knee)
    # Flexion angle is deviation from straight posture (180°)
    trunk_flexion_angle = abs(180.0 - trunk_interior_angle)

    # Calculate Neck interior angle (ear-shoulder-hip)
    neck_interior_angle = calculate_angle(ear, shoulder, hip)
    # Flexion angle is deviation from straight posture (180°)
    neck_flexion_angle = abs(180.0 - neck_interior_angle)

    # Calculate individual body part scores
    trunk_score = calculate_trunk_score(trunk_flexion_angle)
    neck_score = calculate_neck_score(neck_flexion_angle)

    # Apply repetitive motion activity bonus
    repetitive_bonus = REBA_REPETITIVE_MOTION_ACTIVITY_BONUS if is_repetitive_risk else 0

    # Combine scores
    combined_score = trunk_score + neck_score + repetitive_bonus

    # Map to action level and label
    action_level, action_label = map_reba_action_level(combined_score)

    return REBAScoreResult(
        trunk_flexion_angle=round(trunk_flexion_angle, 1),
        neck_flexion_angle=round(neck_flexion_angle, 1),
        trunk_score=trunk_score,
        neck_score=neck_score,
        repetitive_bonus=repetitive_bonus,
        combined_score=combined_score,
        action_level=action_level,
        action_label=action_label
    )
