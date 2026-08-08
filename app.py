import time
from datetime import datetime
import cv2
import numpy as np
import streamlit as st

from pose_detector import PoseDetector
from detection import (
    SafetyDetector,
    calculate_angle,
    detect_repetitive_motion,
    BENDING_ANGLE_THRESHOLD,
    WARNING_DURATION_THRESHOLD,
    HIGH_RISK_DURATION_THRESHOLD,
    RESTRICTED_ZONE,
    STANDING_ANGLE_THRESHOLD,
    BENDING_REPETITIVE_THRESHOLD,
    REPETITIVE_CYCLE_THRESHOLD,
    REPETITIVE_WINDOW_SECONDS,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_KNEE,
    RIGHT_KNEE
)
import uuid
import threading
import pyttsx3
from reba_scoring import evaluate_reba_score
from firebase_db import get_firebase_status, log_incident_to_firestore


def _speak_worker(message_text: str):
    """Worker function executed in a background thread to synthesize speech without blocking UI."""
    try:
        engine = pyttsx3.init()
        engine.say(message_text)
        engine.runAndWait()
        engine.stop()
    except Exception as e:
        print(f"[Voice Alert] Speech synthesis warning: {e}")


def trigger_voice_alert_async(message_text: str):
    """Launches speech synthesis asynchronously in a separate background thread."""
    t = threading.Thread(target=_speak_worker, args=(message_text,), daemon=True)
    t.start()






# Set Streamlit Page Config
st.set_page_config(
    page_title="SafeGuard - Industrial Safety Monitor",
    page_icon="🛡️",
    layout="wide"
)

# ==============================================================================
# INDUSTRIAL SAFETY SIGNAGE STYLING (CSS)
# ==============================================================================
INDUSTRIAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@500;700&family=IBM+Plex+Sans:wght@400;600;700&family=Oswald:wght@500;700&display=swap');

/* Main Page Styling */
.stApp {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Industrial Header Banner */
.safety-header {
    background: linear-gradient(135deg, #161b22 0%, #0d1117 100%);
    border-bottom: 4px solid #ffcc00;
    border-left: 8px solid #ffcc00;
    padding: 16px 24px;
    margin-bottom: 20px;
    border-radius: 4px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5);
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.safety-header-title {
    font-family: 'Oswald', sans-serif;
    font-size: 2.2rem;
    letter-spacing: 2px;
    color: #ffcc00;
    margin: 0;
    text-transform: uppercase;
}

.safety-header-subtitle {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #8b949e;
    letter-spacing: 1px;
    margin-top: 4px;
}

/* Status Cards & Banners */
.status-banner {
    padding: 14px 20px;
    border-radius: 6px;
    font-family: 'Oswald', sans-serif;
    font-size: 1.3rem;
    letter-spacing: 1.5px;
    font-weight: 700;
    text-transform: uppercase;
    margin-bottom: 15px;
    display: flex;
    align-items: center;
    gap: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.3);
}

.status-safe {
    background-color: rgba(16, 185, 129, 0.15);
    border: 2px solid #10b981;
    color: #34d399;
}

.status-warning {
    background-color: rgba(255, 107, 0, 0.15);
    border: 2px solid #ff6b00;
    color: #ff8c38;
}

.status-high-risk {
    background-color: rgba(239, 68, 68, 0.2);
    border: 2px solid #ef4444;
    color: #f87171;
}

.status-critical {
    background-color: rgba(220, 38, 38, 0.25);
    border: 2px solid #dc2626;
    color: #f87171;
}

.status-standby {
    background-color: rgba(107, 114, 128, 0.15);
    border: 2px solid #6b7280;
    color: #9ca3af;
}

/* Industrial Instrument / Gauge Card */
.gauge-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-top: 3px solid #ffcc00;
    border-radius: 6px;
    padding: 16px;
    margin-bottom: 15px;
}

.gauge-title {
    font-family: 'Oswald', sans-serif;
    font-size: 1rem;
    letter-spacing: 1px;
    color: #8b949e;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.gauge-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.5rem;
    font-weight: 700;
    color: #ffcc00;
}

.gauge-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #8b949e;
    margin-top: 4px;
}

/* Sidebar & Log Cards */
.warning-counter-box {
    background: #161b22;
    border: 2px solid #ff6b00;
    border-radius: 6px;
    padding: 12px;
    text-align: center;
    margin-bottom: 16px;
}

.warning-counter-num {
    font-family: 'Oswald', sans-serif;
    font-size: 2.2rem;
    font-weight: 700;
    color: #ff6b00;
}

.warning-log-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 4px solid #ff6b00;
    border-radius: 4px;
    padding: 10px;
    margin-bottom: 8px;
    font-size: 0.85rem;
}

.warning-log-card.high-risk {
    border-left-color: #ef4444;
}

.warning-log-card.zone-breach {
    border-left-color: #dc2626;
}

.warning-log-time {
    font-family: 'IBM Plex Mono', monospace;
    color: #8b949e;
    font-size: 0.75rem;
}

.warning-log-title {
    font-family: 'Oswald', sans-serif;
    font-size: 0.95rem;
    letter-spacing: 0.5px;
    color: #f0f6fc;
    margin-top: 2px;
}

/* Error Card */
.error-card {
    background: rgba(239, 68, 68, 0.1);
    border: 2px dashed #ef4444;
    border-radius: 6px;
    padding: 20px;
    color: #f87171;
    font-family: 'IBM Plex Mono', monospace;
    margin-top: 20px;
}
</style>
"""

st.markdown(INDUSTRIAL_CSS, unsafe_allow_html=True)

# Initialize Session State Variables
if "warning_log" not in st.session_state:
    st.session_state.warning_log = []
if "last_logged_status" not in st.session_state:
    st.session_state.last_logged_status = "SAFE"
if "report_requested" not in st.session_state:
    st.session_state.report_requested = False
if "angle_history" not in st.session_state:
    st.session_state.angle_history = []
if "last_logged_reba_level" not in st.session_state:
    st.session_state.last_logged_reba_level = 0
if "bend_timestamps" not in st.session_state:
    st.session_state.bend_timestamps = []
if "last_voice_time" not in st.session_state:
    st.session_state.last_voice_time = 0.0
if "was_bent" not in st.session_state:
    st.session_state.was_bent = False

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())[:8]


# Header Banner
st.markdown("""
<div class="safety-header">
    <div>
        <h1 class="safety-header-title">🛡️ SAFEGUARD // WORKER SAFETY MONITOR</h1>
        <div class="safety-header-subtitle">REAL-TIME ERGONOMIC POSTURE TIERS & ZONE PROTECTION SYSTEM</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar System Controls
st.sidebar.markdown("### ⚙️ SYSTEM CONTROLS")
camera_index = st.sidebar.selectbox("Webcam Device Index", options=[0, 1, 2], index=0)
run_camera = st.sidebar.checkbox("Start Camera Stream", value=False)
fps_placeholder = st.sidebar.empty()


# Cloud Sync Indicator Badge
fb_connected = get_firebase_status()
if fb_connected:
    st.sidebar.markdown("""
    <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid #10b981; border-radius: 4px; padding: 6px 10px; font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: #34d399; margin-top: 8px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
        <span style="height: 8px; width: 8px; background-color: #10b981; border-radius: 50%; display: inline-block;"></span>
        CLOUD SYNC: CONNECTED (FIRESTORE)
    </div>
    """, unsafe_allow_html=True)
else:
    st.sidebar.markdown("""
    <div style="background: rgba(107, 114, 128, 0.12); border: 1px solid #6b7280; border-radius: 4px; padding: 6px 10px; font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: #9ca3af; margin-top: 8px; margin-bottom: 12px; display: flex; align-items: center; gap: 8px;">
        <span style="height: 8px; width: 8px; background-color: #6b7280; border-radius: 50%; display: inline-block;"></span>
        CLOUD SYNC: OFFLINE
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown("### ⚠️ INCIDENT LOG")

# Placeholders for Sidebar UI components (enables real-time updating during frame loop)
sidebar_counter_placeholder = st.sidebar.empty()
sidebar_actions_placeholder = st.sidebar.empty()
sidebar_log_placeholder = st.sidebar.empty()


def render_sidebar_ui():
    """Renders the reactive sidebar counter, action buttons, and thumbnail log list."""
    warning_count = len(st.session_state.warning_log)
    
    # 1. Update Counter Badge
    sidebar_counter_placeholder.markdown(f"""
    <div class="warning-counter-box">
        <div style="font-family: 'Oswald'; color: #8b949e; font-size: 0.85rem; letter-spacing: 1px;">TOTAL SESSION WARNINGS</div>
        <div class="warning-counter-num">{warning_count}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Render Action Buttons inside placeholder
    with sidebar_actions_placeholder.container():
        col_reset, col_report = st.columns(2)
        with col_reset:
            if st.button("Clear Log", use_container_width=True, key=f"clear_log_btn_{time.time()}"):
                st.session_state.warning_log = []
                st.session_state.last_logged_status = "SAFE"
                st.session_state.report_requested = False
                st.session_state.angle_history = []
                st.session_state.last_logged_reba_level = 0
                st.session_state.bend_timestamps = []
                st.session_state.last_voice_time = 0.0
                st.session_state.was_bent = False
                st.rerun()


        with col_report:
            if st.button("Generate Report", use_container_width=True, key=f"gen_report_btn_{time.time()}"):
                st.session_state.report_requested = True

    # 3. Render Incident Thumbnail List
    with sidebar_log_placeholder.container():
        if not st.session_state.warning_log:
            st.markdown("<div style='color: #8b949e; font-size: 0.85rem; text-align: center;'>No safety incidents logged.</div>", unsafe_allow_html=True)
        else:
            for entry in st.session_state.warning_log:
                status_code = entry["status"]
                if status_code == "POSTURE_WARNING":
                    card_class = "posture"
                    status_text = "⚠️ POSTURE WARNING (5s+)"
                elif status_code == "POSTURE_HIGH_RISK":
                    card_class = "high-risk"
                    status_text = "🚨 HIGH RISK POSTURE (15s+)"
                elif status_code == "RESTRICTED_ZONE_ENTRY":
                    card_class = "zone-breach"
                    status_text = "🚨 RESTRICTED ZONE BREACH"
                elif status_code == "REPETITIVE_MOTION_RISK":
                    card_class = "high-risk"
                    status_text = "🔁 REPETITIVE MOTION RISK (10+ cycles)"
                elif status_code == "REBA_MEDIUM_RISK":
                    card_class = "high-risk"
                    status_text = "⚠️ REBA MEDIUM RISK (Level 2)"
                elif status_code == "REBA_HIGH_RISK":
                    card_class = "zone-breach"
                    status_text = "🚨 REBA HIGH RISK (Level 3+)"
                else:
                    card_class = "posture"
                    status_text = status_code

                st.markdown(f"""
                <div class="warning-log-card {card_class}">
                    <div class="warning-log-time">⏱️ {entry['time']}</div>
                    <div class="warning-log-title">{status_text}</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(entry["thumbnail"], caption=f"Incident #{entry['id']}", use_container_width=True)



# Render Initial Sidebar View
render_sidebar_ui()

# Main Layout Columns (Stream on left, Instrument Panel on right)
col_stream, col_metrics = st.columns([2.2, 1])

with col_stream:
    frame_placeholder = st.empty()

with col_metrics:
    status_placeholder = st.empty()
    reba_card_placeholder = st.empty()
    angle_gauge_placeholder = st.empty()
    cycle_tracker_placeholder = st.empty()
    zone_status_placeholder = st.empty()



report_container = st.container()

def create_compressed_thumbnail(frame_img, quality=70):
    """
    Downscales frame to (160, 120) and encodes snapshot with JPEG quality 70 
    to minimize RAM footprint and prevent UI redraw bottlenecks.
    """
    resized = cv2.resize(frame_img, (160, 120), interpolation=cv2.INTER_LINEAR)
    success, encoded = cv2.imencode('.jpg', resized, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if success:
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


# Frame Streaming Loop
if run_camera:
    detector = PoseDetector()
    safety_detector = SafetyDetector()
    
    cap = cv2.VideoCapture(camera_index)
    # OpenCV Hardware & Buffer Performance Optimizations
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    if not cap.isOpened():
        col_stream.markdown(f"""
        <div class="error-card">
            <h4>⚠️ CAMERA ACCESS FAILURE</h4>
            <p>Unable to initialize video capture on device index {camera_index}.</p>
            <p>Please verify webcam device connections, drivers, or select another device index in the sidebar.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        try:
            frame_count = 0
            prev_time = time.time()
            while cap.isOpened() and run_camera:
                # Fast grab to discard stale hardware buffer frames
                if not cap.grab():
                    col_stream.markdown("""
                    <div class="error-card">
                        <h4>⚠️ STREAM INTERRUPTED</h4>
                        <p>Failed to capture frame from video device.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    break

                frame_count += 1

                # Frame-skipping: Process detection & UI rendering on every 3rd frame ONLY
                if frame_count % 3 != 0:
                    continue

                ret, frame = cap.retrieve()
                if not ret or frame is None:
                    continue

                # Calculate Live Processing FPS
                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0.0
                prev_time = curr_time

                fps_placeholder.markdown(f"""
                <div style="background: rgba(22, 27, 34, 0.8); border: 1px solid #30363d; border-radius: 4px; padding: 6px 10px; font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; color: #8b949e; margin-top: 6px; margin-bottom: 8px;">
                    ⚡ PROCESSING SPEED: <span style="font-size: 0.95rem; font-weight: 700; color: #ffcc00;">{fps:.1f} FPS</span>
                </div>
                """, unsafe_allow_html=True)

                # Resolution & Pre-Processing Downscaling to 640x480
                fh, fw = frame.shape[:2]
                if fw > 640 or fh > 480:
                    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)

                # Process frame with PoseDetector
                annotated_frame, landmarks = detector.process_frame(frame)
                h, w, _ = annotated_frame.shape

                # Draw restricted zone overlay
                annotated_frame = safety_detector.draw_restricted_zone(annotated_frame)

                # Determine Safety Status & Calculate Back Angle
                current_angle = None
                rep_status = None
                rep_cycles = 0

                if landmarks is None:
                    base_status = "NO_PERSON"
                else:
                    base_status = safety_detector.detect(landmarks, w, h)
                    
                    # Calculate Back Angle using landmarks
                    l = landmarks.landmark
                    shoulder = [(l[LEFT_SHOULDER].x + l[RIGHT_SHOULDER].x)/2 * w, (l[LEFT_SHOULDER].y + l[RIGHT_SHOULDER].y)/2 * h]
                    hip = [(l[LEFT_HIP].x + l[RIGHT_HIP].x)/2 * w, (l[LEFT_HIP].y + l[RIGHT_HIP].y)/2 * h]
                    knee = [(l[LEFT_KNEE].x + l[RIGHT_KNEE].x)/2 * w, (l[LEFT_KNEE].y + l[RIGHT_KNEE].y)/2 * h]
                    current_angle = calculate_angle(shoulder, hip, knee)

                # Track bend-overs for voice alert (>10 in 15s with 5s cooldown)
                if current_angle is not None:
                    now_ts = time.time()
                    if current_angle < BENDING_REPETITIVE_THRESHOLD:
                        if not st.session_state.was_bent:
                            st.session_state.was_bent = True
                            st.session_state.bend_timestamps.append(now_ts)
                    elif current_angle > STANDING_ANGLE_THRESHOLD:
                        st.session_state.was_bent = False

                    # Filter timestamps to keep only entries from the last 15 seconds
                    st.session_state.bend_timestamps = [
                        t for t in st.session_state.bend_timestamps if now_ts - t <= 15.0
                    ]

                    # Trigger non-blocking voice alert if > 10 bend-overs occur in last 15s
                    if len(st.session_state.bend_timestamps) > 10:
                        if now_ts - st.session_state.last_voice_time >= 5.0:
                            st.session_state.last_voice_time = now_ts
                            trigger_voice_alert_async("Warning! High frequency repetitive bending detected. Please take a break.")
                            st.toast("🚨 High Frequency Bending Alert!", icon="🔊")

                # Check Repetitive Motion Risk & REBA Ergonomic Risk Score

                rep_status, rep_cycles, st.session_state.angle_history = detect_repetitive_motion(
                    st.session_state.angle_history,
                    current_angle
                )

                reba_res = evaluate_reba_score(
                    landmarks,
                    w,
                    h,
                    is_repetitive_risk=(rep_status == "REPETITIVE_MOTION_RISK")
                )

                # Determine overall status with precedence
                if base_status in ["RESTRICTED_ZONE_ENTRY", "POSTURE_HIGH_RISK"]:
                    status = base_status
                elif rep_status == "REPETITIVE_MOTION_RISK":
                    status = "REPETITIVE_MOTION_RISK"
                else:
                    status = base_status

                # Log incident strictly on STATE TRANSITION to a new unsafe state
                unsafe_states = ["POSTURE_WARNING", "POSTURE_HIGH_RISK", "RESTRICTED_ZONE_ENTRY", "REPETITIVE_MOTION_RISK"]
                if status in unsafe_states and status != st.session_state.last_logged_status:
                    timestamp_str = datetime.now().strftime("%H:%M:%S")
                    
                    # Create JPEG-compressed thumbnail image
                    thumb_img = create_compressed_thumbnail(annotated_frame)
                    
                    st.session_state.warning_log.insert(0, {
                        "id": len(st.session_state.warning_log) + 1,
                        "time": timestamp_str,
                        "status": status,
                        "angle": round(current_angle, 1) if current_angle is not None else "N/A",
                        "thumbnail": thumb_img
                    })
                    st.session_state.last_logged_status = status

                    # Trigger visual toast alert
                    st.toast("⚠️ Unsafe Posture Detected!", icon="🚨")
                    
                    # Reactively refresh sidebar counter and log panel immediately
                    render_sidebar_ui()

                    # Additional persistent write to Firestore
                    log_incident_to_firestore({
                        "timestamp": datetime.now().isoformat(),
                        "incident_type": status,
                        "back_angle": round(current_angle, 1) if current_angle is not None else None,
                        "reba_scores": reba_res.to_dict() if reba_res is not None else None,
                        "session_id": st.session_state.session_id
                    })
                elif status == "SAFE":
                    st.session_state.last_logged_status = "SAFE"

                # REBA Incident Logging (when action level crosses into Medium level 2 or High level 3+)
                if reba_res is not None:
                    if reba_res.action_level >= 2 and st.session_state.last_logged_reba_level < 2:
                        timestamp_str = datetime.now().strftime("%H:%M:%S")
                        thumb_img = create_compressed_thumbnail(annotated_frame)
                        reba_status_code = "REBA_MEDIUM_RISK" if reba_res.action_level == 2 else "REBA_HIGH_RISK"

                        
                        st.session_state.warning_log.insert(0, {
                            "id": len(st.session_state.warning_log) + 1,
                            "time": timestamp_str,
                            "status": reba_status_code,
                            "angle": round(current_angle, 1) if current_angle is not None else "N/A",
                            "thumbnail": thumb_img
                        })
                        st.session_state.last_logged_reba_level = reba_res.action_level

                        # Trigger visual toast alert
                        st.toast("⚠️ Unsafe Posture Detected!", icon="🚨")

                        render_sidebar_ui()


                        # Additional persistent write to Firestore
                        log_incident_to_firestore({
                            "timestamp": datetime.now().isoformat(),
                            "incident_type": reba_status_code,
                            "back_angle": round(current_angle, 1) if current_angle is not None else None,
                            "reba_scores": reba_res.to_dict(),
                            "session_id": st.session_state.session_id
                        })
                    elif reba_res.action_level < 2:
                        st.session_state.last_logged_reba_level = reba_res.action_level


                # Convert BGR frame to RGB for Streamlit image container
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)

                # Render Status Banner UI
                if status == "SAFE":
                    status_placeholder.markdown("""
                    <div class="status-banner status-safe">
                        🟢 OPERATIONAL — SAFE
                    </div>
                    """, unsafe_allow_html=True)
                elif status == "POSTURE_WARNING":
                    status_placeholder.markdown("""
                    <div class="status-banner status-warning">
                        ⚠️ POSTURE WARNING (MODERATE RISK: 5s+)
                    </div>
                    """, unsafe_allow_html=True)
                elif status == "POSTURE_HIGH_RISK":
                    status_placeholder.markdown("""
                    <div class="status-banner status-high-risk">
                        🚨 POSTURE HIGH RISK (SEVERE: 15s+)
                    </div>
                    """, unsafe_allow_html=True)
                elif status == "REPETITIVE_MOTION_RISK":
                    status_placeholder.markdown("""
                    <div class="status-banner status-high-risk">
                        🚨 REPETITIVE MOTION RISK DETECTED
                    </div>
                    """, unsafe_allow_html=True)
                elif status == "RESTRICTED_ZONE_ENTRY":
                    status_placeholder.markdown("""
                    <div class="status-banner status-critical">
                        🚨 RESTRICTED ZONE BREACH
                    </div>
                    """, unsafe_allow_html=True)
                else:  # NO_PERSON
                    status_placeholder.markdown("""
                    <div class="status-banner status-standby">
                        👤 STANDBY — NO WORKER DETECTED
                    </div>
                    """, unsafe_allow_html=True)

                # Render REBA Ergonomic Assessment Panel
                if reba_res is None:
                    reba_card_placeholder.markdown("""
                    <div class="gauge-card" style="border-top-color: #6b7280;">
                        <div class="gauge-title">REBA ERGONOMIC ASSESSMENT</div>
                        <div class="gauge-value" style="color: #6b7280;">--</div>
                        <div class="gauge-meta">TRUNK: -- | NECK: -- | BONUS: --</div>
                        <div class="gauge-meta" style="color: #6b7280; font-weight: 600; margin-top: 6px;">ACTION LEVEL: STANDBY</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    if reba_res.action_level == 0:
                        level_color = "#10b981"  # Green
                    elif reba_res.action_level == 1:
                        level_color = "#eab308"  # Yellow
                    elif reba_res.action_level == 2:
                        level_color = "#ff6b00"  # Orange
                    else:
                        level_color = "#ef4444"  # Red

                    reba_card_placeholder.markdown(f"""
                    <div class="gauge-card" style="border-top-color: {level_color};">
                        <div class="gauge-title">REBA ERGONOMIC ASSESSMENT</div>
                        <div style="display: flex; align-items: baseline; justify-content: space-between;">
                            <div class="gauge-value" style="color: {level_color}; font-size: 2.8rem;">
                                {reba_res.combined_score} <span style="font-size: 1rem; color: #8b949e; font-weight: 500;">/ 12</span>
                            </div>
                            <div style="font-family: 'Oswald'; font-size: 1.1rem; color: {level_color}; font-weight: 700; text-transform: uppercase;">
                                LVL {reba_res.action_level}
                            </div>
                        </div>
                        <div style="font-family: 'IBM Plex Mono'; font-size: 0.8rem; color: #8b949e; margin-top: 2px;">
                            TRUNK: {reba_res.trunk_score} | NECK: {reba_res.neck_score} | BONUS: +{reba_res.repetitive_bonus}
                        </div>
                        <div style="font-family: 'IBM Plex Sans'; font-size: 0.82rem; font-weight: 600; color: {level_color}; margin-top: 6px; padding: 4px 8px; background: rgba(22, 27, 34, 0.8); border: 1px solid {level_color}; border-radius: 4px;">
                            ● {reba_res.action_label}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Render Industrial Gauge / Angle Instrument Readout
                angle_str = f"{current_angle:.1f}°" if current_angle is not None else "N/A"
                if current_angle is None:
                    angle_color = "#6b7280"
                elif current_angle >= BENDING_ANGLE_THRESHOLD:
                    angle_color = "#10b981"
                elif status == "POSTURE_HIGH_RISK":
                    angle_color = "#ef4444"
                else:
                    angle_color = "#ff6b00"
                
                progress_val = min(max(int((current_angle / 180.0) * 100), 0), 100) if current_angle is not None else 0

                angle_gauge_placeholder.markdown(f"""
                <div class="gauge-card">
                    <div class="gauge-title">TORSO / BACK ANGLE INSTRUMENT</div>
                    <div class="gauge-value" style="color: {angle_color};">{angle_str}</div>
                    <div class="gauge-meta">THRESHOLD: &lt;{BENDING_ANGLE_THRESHOLD:.0f}&deg; | WARN: &gt;{WARNING_DURATION_THRESHOLD:.0f}s | SEVERE: &gt;{HIGH_RISK_DURATION_THRESHOLD:.0f}s</div>
                    <div style="background: #21262d; border-radius: 4px; height: 8px; margin-top: 10px; overflow: hidden;">
                        <div style="background: {angle_color}; width: {progress_val}%; height: 100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Render Motion Cycle Tracker Stat Card
                cycle_color = "#ef4444" if rep_cycles >= REPETITIVE_CYCLE_THRESHOLD else ("#ff6b00" if rep_cycles >= 5 else "#10b981")
                cycle_progress = min(int((rep_cycles / REPETITIVE_CYCLE_THRESHOLD) * 100), 100)

                cycle_tracker_placeholder.markdown(f"""
                <div class="gauge-card" style="border-top-color: #ff6b00;">
                    <div class="gauge-title">MOTION CYCLE TRACKER</div>
                    <div class="gauge-value" style="color: {cycle_color};">{rep_cycles} / {REPETITIVE_CYCLE_THRESHOLD} <span style="font-size: 1.1rem; color: #8b949e; font-weight: 500;">cycles</span></div>
                    <div class="gauge-meta">ROLLING WINDOW: {REPETITIVE_WINDOW_SECONDS:.0f}s | RISK THRESHOLD: &ge;{REPETITIVE_CYCLE_THRESHOLD} CYCLES</div>
                    <div style="background: #21262d; border-radius: 4px; height: 8px; margin-top: 10px; overflow: hidden;">
                        <div style="background: {cycle_color}; width: {cycle_progress}%; height: 100%;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Zone Protection Readout Card
                zone_status_placeholder.markdown(f"""
                <div class="gauge-card" style="border-top-color: #ef4444;">
                    <div class="gauge-title">RESTRICTED ZONE COORDINATES</div>
                    <div style="font-family: 'IBM Plex Mono'; font-size: 1rem; color: #f0f6fc;">
                        X: [{RESTRICTED_ZONE[0]}, {RESTRICTED_ZONE[2]}] | Y: [{RESTRICTED_ZONE[1]}, {RESTRICTED_ZONE[3]}]
                    </div>
                    <div class="gauge-meta">STATUS: {"BREACH DETECTED" if status == "RESTRICTED_ZONE_ENTRY" else "CLEAR"}</div>
                </div>
                """, unsafe_allow_html=True)

        finally:
            cap.release()
else:
    col_stream.markdown("""
    <div style="background: #161b22; border: 2px dashed #30363d; border-radius: 8px; padding: 60px; text-align: center; color: #8b949e;">
        <h3 style="font-family: 'Oswald'; color: #ffcc00; letter-spacing: 1px;">SYSTEM STANDBY</h3>
        <p>Toggle "Start Camera Stream" in the sidebar controls to activate real-time monitoring.</p>
    </div>
    """, unsafe_allow_html=True)
    fps_placeholder.empty()
    status_placeholder.empty()
    reba_card_placeholder.empty()
    angle_gauge_placeholder.empty()
    cycle_tracker_placeholder.empty()
    zone_status_placeholder.empty()




# Render Safety Incident Report section when warning count >= 3 or manually requested
show_report = len(st.session_state.warning_log) >= 3 or st.session_state.report_requested

if show_report:
    with report_container:
        st.markdown("---")
        st.markdown(f"""
        <div style="background: #161b22; border: 2px solid #ffcc00; border-radius: 6px; padding: 20px; margin-top: 20px;">
            <h2 style="font-family: 'Oswald'; color: #ffcc00; letter-spacing: 1.5px; text-transform: uppercase; margin-bottom: 10px;">
                📄 SAFETY INCIDENT REPORT
            </h2>
            <div style="font-family: 'IBM Plex Mono'; font-size: 0.85rem; color: #8b949e; margin-bottom: 20px;">
                GENERATED AT: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | TOTAL INCIDENTS RECORDED: {len(st.session_state.warning_log)}
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.warning_log:
            cols = st.columns(3)
            for idx, entry in enumerate(st.session_state.warning_log):
                with cols[idx % 3]:
                    st.markdown(f"""
                    <div style="background: #0d1117; border: 1px solid #30363d; border-radius: 6px; padding: 12px; margin-top: 15px;">
                        <div style="font-family: 'IBM Plex Mono'; font-size: 0.8rem; color: #ffcc00;">INCIDENT #{entry['id']} — {entry['time']}</div>
                        <div style="font-family: 'Oswald'; font-size: 1.1rem; color: #f0f6fc; margin-top: 4px;">{entry['status']}</div>
                        <div style="font-family: 'IBM Plex Mono'; font-size: 0.8rem; color: #8b949e;">Back Angle: {entry['angle']}°</div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.image(entry["thumbnail"], use_container_width=True)
