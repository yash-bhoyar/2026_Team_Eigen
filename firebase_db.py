"""
firebase_db.py
================
Firebase Firestore persistence module for SafeGuard AI.
Handles Firebase initialization using firebase-admin SDK and non-blocking
incident logging to Firestore.
"""

import os
from typing import Dict, Any, Tuple, Optional
import firebase_admin
from firebase_admin import credentials, firestore

# Path to Firebase service account key
SERVICE_ACCOUNT_KEY_PATH = "serviceAccountKey.json"

_db_client: Optional[firestore.Client] = None
_is_connected: bool = False


def init_firebase() -> Tuple[Optional[firestore.Client], bool]:
    """
    Initializes Firebase Admin SDK if serviceAccountKey.json exists.
    Returns (db_client, is_connected).
    Failures are handled gracefully without raising exceptions.
    """
    global _db_client, _is_connected

    if _is_connected and _db_client is not None:
        return _db_client, True

    try:
        if not os.path.exists(SERVICE_ACCOUNT_KEY_PATH):
            print(f"[Firebase] {SERVICE_ACCOUNT_KEY_PATH} not found. Running in OFFLINE mode.")
            _is_connected = False
            return None, False

        # Initialize Firebase Admin App if not already initialized
        if not firebase_admin._apps:
            cred = credentials.Certificate(SERVICE_ACCOUNT_KEY_PATH)
            firebase_admin.initialize_app(cred)

        _db_client = firestore.client()
        _is_connected = True
        print("[Firebase] Successfully connected to Firestore database.")
        return _db_client, True

    except Exception as e:
        print(f"[Firebase] Connection initialization failed: {e}")
        _db_client = None
        _is_connected = False
        return None, False


def get_firebase_status() -> bool:
    """Returns True if connected to Firestore, False otherwise."""
    global _is_connected
    if not _is_connected:
        init_firebase()
    return _is_connected


def log_incident_to_firestore(incident_data: Dict[str, Any]) -> bool:
    """
    Writes an incident record to Firestore collection 'incidents'.
    Wrapped strictly in try/except so connection/network errors never crash the caller.
    
    Args:
        incident_data: Dict containing {timestamp, incident_type, back_angle, reba_scores, session_id}
        
    Returns:
        bool: True if written successfully, False otherwise.
    """
    try:
        db, is_connected = init_firebase()
        if not is_connected or db is None:
            print("[Firebase] Firestore offline — skipping cloud record write.")
            return False

        # Add record to 'incidents' collection
        db.collection("incidents").add(incident_data)
        print(f"[Firebase] Incident successfully logged to Firestore: {incident_data.get('incident_type')}")
        return True

    except Exception as e:
        print(f"[Firebase] Failed to write incident to Firestore: {e}")
        return False
