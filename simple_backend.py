from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import base64
import time
import threading
import json
import os
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000", "http://localhost:3002"], supports_credentials=True)

def _print_registered_routes():
    """Debug helper: print all registered routes and their methods at startup."""
    try:
        print("🧭 Registered Flask routes:")
        for rule in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
            methods = ','.join(sorted(m for m in rule.methods if m not in {"HEAD","OPTIONS"}))
            print(f"   • {rule.rule}  ->  {methods}")
    except Exception as e:
        print(f"⚠️ Failed to list routes: {e}")

# Upload configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
video_cap = None
yolo_model = None
current_model_name = 'yolov8l'  # default model key
current_frame_data = None
current_detections = []
is_playing = False
frame_number = 0
total_frames = 0
current_video_path = None
original_frame_width = 0
original_frame_height = 0
last_processed_frame_width = 0
last_processed_frame_height = 0
process_target_width = 960  # default downscale width; set to 0 to keep original

# Detection configuration (runtime adjustable)
confidence_threshold = 0.3
iou_threshold = 0.45
max_detections = 300
inference_image_size = 640  # imgsz
enabled_classes = {'car', 'truck', 'bus', 'motorcycle'}  # filter classes

# Vehicle counting variables  
total_vehicle_count = 0
vehicle_counts_by_type = {
    'car': 0,
    'truck': 0,
    'bus': 0,
    'motorcycle': 0
}
# Direction-specific crossing counters
upward_crossings = 0
downward_crossings = 0
upward_counts_by_type = {
    'car': 0,
    'truck': 0,
    'bus': 0,
    'motorcycle': 0
}
downward_counts_by_type = {
    'car': 0,
    'truck': 0,
    'bus': 0,
    'motorcycle': 0
}
detection_history = []  # Store last 10 frames for trending
vehicles_per_minute = 0
session_start_time = time.time()

# Line-crossing detection variables
# Store both pixel Y (for drawing) and ratio (for resolution-independent adjustment)
COUNTING_LINE_Y = 250  # Initial pixel line (legacy default)
COUNTING_LINE_RATIO = COUNTING_LINE_Y / 600  # Assumes prior 800x600 processing size; will adapt dynamically
vehicle_tracks = {}  # Store vehicle tracking data
track_id_counter = 0
crossed_vehicles = set()  # Track which vehicles have crossed the line

def reset_counts_state():
    """Internal helper to reset counting-related state without returning a Flask response."""
    global total_vehicle_count, vehicle_counts_by_type, detection_history, session_start_time
    global vehicle_tracks, track_id_counter, crossed_vehicles
    global upward_crossings, downward_crossings, upward_counts_by_type, downward_counts_by_type
    total_vehicle_count = 0
    vehicle_counts_by_type = {
        'car': 0,
        'truck': 0,
        'bus': 0,
        'motorcycle': 0
    }
    detection_history = []
    session_start_time = time.time()
    vehicle_tracks = {}
    track_id_counter = 0
    crossed_vehicles = set()
    upward_crossings = 0
    downward_crossings = 0
    upward_counts_by_type = {k: 0 for k in upward_counts_by_type.keys()}
    downward_counts_by_type = {k: 0 for k in downward_counts_by_type.keys()}
    print("🔄 Internal count state reset")

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_video_info(video_path):
    """Get basic video information"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    return {
        'frame_count': frame_count,
        'fps': fps,
        'width': width,
        'height': height,
        'duration': duration
    }

def resolve_model_path(model_key: str):
    """Return filesystem path for a given model key; raise if unavailable."""
    base_dir = '/Users/bawantharathnayake/Desktop/Academic/semester 7/vision/YOLOv8-Traffic-Counter-main'
    mapping = {
        'yolov8l': os.path.join(base_dir, 'yolov8l.pt'),
        'yolov8n': os.path.join(base_dir, 'yolov8n.pt')
    }
    if model_key not in mapping:
        raise ValueError(f'Unsupported model: {model_key}')
    path = mapping[model_key]
    if not os.path.exists(path):
        raise FileNotFoundError(f'Model weights not found: {path}')
    return path

def load_model(model_key: str):
    """Load (or reload) YOLO model by key."""
    global yolo_model, current_model_name
    path = resolve_model_path(model_key)
    yolo_model = YOLO(path)
    current_model_name = model_key
    print(f"✅ YOLO model loaded: {model_key} -> {os.path.basename(path)}")

def initialize_video_and_model(video_path=None):
    """Initialize video capture and YOLO model"""
    global video_cap, yolo_model, total_frames, current_video_path, original_frame_width, original_frame_height
    
    # Initialize YOLO model if not already loaded
    if yolo_model is None:
        load_model(current_model_name)
    
    # Use default video if no path provided
    if video_path is None:
        video_path = '/Users/bawantharathnayake/Desktop/Academic/semester 7/vision/YOLOv8-Traffic-Counter-main/Highway_Video.mp4'
    
    # Close existing video if open
    if video_cap is not None:
        video_cap.release()
    
    # Initialize video capture
    current_video_path = video_path
    video_cap = cv2.VideoCapture(video_path)
    total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"✅ Video loaded: {total_frames} frames from {os.path.basename(video_path)}")
    print(f"📐 Original video resolution: {original_frame_width}x{original_frame_height}")

def calculate_distance(box1, box2):
    """Calculate distance between two bounding box centers"""
    x1, y1 = box1['x'] + box1['width'] / 2, box1['y'] + box1['height'] / 2
    x2, y2 = box2['x'] + box2['width'] / 2, box2['y'] + box2['height'] / 2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

def assign_track_ids(detections):
    """Assign track IDs to detections based on proximity"""
    global vehicle_tracks, track_id_counter
    
    current_frame_tracks = {}
    used_track_ids = set()
    
    for detection in detections:
        bbox = detection['bbox']
        center_x = bbox['x'] + bbox['width'] / 2
        center_y = bbox['y'] + bbox['height'] / 2
        
        # Find closest existing track that hasn't been used in this frame
        min_distance = float('inf')
        assigned_track_id = None
        
        for track_id, track_data in vehicle_tracks.items():
            if track_id in used_track_ids:
                continue  # Skip already used tracks
                
            if track_data['type'] == detection['type']:  # Same vehicle type
                distance = calculate_distance(bbox, track_data['last_bbox'])
                if distance < min_distance and distance < 80:  # Increased distance threshold
                    min_distance = distance
                    assigned_track_id = track_id
        
        # Assign track ID
        if assigned_track_id is None:
            # Create new track
            track_id_counter += 1
            assigned_track_id = track_id_counter
        
        # Mark this track as used
        used_track_ids.add(assigned_track_id)
        
        # Update track data
        current_frame_tracks[assigned_track_id] = {
            'type': detection['type'],
            'center_x': center_x,
            'center_y': center_y,
            'last_bbox': bbox,
            'last_seen': frame_number,
            'prev_center_y': vehicle_tracks.get(assigned_track_id, {}).get('center_y', center_y)
        }
        
        # Add track_id to detection
        detection['track_id'] = assigned_track_id
    
    # Update global tracks
    vehicle_tracks = current_frame_tracks
    
    return detections

def check_line_crossings(detections):
    """Check if any vehicles crossed the counting line"""
    global total_vehicle_count, vehicle_counts_by_type, crossed_vehicles
    global upward_crossings, downward_crossings, upward_counts_by_type, downward_counts_by_type
    
    new_crossings = 0
    
    for detection in detections:
        track_id = detection.get('track_id')
        if track_id is None:
            continue
            
        center_y = detection['bbox']['y'] + detection['bbox']['height'] / 2
        
        # Check if vehicle crossed the horizontal line
        if track_id in vehicle_tracks:
            prev_y = vehicle_tracks[track_id]['prev_center_y']
            
            line_crossed = False
            
            # Horizontal line crossing detection: was above line, now below line (downward traffic)
            if prev_y < COUNTING_LINE_Y and center_y >= COUNTING_LINE_Y:
                line_crossed = True
                print(f"🚗 Vehicle crossed line (downward)! Track ID: {track_id}")
            
            # Horizontal line crossing detection: was below line, now above line (upward traffic)  
            elif prev_y > COUNTING_LINE_Y and center_y <= COUNTING_LINE_Y:
                line_crossed = True
                print(f"🚗 Vehicle crossed line (upward)! Track ID: {track_id}")
            
            if line_crossed and track_id not in crossed_vehicles:
                # Vehicle crossed the line for the first time
                crossed_vehicles.add(track_id)
                vehicle_type = detection['type']
                
                # Increment counters
                if vehicle_type in vehicle_counts_by_type:
                    vehicle_counts_by_type[vehicle_type] += 1
                    total_vehicle_count += 1
                    new_crossings += 1

                    # Direction specific counters
                    if prev_y < COUNTING_LINE_Y and center_y >= COUNTING_LINE_Y:
                        downward_crossings += 1
                        if vehicle_type in downward_counts_by_type:
                            downward_counts_by_type[vehicle_type] += 1
                    elif prev_y > COUNTING_LINE_Y and center_y <= COUNTING_LINE_Y:
                        upward_crossings += 1
                        if vehicle_type in upward_counts_by_type:
                            upward_counts_by_type[vehicle_type] += 1
                    
                    print(f"✅ Vehicle counted! Track ID: {track_id}, Type: {vehicle_type}, Total: {total_vehicle_count}")
            
            # Previous positions are updated in assign_track_ids
    
    return new_crossings

def update_vehicle_counts(detections):
    """Update vehicle counting with line-crossing detection"""
    global detection_history, vehicles_per_minute
    
    # Assign track IDs
    detections = assign_track_ids(detections)
    
    # Check for line crossings
    new_crossings = check_line_crossings(detections)
    
    # Count current frame detections by type
    frame_counts = {'car': 0, 'truck': 0, 'bus': 0, 'motorcycle': 0}
    for detection in detections:
        vehicle_type = detection['type']
        if vehicle_type in frame_counts:
            frame_counts[vehicle_type] += 1
    
    # Store detection history for trending (last 10 frames)
    detection_history.append({
        'frame': frame_number,
        'timestamp': time.time(),
        'detections': len(detections),
        'by_type': frame_counts.copy(),
        'new_crossings': new_crossings
    })
    
    # Keep only last 10 frames
    if len(detection_history) > 10:
        detection_history.pop(0)
    
    # Calculate vehicles per minute based on crossings
    session_duration = time.time() - session_start_time
    if session_duration > 0:
        vehicles_per_minute = int((total_vehicle_count / session_duration) * 60)
    
    return detections

def process_frame():
    """Process one frame with YOLO detection"""
    global video_cap, yolo_model, current_frame_data, current_detections, frame_number, is_playing
    global last_processed_frame_width, last_processed_frame_height
    global COUNTING_LINE_Y, COUNTING_LINE_RATIO
    
    if not video_cap or not video_cap.isOpened():
        return False
    
    # Read frame
    ret, frame = video_cap.read()
    if not ret:
        # Reached end of video: stop playback instead of looping
        is_playing = False
        print("🏁 Video reached the end. Auto-stopping playback.")
        return False
    
    frame_number += 1
    
    # Optional resize for performance if process_target_width > 0
    h, w = frame.shape[:2]
    if process_target_width and process_target_width > 0 and w > process_target_width:
        scale = process_target_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        frame = cv2.resize(frame, (new_w, new_h))
    # Track processed frame size
    last_processed_frame_height, last_processed_frame_width = frame.shape[0], frame.shape[1]
    # Recompute pixel Y from ratio (keeps line consistent across resizes)
    if last_processed_frame_height > 0:
        COUNTING_LINE_Y = int(COUNTING_LINE_RATIO * last_processed_frame_height)
    
    # YOLO detection
    results = yolo_model(frame, verbose=False, conf=confidence_threshold, iou=iou_threshold, imgsz=inference_image_size, max_det=max_detections)
    detections = []
    
    # Process detections
    for result in results:
        if result.boxes is not None:
            boxes = result.boxes
            for i, box in enumerate(boxes):
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = box.cls[0].cpu().numpy()
                
                # Filter by confidence
                if confidence >= confidence_threshold:
                    # Get class name
                    class_name = yolo_model.names[int(class_id)]
                    # Filter classes
                    if class_name in enabled_classes:
                        detections.append({
                            'bbox': {
                                'x': int(x1),
                                'y': int(y1),
                                'width': int(x2 - x1),
                                'height': int(y2 - y1)
                            },
                            'confidence': float(confidence),
                            'type': class_name,
                            'id': i
                        })
                        
    # Update vehicle counting (includes track ID assignment)
    detections = update_vehicle_counts(detections)
    
    # Draw detections with track IDs
    for detection in detections:
        bbox = detection['bbox']
        x1, y1 = bbox['x'], bbox['y']
        x2, y2 = x1 + bbox['width'], y1 + bbox['height']
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw track ID and vehicle info
        track_id = detection.get('track_id', '?')
        label = f'{detection["type"]} #{track_id} {detection["confidence"]:.2f}'
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        
        # Draw center point
        center_x = x1 + bbox['width'] // 2
        center_y = y1 + bbox['height'] // 2
        cv2.circle(frame, (center_x, center_y), 3, (255, 255, 0), -1)
    
    # Draw counting line
    cv2.line(frame, (0, COUNTING_LINE_Y), (frame.shape[1], COUNTING_LINE_Y), (0, 0, 255), 3)
    cv2.putText(frame, 'COUNTING LINE', (10, COUNTING_LINE_Y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Draw counter on frame
    cv2.putText(frame, f'Vehicles Crossed: {total_vehicle_count}', (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Convert frame to base64
    _, buffer = cv2.imencode('.jpg', frame)
    frame_base64 = base64.b64encode(buffer).decode('utf-8')
    current_frame_data = f"data:image/jpeg;base64,{frame_base64}"
    current_detections = detections
    
    print(f"📹 Processed frame {frame_number} with {len(detections)} detections (Total: {total_vehicle_count})")
    return True

def video_processing_loop():
    """Main video processing loop"""
    global is_playing
    
    while True:
        if is_playing:
            success = process_frame()
            if not success:
                print("❌ Failed to process frame")
                time.sleep(0.1)
        else:
            time.sleep(0.1)  # Sleep when not playing
        
        time.sleep(1/15)  # Target 15 FPS

@app.route('/api/status')
def get_status():
    """Get current status"""
    # Recompute direction totals from per-type dictionaries for consistency
    computed_upward = sum(upward_counts_by_type.values())
    computed_downward = sum(downward_counts_by_type.values())
    # Prefer last processed frame size; fall back to original video resolution
    res_w = last_processed_frame_width or original_frame_width
    res_h = last_processed_frame_height or original_frame_height
    return jsonify({
        'status': 'online',
        'model_loaded': yolo_model is not None,
    'model_name': current_model_name,
        'video_loaded': video_cap is not None and video_cap.isOpened(),
        'is_playing': is_playing,
        'frame_number': frame_number,
        'total_frames': total_frames,
        'current_detections': len(current_detections),
        'fps': 25,
        'resolution': f'{res_w}x{res_h}' if res_w and res_h else None,
    'frame_width': res_w,
    'frame_height': res_h,
    'processing_target_width': process_target_width,
        'original_frame_width': original_frame_width,
        'original_frame_height': original_frame_height,
        'total_count': total_vehicle_count,
        'vehicle_counts': vehicle_counts_by_type,
        'vehicles_per_minute': vehicles_per_minute,
    'upward_count': computed_upward,
    'downward_count': computed_downward,
    'upward_counts_by_type': upward_counts_by_type,
    'downward_counts_by_type': downward_counts_by_type,
        'session_duration': int(time.time() - session_start_time),
    'counting_line_y': COUNTING_LINE_Y,
    'counting_line_ratio': COUNTING_LINE_RATIO,
    'detection_config': {
        'confidence_threshold': confidence_threshold,
        'iou_threshold': iou_threshold,
        'max_detections': max_detections,
        'inference_image_size': inference_image_size,
        'enabled_classes': list(enabled_classes)
    }
    })

@app.route('/api/current_frame')
def get_current_frame():
    """Get current frame with detections"""
    if current_frame_data:
        computed_upward = sum(upward_counts_by_type.values())
        computed_downward = sum(downward_counts_by_type.values())
        res_w = last_processed_frame_width or original_frame_width
        res_h = last_processed_frame_height or original_frame_height
        return jsonify({
            'frame': current_frame_data,
            'detections': current_detections,
            'total_count': total_vehicle_count,
            'current_frame_detections': len(current_detections),
            'vehicle_counts': vehicle_counts_by_type,
            'vehicles_per_minute': vehicles_per_minute,
            'frame_number': frame_number,
            'total_frames': total_frames,
            'timestamp': time.time(),
            'is_playing': is_playing,
            'detection_history': detection_history[-5:] if detection_history else [],  # Last 5 frames
            'upward_count': computed_upward,
            'downward_count': computed_downward,
            'upward_counts_by_type': upward_counts_by_type,
            'downward_counts_by_type': downward_counts_by_type,
            'frame_width': res_w,
            'frame_height': res_h,
            'processing_target_width': process_target_width,
            'original_frame_width': original_frame_width,
            'original_frame_height': original_frame_height,
            'counting_line_y': COUNTING_LINE_Y,
            'counting_line_ratio': COUNTING_LINE_RATIO,
            'model_name': current_model_name
        })
    else:
        computed_upward = sum(upward_counts_by_type.values())
        computed_downward = sum(downward_counts_by_type.values())
        res_w = last_processed_frame_width or original_frame_width
        res_h = last_processed_frame_height or original_frame_height
        return jsonify({
            'frame': None,
            'detections': [],
            'total_count': total_vehicle_count,
            'current_frame_detections': 0,
            'vehicle_counts': vehicle_counts_by_type,
            'vehicles_per_minute': vehicles_per_minute,
            'frame_number': frame_number,
            'total_frames': total_frames,
            'timestamp': time.time(),
            'is_playing': is_playing,
            'detection_history': detection_history[-5:] if detection_history else [],
            'upward_count': computed_upward,
            'downward_count': computed_downward,
            'upward_counts_by_type': upward_counts_by_type,
            'downward_counts_by_type': downward_counts_by_type,
            'frame_width': res_w,
            'frame_height': res_h,
            'processing_target_width': process_target_width,
            'original_frame_width': original_frame_width,
            'original_frame_height': original_frame_height,
            'counting_line_y': COUNTING_LINE_Y,
            'counting_line_ratio': COUNTING_LINE_RATIO,
            'model_name': current_model_name
        })

@app.route('/api/start_video', methods=['POST'])
def start_video():
    """Start video playback"""
    global is_playing
    is_playing = True
    print("🎬 Video started")
    return jsonify({'status': 'started', 'is_playing': True})

@app.route('/api/stop_video', methods=['POST'])
def stop_video():
    """Stop video playback"""
    global is_playing
    is_playing = False
    print("🛑 Video stopped")
    return jsonify({'status': 'stopped', 'is_playing': False})

@app.route('/api/restart_video', methods=['POST'])
def restart_video():
    """Restart current video: stop playback, reset counts, seek to beginning, and start again."""
    global is_playing, video_cap, current_video_path, frame_number
    try:
        # Stop if running
        was_playing = is_playing
        is_playing = False
        # Small pause to let processing loop observe state
        time.sleep(0.2)

        # Reset counts/state
        reset_counts_state()

        # Re-open current video (or default if none)
        reopen_path = current_video_path
        if reopen_path is None:
            reopen_path = '/Users/bawantharathnayake/Desktop/Academic/semester 7/vision/YOLOv8-Traffic-Counter-main/Highway_Video.mp4'
        if video_cap is not None:
            video_cap.release()
        video_cap = cv2.VideoCapture(reopen_path)
        frame_number = 0
        print(f"🔁 Video restarted from beginning: {os.path.basename(reopen_path)}")

        # Auto-start playback unless explicitly stopped
        auto_start = True
        data = None
        if request.is_json:
            data = request.get_json(silent=True) or {}
            auto_start = data.get('auto_start', True)
        if auto_start:
            is_playing = True

        return jsonify({
            'status': 'restarted',
            'is_playing': is_playing,
            'auto_started': auto_start,
            'video': os.path.basename(reopen_path)
        })
    except Exception as e:
        print(f"❌ Restart error: {str(e)}")
        return jsonify({'error': f'Failed to restart video: {str(e)}'}), 500

@app.route('/api/reset_counts', methods=['POST'])
def reset_counts():
    """Reset all vehicle counts"""
    global total_vehicle_count, vehicle_counts_by_type, detection_history, session_start_time
    global vehicle_tracks, track_id_counter, crossed_vehicles
    global upward_crossings, downward_crossings, upward_counts_by_type, downward_counts_by_type
    
    total_vehicle_count = 0
    vehicle_counts_by_type = {
        'car': 0,
        'truck': 0,
        'bus': 0,
        'motorcycle': 0
    }
    detection_history = []
    session_start_time = time.time()
    
    # Reset line-crossing tracking
    vehicle_tracks = {}
    track_id_counter = 0
    crossed_vehicles = set()
    upward_crossings = 0
    downward_crossings = 0
    upward_counts_by_type = {k: 0 for k in upward_counts_by_type.keys()}
    downward_counts_by_type = {k: 0 for k in downward_counts_by_type.keys()}
    
    print("🔄 Vehicle counts and tracking reset")
    return jsonify({
        'status': 'reset',
        'total_count': total_vehicle_count,
    'vehicle_counts': vehicle_counts_by_type,
    'upward_count': upward_crossings,
    'downward_count': downward_crossings,
    'upward_counts_by_type': upward_counts_by_type,
    'downward_counts_by_type': downward_counts_by_type
    })

@app.route('/api/vehicle_stats')
def get_vehicle_stats():
    """Get detailed vehicle statistics"""
    computed_upward = sum(upward_counts_by_type.values())
    computed_downward = sum(downward_counts_by_type.values())
    return jsonify({
        'total_count': total_vehicle_count,
        'vehicle_counts': vehicle_counts_by_type,
        'vehicles_per_minute': vehicles_per_minute,
        'session_duration': int(time.time() - session_start_time),
        'detection_history': detection_history,
        'average_detections_per_frame': sum([h['detections'] for h in detection_history]) / len(detection_history) if detection_history else 0,
    'peak_detections': max([h['detections'] for h in detection_history]) if detection_history else 0,
    'upward_count': computed_upward,
    'downward_count': computed_downward,
    'upward_counts_by_type': upward_counts_by_type,
    'downward_counts_by_type': downward_counts_by_type,
    'counting_line_y': COUNTING_LINE_Y,
    'counting_line_ratio': COUNTING_LINE_RATIO
    })

@app.route('/api/set_processing_size', methods=['POST'])
def set_processing_size():
    """Adjust processing resize target. Body: {"width": <int>} or {"mode":"original"} to disable resize."""
    global process_target_width
    try:
        if not request.is_json:
            return jsonify({'error': 'JSON body required'}), 400
        data = request.get_json(silent=True) or {}
        if 'mode' in data and data['mode'] == 'original':
            process_target_width = 0
        elif 'width' in data:
            try:
                w = int(data['width'])
            except (TypeError, ValueError):
                return jsonify({'error': 'Invalid width'}), 400
            if w < 320:
                return jsonify({'error': 'Width too small (min 320)'}), 400
            process_target_width = w
        else:
            return jsonify({'error': 'Provide "mode":"original" or numeric "width"'}), 400
        return jsonify({'status': 'updated', 'processing_target_width': process_target_width})
    except Exception as e:
        return jsonify({'error': f'Failed to set processing size: {str(e)}'}), 500

@app.route('/api/set_detection_settings', methods=['POST'])
def set_detection_settings():
    """Update detection parameters: confidence, iou, max_detections, imgsz, enabled_classes.
    Body JSON any subset: {confidence:0.5, iou:0.45, max_detections:200, imgsz:640, classes:['car','bus']}"""
    global confidence_threshold, iou_threshold, max_detections, inference_image_size, enabled_classes
    try:
        if not request.is_json:
            return jsonify({'error': 'JSON body required'}), 400
        data = request.get_json(silent=True) or {}
        if 'confidence' in data:
            v = float(data['confidence'])
            if not 0 < v <= 1: return jsonify({'error': 'confidence must be (0,1]'}), 400
            confidence_threshold = v
        if 'iou' in data:
            v = float(data['iou'])
            if not 0 < v <= 1: return jsonify({'error': 'iou must be (0,1]'}), 400
            iou_threshold = v
        if 'max_detections' in data:
            v = int(data['max_detections'])
            if v <= 0: return jsonify({'error': 'max_detections must be >0'}), 400
            max_detections = v
        if 'imgsz' in data:
            v = int(data['imgsz'])
            if v < 160 or v > 1536: return jsonify({'error': 'imgsz out of range (160-1536)'}), 400
            inference_image_size = v
        if 'classes' in data:
            if not isinstance(data['classes'], list):
                return jsonify({'error': 'classes must be list'}), 400
            allowed = {'car','truck','bus','motorcycle'}
            newset = set()
            for c in data['classes']:
                if c not in allowed:
                    return jsonify({'error': f'unsupported class: {c}'}), 400
                newset.add(c)
            if not newset:
                return jsonify({'error': 'At least one class must be enabled'}), 400
            enabled_classes = newset
        return jsonify({'status': 'updated', 'detection_config': {
            'confidence_threshold': confidence_threshold,
            'iou_threshold': iou_threshold,
            'max_detections': max_detections,
            'inference_image_size': inference_image_size,
            'enabled_classes': list(enabled_classes)
        }})
    except Exception as e:
        return jsonify({'error': f'Failed to set detection settings: {str(e)}'}), 500

@app.route('/api/set_model', methods=['POST'])
def set_model():
    """Switch YOLO model. Body: {model: 'yolov8n'|'yolov8l'}"""
    global is_playing
    try:
        if not request.is_json:
            return jsonify({'error': 'JSON body required'}), 400
        data = request.get_json(silent=True) or {}
        model_key = data.get('model')
        if model_key is None:
            return jsonify({'error': 'model key required'}), 400
        # Pause processing during switch
        was_playing = is_playing
        is_playing = False
        load_model(model_key)
        # After switching model we typically want to reset counts optionally
        if data.get('reset_counts', True):
            reset_counts_state()
        is_playing = was_playing and data.get('resume', True)
        return jsonify({'status': 'model_switched', 'model': current_model_name, 'resume': is_playing})
    except Exception as e:
        return jsonify({'error': f'Failed to switch model: {str(e)}'}), 500

@app.route('/api/detection_settings', methods=['GET'])
def get_detection_settings():
    """Return current detection configuration."""
    return jsonify({
        'model': current_model_name,
        'confidence_threshold': confidence_threshold,
        'iou_threshold': iou_threshold,
        'max_detections': max_detections,
        'inference_image_size': inference_image_size,
        'enabled_classes': list(enabled_classes)
    })

# Alias route (hyphen form) in case frontend or caching requests alternative pattern
@app.route('/api/detection-settings', methods=['GET'])
def get_detection_settings_alias():
    return get_detection_settings()

@app.route('/api/debug/routes', methods=['GET'])
def debug_routes():
    """Return all registered routes for debugging (no auth)."""
    routes = []
    for rule in sorted(app.url_map.iter_rules(), key=lambda r: r.rule):
        routes.append({
            'rule': rule.rule,
            'methods': sorted(m for m in rule.methods if m not in {"HEAD","OPTIONS"})
        })
    return jsonify({'routes': routes})

@app.route('/api/set_counting_line', methods=['POST'])
def set_counting_line():
    """Set counting line by ratio (0-1) or absolute pixel y of processed frame."""
    global COUNTING_LINE_RATIO, COUNTING_LINE_Y, last_processed_frame_height, original_frame_height
    try:
        if not request.is_json:
            return jsonify({'error': 'JSON body required'}), 400
        data = request.get_json(silent=True) or {}
        if 'ratio' in data:
            try:
                r = float(data['ratio'])
            except (TypeError, ValueError):
                return jsonify({'error': 'Invalid ratio value'}), 400
            if not 0 <= r <= 1:
                return jsonify({'error': 'ratio must be between 0 and 1'}), 400
            COUNTING_LINE_RATIO = r
        elif 'y' in data:
            try:
                y_val = int(data['y'])
            except (TypeError, ValueError):
                return jsonify({'error': 'Invalid y value'}), 400
            frame_h = last_processed_frame_height or original_frame_height
            if frame_h <= 0:
                return jsonify({'error': 'Frame height unknown yet – start video first'}), 400
            if not 0 <= y_val <= frame_h:
                return jsonify({'error': f'y must be between 0 and {frame_h}'}), 400
            COUNTING_LINE_RATIO = y_val / frame_h
        else:
            return jsonify({'error': 'Provide ratio (0-1) or y'}), 400
        # Update pixel position
        frame_h = last_processed_frame_height or original_frame_height
        if frame_h > 0:
            COUNTING_LINE_Y = int(COUNTING_LINE_RATIO * frame_h)
        return jsonify({
            'status': 'updated',
            'counting_line_ratio': COUNTING_LINE_RATIO,
            'counting_line_y': COUNTING_LINE_Y
        })
    except Exception as e:
        return jsonify({'error': f'Failed to set counting line: {str(e)}'}), 500

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    """Upload a new video file"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            # Generate unique filename
            file_extension = file.filename.rsplit('.', 1)[1].lower()
            unique_filename = f"{uuid.uuid4().hex}.{file_extension}"
            
            # Save file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            
            # Get video info
            video_info = get_video_info(file_path)
            if video_info is None:
                os.remove(file_path)
                return jsonify({'error': 'Invalid video file'}), 400
            
            print(f"📹 Video uploaded: {file.filename} -> {unique_filename}")
            print(f"📊 Video info: {video_info}")
            
            return jsonify({
                'message': 'Video uploaded successfully',
                'filename': unique_filename,
                'original_name': file.filename,
                'video_info': video_info
            })
        else:
            return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv, wmv, flv, webm'}), 400
            
    except Exception as e:
        print(f"❌ Upload error: {str(e)}")
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/api/videos', methods=['GET'])
def list_videos():
    """List all uploaded videos"""
    try:
        videos = []
        
        # Add default highway video
        default_video_path = '/Users/bawantharathnayake/Desktop/Academic/semester 7/vision/YOLOv8-Traffic-Counter-main/Highway_Video.mp4'
        if os.path.exists(default_video_path):
            video_info = get_video_info(default_video_path)
            videos.append({
                'filename': 'Highway_Video.mp4',
                'original_name': 'Highway_Video.mp4',
                'is_default': True,
                'is_current': current_video_path == default_video_path,
                'video_info': video_info,
                'upload_time': 'Default'
            })
        
        # Add uploaded videos
        if os.path.exists(UPLOAD_FOLDER):
            for filename in os.listdir(UPLOAD_FOLDER):
                if allowed_file(filename):
                    file_path = os.path.join(UPLOAD_FOLDER, filename)
                    video_info = get_video_info(file_path)
                    if video_info:
                        file_stat = os.stat(file_path)
                        videos.append({
                            'filename': filename,
                            'original_name': filename,
                            'is_default': False,
                            'is_current': current_video_path == file_path,
                            'video_info': video_info,
                            'upload_time': time.ctime(file_stat.st_ctime),
                            'file_size': file_stat.st_size
                        })
        
        return jsonify({'videos': videos})
        
    except Exception as e:
        print(f"❌ List videos error: {str(e)}")
        return jsonify({'error': f'Failed to list videos: {str(e)}'}), 500

@app.route('/api/switch_video', methods=['POST'])
def switch_video():
    """Switch to a different video"""
    global is_playing
    
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        # Stop current video
        is_playing = False
        time.sleep(0.5)  # Wait for processing to stop
        
        # Determine video path
        if filename == 'Highway_Video.mp4':
            video_path = '/Users/bawantharathnayake/Desktop/Academic/semester 7/vision/YOLOv8-Traffic-Counter-main/Highway_Video.mp4'
        else:
            video_path = os.path.join(UPLOAD_FOLDER, filename)
        
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 404
        
        # Reset counters
        reset_counts()
        
        # Initialize new video
        initialize_video_and_model(video_path)
        
        print(f"🔄 Switched to video: {filename}")
        
        return jsonify({
            'message': f'Switched to video: {filename}',
            'video_path': video_path,
            'current_video': os.path.basename(video_path)
        })
        
    except Exception as e:
        print(f"❌ Switch video error: {str(e)}")
        return jsonify({'error': f'Failed to switch video: {str(e)}'}), 500

@app.route('/api/delete_video', methods=['DELETE'])
def delete_video():
    """Delete an uploaded video"""
    try:
        data = request.get_json()
        filename = data.get('filename')
        
        if not filename:
            return jsonify({'error': 'No filename provided'}), 400
        
        if filename == 'Highway_Video.mp4':
            return jsonify({'error': 'Cannot delete default video'}), 400
        
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        if not os.path.exists(file_path):
            return jsonify({'error': 'Video file not found'}), 404
        
        # If this is the current video, switch to default
        if current_video_path == file_path:
            is_playing = False
            time.sleep(0.5)
            reset_counts()
            initialize_video_and_model()  # Switch to default
        
        # Delete file
        os.remove(file_path)
        print(f"🗑️ Deleted video: {filename}")
        
        return jsonify({'message': f'Video deleted: {filename}'})
        
    except Exception as e:
        print(f"❌ Delete video error: {str(e)}")
        return jsonify({'error': f'Failed to delete video: {str(e)}'}), 500

if __name__ == '__main__':
    print("🚀 Starting YOLOv8 Large Model Traffic Counter...")
    print("🎯 Using YOLOv8-Large for HIGH ACCURACY detection")
    
    # Initialize components
    initialize_video_and_model()
    
    # Start video processing thread
    processing_thread = threading.Thread(target=video_processing_loop, daemon=True)
    processing_thread.start()
    print("🎬 Video processing thread started")
    
    # Debug: list all routes so we can confirm detection_settings endpoint present
    _print_registered_routes()

    print("🌐 Server starting on http://localhost:5003")
    app.run(host='0.0.0.0', port=5003, debug=False)












