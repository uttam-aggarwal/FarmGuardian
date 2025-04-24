import cv2
import time
import threading
import sqlite3
import json
import numpy as np
from collections import defaultdict
from ultralytics import YOLO

# Configuration
CONFIG = {
    "farm_id": "Farm_1",
    "cameras": {
        "cam_1": {
            "source": "sample1.mp4",
            "zone": "North Pasture",
            "active": True
        },
        "cam_2": {
            "source": "sample2.mp4",
            "zone": "South Barn",
            "active": True
        },
        "cam_3": {
            "source": "sample3.mp4",
            "zone": "east Barn",
            "active": True
        },
        "cam_4": {
            "source": "sample4.mp4",
            "zone": "west Barn",
            "active": True
        }
    },
    "model": {
        "name": "yolov8n.pt",
        "conf_threshold": 0.4,
        "classes": [14, 15, 16, 17, 18, 19]
    },
    "alert": {
        "consecutive_frames": 3,
        "cooldown": 50,
        "activity_window": 5
    },
    "system": {
        "grid_layout": (2, 2),
        "base_height": 300,
        "aspect_ratio": 16/9,
        "frame_skip": 1,
        "fps_display": True
    }
}

COCO_CLASS_NAMES = {
    14: "bird", 15: "cat", 16: "dog",
    17: "horse", 18: "sheep", 19: "cow"
}

class FarmMonitor:
    def __init__(self):
        self.model = YOLO(CONFIG["model"]["name"])
        self.lock = threading.Lock()
        self.animal_tracks = defaultdict(lambda: {
            'class': None, 'consecutive': 0, 'last_alert': 0,
            'last_seen': 0, 'positions': [], 'camera': None,
            'zone': None, 'trail': []
        })
        self.frames = {}
        self.db = sqlite3.connect('farm_logs.db', check_same_thread=False)
        self._init_db()

    def _init_db(self):
        with self.lock:
            cursor = self.db.cursor()
            cursor.execute('''CREATE TABLE IF NOT EXISTS alerts
                          (timestamp REAL, farm TEXT, cameras TEXT, 
                          zone TEXT, animals TEXT, counts TEXT,
                          positions TEXT, track_id TEXT)''')
            self.db.commit()

    def update_track(self, camera_id, detection):
        with self.lock:
            track_id = f"{camera_id}_{detection['track_id']}"
            current_time = time.time()
            
            # Update trail with new position
            trail = self.animal_tracks[track_id].get('trail', [])
            trail.append(detection['trail'][-1])  # Add latest position
            if len(trail) > 15:  # Keep only last 15 positions
                trail.pop(0)
                
            self.animal_tracks[track_id].update({
                'class': detection['class'],
                'consecutive': self.animal_tracks[track_id]['consecutive'] + 1,
                'last_seen': current_time,
                'positions': detection['position'],
                'camera': camera_id,
                'zone': CONFIG["cameras"][camera_id]["zone"],
                'trail': trail
            })

            if self._should_trigger_alert(track_id, current_time):
                self._trigger_alert(track_id, current_time)

    def _should_trigger_alert(self, track_id, current_time):
        track = self.animal_tracks[track_id]
        return (track['consecutive'] >= CONFIG["alert"]["consecutive_frames"] and
                (current_time - track['last_alert']) >= CONFIG["alert"]["cooldown"])

    def _trigger_alert(self, track_id, current_time):
        track = self.animal_tracks[track_id]
        active_counts = defaultdict(int)
        positions = defaultdict(list)

        for tid, t in self.animal_tracks.items():
            if current_time - t['last_seen'] <= CONFIG["alert"]["activity_window"]:
                active_counts[t['class']] += 1
                positions[t['class']].append(t['positions'])

        alert_msg = []
        for cls, count in active_counts.items():
            locations = [f"{CONFIG['cameras'][t['camera']]['zone']}" 
                       for tid, t in self.animal_tracks.items()
                       if t['class'] == cls and 
                       (current_time - t['last_seen']) <= CONFIG["alert"]["activity_window"]]

            alert_msg.append(f"{count} {cls}(s) in {', '.join(set(locations))}")

        print(f"\n[ALERT] {time.ctime(current_time)}")
        print(f" - Triggered by: {track['class'].upper()} (ID: {track_id.split('_')[-1]})")
        print(f" - Location: {track['zone']}")
        print(f" - Active animals: {'; '.join(alert_msg)}")
        print(f" - Total count: {sum(active_counts.values())}")

        cursor = self.db.cursor()
        cursor.execute('''INSERT INTO alerts VALUES
                      (?, ?, ?, ?, ?, ?, ?, ?)''', (
            current_time,
            CONFIG["farm_id"],
            json.dumps(list(CONFIG["cameras"].keys())),
            track['zone'],
            json.dumps(list(active_counts.keys())),
            json.dumps(list(active_counts.values())),
            json.dumps(positions),
            track_id
        ))
        self.db.commit()
        self.animal_tracks[track_id]['last_alert'] = current_time

    def stitch_frames(self):
        active_cams = [cam_id for cam_id, cfg in CONFIG["cameras"].items() if cfg["active"]]
        if not active_cams:
            return None

        rows, cols = CONFIG["system"]["grid_layout"]
        base_height = CONFIG["system"]["base_height"]
        aspect_ratio = CONFIG["system"]["aspect_ratio"]
        cell_width = int(base_height * aspect_ratio)
        
        processed = []
        for cam_id in active_cams:
            frame = self.frames.get(cam_id, None)
            
            if frame is None:
                placeholder = np.zeros((base_height, cell_width, 3), dtype=np.uint8)
                cv2.putText(placeholder, f"{cam_id} offline", (10, base_height//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                processed.append(placeholder)
                continue

            h, w = frame.shape[:2]
            scale = min(cell_width/w, base_height/h)
            new_w, new_h = int(w*scale), int(h*scale)
            resized = cv2.resize(frame, (new_w, new_h))

            delta_w = cell_width - new_w
            delta_h = base_height - new_h
            top, bottom = delta_h//2, delta_h - delta_h//2
            left, right = delta_w//2, delta_w - delta_w//2
            bordered = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                        cv2.BORDER_CONSTANT, value=(0,0,0))
            processed.append(bordered)

        # Fill grid with placeholders if needed
        total_cells = rows * cols
        while len(processed) < total_cells:
            placeholder = np.zeros((base_height, cell_width, 3), dtype=np.uint8)
            processed.append(placeholder)

        # Create grid
        grid = []
        for i in range(0, len(processed), cols):
            row = processed[i:i+cols]
            grid.append(np.hstack(row))
        
        return np.vstack(grid[:rows])

class CameraThread(threading.Thread):
    def __init__(self, camera_id, shared_monitor):
        super().__init__()
        self.camera_id = camera_id
        self.monitor = shared_monitor
        self.running = True
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(CONFIG["cameras"][self.camera_id]["source"])
        if not self.cap.isOpened():
            print(f"Error opening {self.camera_id}")
            with self.monitor.lock:
                self.monitor.frames[self.camera_id] = None
            return

        while self.running:
            success, frame = self.cap.read()
            if not success:
                # Reset video if we reach the end
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Process frame
            processed_frame = self._process_frame(frame)
            
            with self.monitor.lock:
                self.monitor.frames[self.camera_id] = processed_frame

            time.sleep(0.03)  # Control frame processing rate

        self.cap.release()

    def _process_frame(self, frame):
        with self.monitor.lock:  # Lock model access
            results = self.monitor.model.track(
                frame,
                classes=CONFIG["model"]["classes"],
                conf=CONFIG["model"]["conf_threshold"],
                verbose=False
            )

        if results and results[0].boxes.id is not None:
            for box, track_id in zip(results[0].boxes, results[0].boxes.id):
                detection = self._create_detection(box, track_id)
                self._draw_boxes(frame, detection)
                self.monitor.update_track(self.camera_id, detection)

        return frame

    def _create_detection(self, box, track_id):
        cls_id = int(box.cls.item())
        position = box.xyxy[0].tolist()
        center = ((position[0] + position[2])//2, (position[1] + position[3])//2)
        
        return {
            'track_id': int(track_id.item()),
            'class': COCO_CLASS_NAMES.get(cls_id, "unknown"),
            'confidence': box.conf.item(),
            'position': position,
            'trail': [center]
        }

    def _draw_boxes(self, frame, detection):
        x1, y1, x2, y2 = map(int, detection['position'])
        label = f"{detection['class']} {detection['track_id']} ({detection['confidence']:.2f})"
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw movement path
        if len(detection['trail']) > 1:
            points = np.array(detection['trail'], dtype=np.int32)
            cv2.polylines(frame, [points], False, (255, 0, 0), 2)

def main():
    shared_monitor = FarmMonitor()
    threads = []

    # Start camera threads
    for cam_id, config in CONFIG["cameras"].items():
        if config["active"]:
            thread = CameraThread(cam_id, shared_monitor)
            thread.start()
            threads.append(thread)
            print(f"Started {cam_id} in zone {config['zone']}")

    # Main display loop
    cv2.namedWindow("Livestock Monitoring System", cv2.WINDOW_NORMAL)
    try:
        while True:
            stitched = shared_monitor.stitch_frames()
            if stitched is not None:
                cv2.imshow("Livestock Monitoring System", stitched)
                
                # Auto-resize window
                h, w = stitched.shape[:2]
                cv2.resizeWindow("Livestock Monitoring System", w, h)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
            elif key == ord('l'):
                print("\nLast 5 alerts:")
                cursor = shared_monitor.db.cursor()
                cursor.execute("SELECT * FROM alerts ORDER BY timestamp DESC LIMIT 5")
                for row in cursor.fetchall():
                    print(f"\n[{time.ctime(row[0])}] {json.loads(row[4])} detected in {row[3]}")
                    print(f"Cameras: {json.loads(row[2])}")
                    print(f"Counts: {json.loads(row[5])}")
                    
    finally:
        for thread in threads:
            thread.running = False
        for thread in threads:
            thread.join()
        shared_monitor.db.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()