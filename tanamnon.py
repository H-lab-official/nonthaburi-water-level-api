# tanamnon.py
import cv2
import numpy as np
import requests
import os
from flask import  jsonify
import time

# Base URL pattern for the HLS stream
base_url = 'http://127.0.0.1:5000/'
tanamnon_url = "https://stream.firsttech.co.th/live/nakornnont.stream/chunklist_{}.m3u8"
image_save_path = './images'

# Water level mapping for Tanamnon
tanamnon_water_level_mapping = {
    351: 1.8,
    329.1: 1.9,
    309.8: 2.00,
    289.2: 2.1,
    269.8: 2.2,
    259.1: 2.24,
    254.8: 2.26,
    250.7: 2.28,
    247.2: 2.3,
    227: 2.4,
    215.7: 2.45,
    205.8: 2.5,
    196.1: 2.55,
    185.5: 2.6,
    174.2: 2.65,
    164.2: 2.7,
    153.4: 2.75,
    143: 2.8,
    131.8: 2.85,
    121.2: 2.9,
    110: 2.95,
    100.3: 3.0,
    87.8: 3.05,
    77.8: 3.1,
    67: 3.15,
    56.4: 3.2,
    44.8: 3.25,
    34.8: 3.3,
    29.5: 3.32,
    24.6: 3.34,
    19.8: 3.36,
    16.4: 3.38,
    12.7: 3.4,
}

# Ensure the image save directory exists
if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)

def is_valid_stream_url(url):
    """Check if the stream URL is valid."""
    response = requests.get(url)
    return response.status_code == 200

def detect_water(frame):
    """Detect water level in the frame."""
    height, width, _ = frame.shape
    lower_line_y = None
    
    zone_x_start = int(width * 0.42)
    zone_x_end = int(width * 0.58)
    zone_y_start = int(height * 0)
    zone_y_end = int(height * 0.7)

    cropped_zone = frame[zone_y_start:zone_y_end, zone_x_start:zone_x_end]
    hsv = cv2.cvtColor(cropped_zone, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([90, 255, 255])
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 100, 100])
        upper_red2 = np.array([180, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = mask1 + mask2
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        lower_line_y = zone_y_start + y + h
        cv2.line(frame, (0, lower_line_y), (width, lower_line_y), (0, 255, 0), 1)

    return frame, lower_line_y

def map_water_level(lower_line_y):
    """Map the detected pixel line (lower_line_y) to a water level."""
    sorted_mapping = sorted(tanamnon_water_level_mapping.items(), key=lambda x: x[0], reverse=True)
    
    for pixel_value, water_level in sorted_mapping:
        if lower_line_y <= pixel_value:
            return water_level
    
    return "over"


def tamannon_capture_frame():
    start_chunklist_id = 1000
    end_chunklist_id = 1100

    for chunklist_id in range(start_chunklist_id, end_chunklist_id + 1):
        stream_url = tanamnon_url.format(chunklist_id)
        
        if is_valid_stream_url(stream_url):
            try:
                cap = cv2.VideoCapture(stream_url)
                if not cap.isOpened():
                    return jsonify({"status": "error", "message": f"Unable to open stream {stream_url}"}), 500

                ret, frame = cap.read()
                if not ret:
                    return jsonify({"status": "error", "message": "Failed to capture frame"}), 500

                current_time = time.strftime("%d/%m/%Y %H:%M:%S", time.localtime())
                timestamp=int(time.time())*1000

                original_image_name = f"tanamnon_original_{timestamp}.jpg"
                original_image_path = os.path.join(image_save_path, original_image_name)
                cv2.imwrite(original_image_path, frame)

                frame_with_water, lower_line_y = detect_water(frame)
                detection_image_name = f"tanamnon_processed_{timestamp}.jpg"
                detection_image_path = os.path.join(image_save_path, detection_image_name)
                cv2.imwrite(detection_image_path, frame_with_water)

                water_level = map_water_level(lower_line_y)

                cap.release()

                return jsonify(
                        {
                            "cctv_location": "Tanamnon",
                            "original_image_url": f"{base_url}images/{original_image_name}",
                            "processed_image_url": f"{base_url}images/{detection_image_name}",
                            # "line_in_pixels": lower_line_y,
                            "water_level_range(m)": water_level,
                            "timestamp": current_time
                        }
                    )

            except Exception as e:
                return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({"status": "error", "message": "No valid stream found"}), 404

