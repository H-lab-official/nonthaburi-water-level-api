import cv2
import numpy as np
import os
import time
from flask import Flask, jsonify, send_from_directory
import yt_dlp as youtube_dl  # yt-dlp for handling YouTube live streams

app = Flask(__name__)

base_url = 'http://127.0.0.1:5000/'

# YouTube video URL and image save path
youtube_url = "https://www.youtube.com/watch?v=Isvol4kdEb0"
image_save_path = './images'

# Water level mapping
water_level_mapping = {
    1080: 1.40,
    1045: 1.50,
    1017: 1.60,
    984: 1.70,
    950: 1.80,
    922: 1.90,
    892: 2.00,
    856: 2.10,
    822: 2.20,
    785: 2.30,
    750: 2.40,
    708: 2.50,
    670: 2.60,
    627: 2.70,
    585: 2.80,
    540: 2.90,
    495: 3.00,
    445: 3.10,
    400: 3.20,
    351: 3.30,
    304: 3.40,
    254: 3.50,
    206: 3.60,
    155: 3.70,
    105: 3.80,
    53: 3.90,
    20: 4.00,
}
# Ensure the image save directory exists
if not os.path.exists(image_save_path):
    os.makedirs(image_save_path)

def get_youtube_live_stream_url(youtube_url):
    """Get the live stream URL from a YouTube video."""
    ydl_opts = {
        'format': 'best',
        'noplaylist': True,
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(youtube_url, download=False)
        return info['url']

def detect_water(frame):
    """Detect water level in the frame without showing the detection zone."""
    height, width, _ = frame.shape
    lower_line_y = None
    
    # Define the zone where we detect the water level
    zone_x_start = int(width * 0.52)
    zone_x_end = int(width * 0.58)
    zone_y_start = int(height * 0)
    zone_y_end = int(height * 0.7)

    # Crop the zone of interest
    cropped_zone = frame[zone_y_start:zone_y_end, zone_x_start:zone_x_end]
    hsv = cv2.cvtColor(cropped_zone, cv2.COLOR_BGR2HSV)

    # Define the range for yellow in the HSV space
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([90, 255, 255])

    # Create a mask for detecting yellow
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Find contours of the yellow areas
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
        # Draw the detected water level line on the full frame
        cv2.line(frame, (0, lower_line_y), (width, lower_line_y), (0, 255, 0), 2)

    return frame, lower_line_y

def map_water_level(lower_line_y):
    """Map the detected pixel line (lower_line_y) to a water level."""
    sorted_mapping = sorted(water_level_mapping.items(), key=lambda x: x[0], reverse=True)
    
    for pixel_value, water_level in sorted_mapping:
        if lower_line_y <= pixel_value:
            return water_level
    
    return "under 1.8 m"

@app.route('/', methods=['GET'])
def pakkred_capture_frame():
    stream_url = get_youtube_live_stream_url(youtube_url)
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        return jsonify({"status": "error", "message": "Unable to open YouTube live stream"}), 500

    ret, frame = cap.read()
    if not ret:
        return jsonify({"status": "error", "message": "Failed to capture frame"}), 500

    current_time = time.strftime("%d/%m/%Y %H:%M:%S", time.localtime())
    timestamp= int(time.time())*1000

    original_image_name = f"pakkred_original_{timestamp}.jpg"
    original_image_path = os.path.join(image_save_path, original_image_name)
    cv2.imwrite(original_image_path, frame)

    frame_with_water, lower_line_y = detect_water(frame)
    detection_image_name = f"pakkred_processed_{timestamp}.jpg"
    detection_image_path = os.path.join(image_save_path, detection_image_name)
    cv2.imwrite(detection_image_path, frame_with_water)

    # Map the lower_line_y to a water level
    water_level = map_water_level(lower_line_y)

    cap.release()

    return jsonify(
            {
                "cctv_location": "Pakkred",
                "original_image_url": f"{base_url}images/{original_image_name}",
                "processed_image_url": f"{base_url}images/{detection_image_name}",
                # "line_in_pixels": lower_line_y,
                "water_level_range(m)": water_level,
                "timestamp": current_time
            }
        )

