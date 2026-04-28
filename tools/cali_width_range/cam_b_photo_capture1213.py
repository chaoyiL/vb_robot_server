from __future__ import annotations

import collections
import time
import os
from datetime import datetime
import sys
import threading
from pathlib import Path

import cv2

# Add project root to path for V4L2Camera import
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))
from utils.camera_device import V4L2Camera

FPS = 30

HEADLESS_MODE = 'SSH_CLIENT' in os.environ or 'SSH_TTY' in os.environ or not os.environ.get('DISPLAY')
if HEADLESS_MODE:
    print("Headless environment detected, enabling headless mode")

# Camera configuration for left hand (matches 00_get_data.py)
DEVICE_PATH = "/dev/video0"  # Left hand camera
CAMERA_FORMAT = "MJPG"
CAMERA_WIDTH = 3840
CAMERA_HEIGHT = 800

# Crop parameters (matches 01_crop_img.py)
CROP_WIDTH = 1280
TOTAL_WIDTH = 3840


def crop_and_rotate_visual(frame):
    """
    Crop visual part from 3840x800 image and rotate 180 degrees (for left hand).
    
    Args:
        frame: Input frame (3840x800)
    
    Returns:
        Rotated visual image (1280x800) or None if invalid
    """
    if frame is None:
        return None
    
    h, w = frame.shape[:2]
    
    # Verify image dimensions
    if w != TOTAL_WIDTH or h != CAMERA_HEIGHT:
        print(f"[WARN] Unexpected image size: {w}x{h}, expected {TOTAL_WIDTH}x{CAMERA_HEIGHT}")
        # Need at least 2*CROP_WIDTH to crop visual part
        if w < 2 * CROP_WIDTH:
            return None
    
    # Crop visual part (middle section: 1280-2560)
    # Ensure we don't exceed image bounds
    end_x = min(2 * CROP_WIDTH, w)
    visual = frame[:, CROP_WIDTH:end_x]
    
    # Rotate 180 degrees (for left hand)
    visual_rotated = cv2.rotate(visual, cv2.ROTATE_180)
    
    return visual_rotated


def main():
    photo_counter = 0
    current_frame = None
    current_visual = None
    
    def take_photo():
        nonlocal photo_counter
        
        if current_visual is None:
            print("No visual frame available, please try again later")
            return
        
        os.makedirs("photos", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        photo_counter += 1
        filename = f"photos/CAM_{photo_counter:04d}_{timestamp}.jpg"
        
        # Save rotated visual image (1280x800)
        success = cv2.imwrite(filename, current_visual)
        
        if success:
            print(f"Photo saved: {filename}")
            print(f"   Resolution: {current_visual.shape[1]}x{current_visual.shape[0]} (visual, rotated 180°)")
            print(f"   Total photos: {photo_counter}\n")
        else:
            print(f"Failed to save photo: {filename}")
    
    # Initialize V4L2Camera
    print(f"Initializing camera: {DEVICE_PATH}")
    try:
        camera = V4L2Camera(
            device_path=DEVICE_PATH,
            format=CAMERA_FORMAT,
            width=CAMERA_WIDTH,
            height=CAMERA_HEIGHT
        )
    except Exception as e:
        print(f"[ERROR] Cannot open camera {DEVICE_PATH}: {e}")
        print("        Please check if the device exists and has proper permissions")
        return
    
    # Configure camera parameters (matches 00_get_data.py config)
    print("Configuring camera parameters...")
    camera.set_white_balance(auto=False, temperature=4600)  # Manual WB, 4600K
    camera.set_exposure(auto=False, exposure_time=400)  # Manual exposure, 400
    camera.set_brightness(brightness=0)
    camera.set_gain(100)
    camera.set_gamma(100)
    
    print(f"Camera initialized: {CAMERA_WIDTH}x{CAMERA_HEIGHT}, format={CAMERA_FORMAT}")
    
    if not HEADLESS_MODE:
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera", 1200, 400)  # Adjusted for 3840x800 display
    
    fps_handler = FPSHandler()
    
    print("\n" + "="*50)
    print("V4L2 Camera Photo Capture Program (Left Hand Visual)")
    print("="*50)
    print(f"Using camera: {DEVICE_PATH}")
    print(f"Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}")
    print(f"Crop: Visual part ({CROP_WIDTH}x{CAMERA_HEIGHT})")
    print(f"Rotation: 180° (for left hand)")
    if HEADLESS_MODE:
        print("Headless mode controls:")
        print("   Press Enter - Take photo")
        print("   Type 'q' + Enter - Exit program")
    else:
        print("Controls:")
        print("   Press 's' key - Take photo")
        print("   Press 'q' key - Exit program")
    print("Photos will be saved to photos/ directory (visual, rotated)")
    print("="*50 + "\n")
    
    should_exit = False
    
    if HEADLESS_MODE:
        def input_handler():
            nonlocal should_exit
            while True:
                try:
                    user_input = input().strip().lower()
                    if user_input == 'q':
                        should_exit = True
                        break
                    elif user_input == '':
                        take_photo()
                except EOFError:
                    should_exit = True
                    break
        
        import threading
        input_thread = threading.Thread(target=input_handler, daemon=True)
        input_thread.start()
    
    try:
        while not should_exit:
            ret, frame = camera.read()
            
            # V4L2Camera.read() returns:
            #   - On timeout: (False, None)
            #   - On success: ((timestamp_str, ram_time_str), frame)
            if ret is False or frame is None:
                time.sleep(0.1)
                continue
            
            # Crop and rotate visual part
            visual_rotated = crop_and_rotate_visual(frame)
            if visual_rotated is None:
                time.sleep(0.1)
                continue
            
            current_frame = frame
            current_visual = visual_rotated
            fps_handler.tick("FRAME")
            
            # Create display frame (show original with visual region highlighted)
            display_frame = frame.copy()
            fps_handler.draw_fps(display_frame, "FRAME")
            
            # Draw rectangle around visual region
            cv2.rectangle(display_frame, 
                         (CROP_WIDTH, 0), 
                         (2*CROP_WIDTH - 1, CAMERA_HEIGHT - 1), 
                         (0, 255, 0), 3)
            
            # Draw dividing lines
            cv2.line(display_frame, (CROP_WIDTH, 0), (CROP_WIDTH, CAMERA_HEIGHT - 1), (0, 255, 0), 2)
            cv2.line(display_frame, (2*CROP_WIDTH, 0), (2*CROP_WIDTH, CAMERA_HEIGHT - 1), (0, 255, 0), 2)
            
            draw_text(display_frame, f"Camera: {DEVICE_PATH}", (10, 30), 
                    color=(0, 255, 0), bg_color=(0, 0, 0), font_scale=0.7, thickness=2)
            draw_text(display_frame, f"Photos taken: {photo_counter}", (10, 60), 
                    color=(0, 255, 0), bg_color=(0, 0, 0), font_scale=0.7, thickness=2)
            draw_text(display_frame, f"Resolution: {CAMERA_WIDTH}x{CAMERA_HEIGHT}", (10, 90), 
                    color=(0, 255, 0), bg_color=(0, 0, 0), font_scale=0.7, thickness=2)
            draw_text(display_frame, "Visual region (green box, will be rotated 180°)", (CROP_WIDTH + 10, 30), 
                    color=(0, 255, 0), bg_color=(0, 0, 0), font_scale=0.6, thickness=1)
            
            hint_text = "Press 's' to take photo" if not HEADLESS_MODE else "Press Enter to take photo"
            draw_text(display_frame, hint_text, (10, display_frame.shape[0] - 30), 
                    color=(255, 255, 255), bg_color=(0, 0, 0), font_scale=0.6, thickness=1)
            
            if not HEADLESS_MODE:
                cv2.imshow("Camera", display_frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    should_exit = True
                    break
                elif key == ord("s"):
                    take_photo()
            else:
                if should_exit:
                    break
                time.sleep(1.0 / FPS)
    
    except KeyboardInterrupt:
        print("\nUser interrupted program")
    
    finally:
        camera.release()
        if not HEADLESS_MODE:
            cv2.destroyAllWindows()
        
        print("\n" + "="*50)
        print(f"Photo capture completed! Total photos taken: {photo_counter}")
        if photo_counter > 0:
            print("Photos saved in photos/ directory (visual, rotated 180°)")
        print("Thank you for using!")
        print("="*50)


class FPSHandler:

    def __init__(self, max_ticks=100):
        self._ticks = {}
        self._maxTicks = max_ticks

    def tick(self, name):
        if name not in self._ticks:
            self._ticks[name] = collections.deque(maxlen=self._maxTicks)
        self._ticks[name].append(time.monotonic())

    def tick_fps(self, name):
        if name in self._ticks and len(self._ticks[name]) > 1:
            time_diff = self._ticks[name][-1] - self._ticks[name][0]
            return (len(self._ticks[name]) - 1) / time_diff if time_diff != 0 else 0.0
        return 0.0

    def draw_fps(self, frame, name):
        frame_fps = f"{name} FPS: {round(self.tick_fps(name), 1)}"
        draw_text(
            frame,
            frame_fps,
            (5, 15),
            color=(255, 255, 255),
            bg_color=(0, 0, 0),
        )


def draw_text(
    frame,
    text,
    org,
    color=(255, 255, 255),
    bg_color=(128, 128, 128),
    font_scale=0.5,
    thickness=1,
):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, bg_color, thickness + 3, cv2.LINE_AA)
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)


if __name__ == "__main__":
    main()
