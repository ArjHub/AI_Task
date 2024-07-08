# Ball Tracking System

This system is designed to track multiple colored balls and log their entry and exit timings in predefined quadrants.

## Overview

The system utilizes the YOLOv5 model for initial ball detection and the OpenCV library to track the balls' positions and colors within the quadrants. The color classification is done using the HSV color space.

## Key Features

- **YOLOv5 for Ball Detection**: The system leverages the pre-trained YOLOv5 model to detect balls. The model is fine-tuned to recognize the ball class from the COCO dataset.
  
- **Color Classification**: Using the HSV color space, the system classifies the detected balls into colors such as white, red, yellow, green, blue, and pink.
  
- **Grid Detection**: The system detects grids by identifying horizontal and vertical lines using the Hough Line Transform. These lines help in calculating the grid boundaries and defining each quadrant.

- **IoU-based Tracker Management**: The system uses Intersection over Union (IoU) scores with a specific threshold to prevent multiple labels for the same object. This ensures efficient addition and updating of trackers.

## How It Works

1. **Model Loading**: The YOLOv5 model is loaded and configured to detect balls.
2. **Grid Detection**: The system detects the grid on the first frame to define the quadrants.
3. **Ball Detection and Tracking**: Balls are detected in each frame, and new trackers are added if the IoU score is below the threshold.
4. **Color Classification**: Each detected ball's color is classified using its HSV values.
5. **Quadrant Tracking**: The system tracks the movement of balls across different quadrants and logs the entry and exit events.

## Code Structure

- **load_model()**: Loads and configures the YOLOv5 model.
- **detect_grid(image)**: Detects the grid and defines quadrants.
- **draw_grid(image, rectangle, middle_lines)**: Draws the detected grid on the image.
- **classify_color(roi)**: Classifies the color of a ball using HSV values.
- **update_trackers(multi_tracker, frame, ball_colors)**: Updates the positions of the tracked balls.
- **add_new_trackers(model, frame, multi_tracker, ball_colors, ball_ids, current_id, iou_threshold=0.1)**: Adds new trackers for newly detected balls.
- **log_event(event_log, frame_count, fps, current_quadrant, color, event_type)**: Logs the entry and exit events of the balls.
- **process_video(video_path, output_video_path, output_text_path)**: Main function to process the video, track balls, and log events.

## Getting Started

1. **Install Dependencies**: Ensure you have the necessary dependencies installed.
   ```bash
   pip install opencv-python-headless torch
   ```
2. **Run the System**: Execute the main script to process the video and track the balls.
   ```bash
   python combine.py
   ```

## Example

The following command processes the input video, tracks the balls, and saves the output video along with the event log.
```bash
python combine.py ../Input/video.mp4 ../Output/output_video.mp4 ../Output/event_log.txt
```

## Notes

- Ensure the input video is less than 100 MB if uploading to GitHub. Use Git Large File Storage (LFS) for larger files.
- The system's efficiency depends on the quality of the grid detection and the accuracy of the YOLOv5 model.

---

Feel free to modify and expand this README as per your project's needs.