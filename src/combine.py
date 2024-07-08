
import cv2
import torch
import numpy as np


def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    model.classes = [32]  # 32 is the class index for ball in COCO dataset
    return model


def detect_grid(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    # Apply morphological operations to enhance lines
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Detect lines using Hough Line Transform
    lines = cv2.HoughLinesP(morph, 1, np.pi / 180,
                            threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None:
        return None, None, None, None

    # Separate horizontal and vertical lines
    horizontal_lines = []
    vertical_lines = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x2 - x1) > abs(y2 - y1):
            horizontal_lines.append((x1, y1, x2, y2))
        else:
            vertical_lines.append((x1, y1, x2, y2))

    # Find the outermost lines
    if len(horizontal_lines) < 2 or len(vertical_lines) < 2:
        return None, None, None, None

    horizontal_lines.sort(key=lambda line: line[1])  # Sort by y-coordinate
    vertical_lines.sort(key=lambda line: line[0])    # Sort by x-coordinate

    top = horizontal_lines[0]
    bottom = horizontal_lines[-1]
    left = vertical_lines[0]
    right = vertical_lines[-1]

    # Calculate grid boundaries
    x = min(left[0], left[2])
    y = min(top[1], top[3])
    w = max(right[0], right[2]) - x
    h = max(bottom[1], bottom[3]) - y

    # Define quadrants
    mid_x = x + w // 2
    mid_y = y + h // 2
    quadrants = {
        3: (x, mid_x, y, mid_y),
        4: (mid_x, x + w, y, mid_y),
        2: (x, mid_x, mid_y, y + h),
        1: (mid_x, x + w, mid_y, y + h)
    }

    # Return the rectangle coordinates, vertical lines, and quadrants
    rectangle = (x, y, x + w, y + h)
    middle_lines = [(mid_x, y, mid_x, y + h), (x, mid_y, x + w, mid_y)]

    return rectangle, middle_lines, quadrants


def draw_grid(image, rectangle, middle_lines):
    x1, y1, x2, y2 = rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for line in middle_lines:
        x1, y1, x2, y2 = line
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


def classify_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    avg_hsv = np.mean(hsv, axis=(0, 1))
    h, s, v = avg_hsv

    if v > 150 and s < 50:
        return 'white'
    elif h < 11 or h > 170:
        return 'red'
    elif 19 < h < 34:
        return 'yellow'
    elif 33 < h < 70:
        return 'green'
    elif 70 < h < 150:
        return 'blue'
    else:
        return 'pink'


def update_trackers(multi_tracker, frame, ball_colors):
    success, boxes = multi_tracker.update(frame)
    updated_boxes = []
    for i, newbox in enumerate(boxes):
        x, y, w, h = [int(v) for v in newbox]
        track_id = list(ball_colors.keys())[i]
        updated_boxes.append((x, y, w, h, track_id))
    return updated_boxes


def add_new_trackers(model, frame, multi_tracker, ball_colors, ball_ids, current_id, iou_threshold=0.1):
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        track_id = f"{x1}{y1}{x2}_{y2}"

        if track_id not in ball_colors:
            # Calculate IoU with existing trackers
            iou_scores = []
            for existing_box in multi_tracker.getObjects():
                existing_x1, existing_y1, existing_w, existing_h = existing_box
                existing_x2, existing_y2 = existing_x1 + existing_w, existing_y1 + existing_h
                intersection_area = max(0, min(x2, existing_x2) - max(x1, existing_x1)) * max(
                    0, min(y2, existing_y2) - max(y1, existing_y1))
                union_area = (x2 - x1) * (y2 - y1) + (existing_x2 - existing_x1) * \
                    (existing_y2 - existing_y1) - intersection_area
                iou = intersection_area / union_area
                iou_scores.append(iou)

            if max(iou_scores, default=0) < iou_threshold:
                tracker = cv2.legacy.TrackerKCF_create()
                multi_tracker.add(tracker, frame, (x1, y1, x2 - x1, y2 - y1))
                ball_colors[track_id] = classify_color(frame[y1:y2, x1:x2])
                ball_ids[track_id] = current_id
                current_id += 1

    return current_id


def log_event(event_log, frame_count, fps, current_quadrant, color, event_type):
    timestamp = frame_count / fps * 1000
    event_log.append(
        f"{timestamp:.0f}, {current_quadrant}, {color}, {event_type}")


# Process the video and track balls
def process_video(video_path, output_video_path, output_text_path):
    model = load_model()
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    event_log = []
    frame_count = 0
    process_frame_interval = 1
    multi_tracker = cv2.legacy.MultiTracker_create()
    ball_colors = {}
    ball_quadrants = {}
    ball_ids = {}
    current_id = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count == 0:
            rectangle, middle_lines, quadrants = detect_grid(frame)
            if quadrants is None:
                raise RuntimeError("Failed to detect grid in the first frame")

        display_frame = frame.copy()

        if frame_count % process_frame_interval == 0:
            current_id = add_new_trackers(
                model, frame, multi_tracker, ball_colors, ball_ids, current_id)

            updated_boxes = update_trackers(multi_tracker, frame, ball_colors)
            draw_grid(display_frame, rectangle, middle_lines)

        for (x, y, w, h, track_id) in updated_boxes:
            cx = int(x + w / 2)
            cy = int(y + h / 2)
            color = ball_colors[track_id]
            ball_id = ball_ids[track_id]

            cv2.rectangle(display_frame, (x, y),
                          (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(display_frame, f"{color} ball", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            current_quadrant = next(
                (q for q, (qx1, qx2, qy1, qy2) in quadrants.items() if qx1 <= cx < qx2 and qy1 <= cy < qy2), None)

            if ball_id not in ball_quadrants:
                if current_quadrant is not None:
                    log_event(event_log, frame_count, fps,
                              current_quadrant, color, "Entry")
                ball_quadrants[ball_id] = current_quadrant
            elif ball_quadrants[ball_id] != current_quadrant:
                if ball_quadrants[ball_id] is not None:
                    log_event(event_log, frame_count, fps,
                              ball_quadrants[ball_id], color, "Exit")
                if current_quadrant is not None:
                    log_event(event_log, frame_count, fps,
                              current_quadrant, color, "Entry")
                ball_quadrants[ball_id] = current_quadrant

        for track_id in list(ball_colors.keys()):
            if track_id not in [updated_boxes[i][4] for i in range(len(updated_boxes))]:
                ball_id = ball_ids[track_id]
                if ball_id in ball_quadrants and ball_quadrants[ball_id] is not None:
                    log_event(event_log, frame_count, fps,
                              ball_quadrants[ball_id], ball_colors[track_id], "Exit")
                del ball_colors[track_id]
                del ball_quadrants[track_id]
                del ball_ids[track_id]

        out.write(display_frame)
        frame_count += 1

        cv2.imshow('Frame', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    with open(output_text_path, 'w') as f:
        f.write("Time (ms), Quadrant Number, Ball Colour, Type (Entry or Exit)\n")
        for event in event_log:
            f.write(event + "\n")


if __name__ == "__main__":
    process_video('../Input/video.mp4', '../Output/output_video.mp4',
                  '../Output/event_log.txt')
