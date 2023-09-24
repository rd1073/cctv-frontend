from flask import Flask, render_template, Response
import cvlib as cv
import cv2
from vidgear.gears import CamGear
from cvlib.object_detection import draw_bbox
import numpy as np

app = Flask(__name__, static_folder='static')




#CROWD DETECTION
# Function to create a density map
def create_density_map(image, bboxes):
    height, width = image.shape[:2]
    density_map = np.zeros((height, width), dtype=np.float32)

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        density_map[y1:y2, x1:x2] += 1

    return density_map




# Initialize the video stream
# Initialize the video stream
def generate_frames():
    stream = CamGear(source='https://www.youtube.com/watch?v=gFRtAAmiFbE', stream_mode=True,logging=True)
    count = 0
    stream.start()

    while True:
        frame = stream.read()
        count += 1
        if count % 10 != 0:
            continue

        # Resize the frame if needed
        frame = cv2.resize(frame, (1020, 600))

        # Perform object detection
        bbox, label, conf = cv.detect_common_objects(frame)

        # Filter the results to include only "person" labels
        person_indices = [i for i, lbl in enumerate(label) if lbl == "person"]
        filtered_bbox = [bbox[i] for i in person_indices]
        filtered_label = ["person"] * len(person_indices)
        filtered_conf = [conf[i] for i in person_indices]

        frame = draw_bbox(frame, filtered_bbox, filtered_label, filtered_conf)
        # Create a density map based on the detected people
        density_map = create_density_map(frame, filtered_bbox)

        # Normalize the density map
        density_map = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)

        # Convert the density map to a color map for visualization
        density_map_color = cv2.applyColorMap(np.uint8(density_map), cv2.COLORMAP_JET)

        # Overlay the density map on top of the original frame
        result_frame = cv2.addWeighted(frame, 0.7, density_map_color, 0.3, 0)

        # Encode the frame into JPEG format
        ret, buffer = cv2.imencode('.jpg', result_frame)

        if not ret:
            break

        # Yield the frame as bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')


@app.route('/')
def index():
    return render_template('technology.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')





#CROWD MANAGEMENT
#DYNAMIC

def create_density_map_dynamic(image, bboxes, num_people, alarm_threshold, min_people_per_box):
    density_map = np.zeros_like(image, dtype=np.uint8)
    background_color = (173, 216, 230)  # Set the background color (white)

    # Fill the density map with the background color
    density_map[:] = background_color

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x, y = (x1 + x2) // 2, (y1 + y2) // 2  # Use the center of the bounding box

        # Draw horizontal crossbar
        cv2.line(density_map, (x - 10, y), (x + 10, y), (0, 0, 255), 2)

        # Draw vertical crossbar
        cv2.line(density_map, (x, y - 10), (x, y + 10), (0, 0, 255), 2)

    # Initialize the count of boxes
    num_boxes = 0

    # Sort bounding boxes by their x-coordinate (left to right)
    bboxes.sort(key=lambda x: (x[0] + x[2]) / 2)

    # Detect groups of at least min_people_per_box people who are closeby
    group = []
    for bbox in bboxes:
        if not group or bbox[0] - group[-1][2] <= 20:
            group.append(bbox)
        else:
            if len(group) >= min_people_per_box:
                # Draw a border around the group
                x1 = group[0][0]
                y1 = min(bbox[1] for bbox in group)
                x2 = group[-1][2]
                y2 = max(bbox[3] for bbox in group)
                border_color = (0, 0, 255)  # Red color for the border
                cv2.rectangle(density_map, (x1, y1), (x2, y2), border_color, 2)
                num_boxes += 1
            group = [bbox]

    # Display the count of people at the bottom left corner
    count_message = f"People Count: {num_people}"
    cv2.putText(density_map, count_message, (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Check if the alarm threshold is exceeded and display the alarm message
    if num_people > alarm_threshold:
        alarm_message = f"ALARM: {num_people} people detected in {num_boxes} groups!"
        cv2.putText(density_map, alarm_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return density_map, num_boxes

# Threshold for the number of people to trigger an alarm
people_threshold = 5

# Video capture using CamGear
stream = CamGear(source='https://www.youtube.com/watch?v=gFRtAAmiFbE', stream_mode=True, logging=True).start()

def generate_frames_dynamic():
    count = 0
    while True:
        frame = stream.read()
        count += 1
        if count % 10 != 0:
            continue

        frame = cv2.resize(frame, (1020, 600))
        bbox, label, conf = cv.detect_common_objects(frame)

        # Filter the results to include only "person" labels
        person_indices = [i for i, lbl in enumerate(label) if lbl == "person"]
        filtered_bbox = [bbox[i] for i in person_indices]

        # Count the number of people
        num_people = len(filtered_bbox)

        # Create a density map with moving crosses, dynamic boxes, and people count
        density_map, num_boxes = create_density_map_dynamic(frame, filtered_bbox, num_people, people_threshold, min_people_per_box=5)

        # Encode the image as JPEG
        _, buffer = cv2.imencode('.jpg', density_map)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route('/dynamic')
def dynamic():
    return Response(generate_frames_dynamic(), mimetype='multipart/x-mixed-replace; boundary=frame')


'''
#STATIC

# Define the coordinates of the specified box for counting people
box_x1, box_y1, box_x2, box_y2 = 300, 300, 700, 600  # Adjust the box coordinates as needed

# Function to create a density map with moving crosses
def create_density_map_static(image, bboxes, num_people, alarm_threshold):
    density_map = np.zeros_like(image, dtype=np.uint8)
    background_color = (255, 200, 200)  # Set the background color (white)

    # Fill the density map with the background color
    density_map[:] = background_color

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x, y = (x1 + x2) // 2, (y1 + y2) // 2  # Use the center of the bounding box

        # Draw horizontal crossbar
        cv2.line(density_map, (x - 10, y), (x + 10, y), (0, 0, 255), 2)

        # Draw vertical crossbar
        cv2.line(density_map, (x, y - 10), (x, y + 10), (0, 0, 255), 2)

    # Draw a border around the specified area for measurement
    border_color = (0, 0, 255)  # Red color for the border
    cv2.rectangle(density_map, (box_x1, box_y1), (box_x2, box_y2), border_color, 2)

    # Display the count of people at the bottom left corner
    count_message = f"People Count: {num_people}"
    cv2.putText(density_map, count_message, (10, image.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Check if the alarm threshold is exceeded and display the alarm message
    if num_people > alarm_threshold:
        alarm_message = f"ALARM: {num_people} people detected!"
        cv2.putText(density_map, alarm_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    return density_map

# Threshold for the number of people to trigger an alarm
people_threshold = 5

# Video capture using CamGear
stream = CamGear(source='https://www.youtube.com/watch?v=gFRtAAmiFbE', stream_mode=True, logging=True).start()

def generate_frames_static():
    count = 0
    while True:
        frame = stream.read()
        count += 1
        if count % 10 != 0:
            continue

        frame = cv2.resize(frame, (1020, 600))
        bbox, label, conf = cv.detect_common_objects(frame)

        # Filter the results to include only "person" labels
        person_indices = [i for i, lbl in enumerate(label) if lbl == "person"]
        filtered_bbox = [bbox[i] for i in person_indices]

        # Count the number of people in the specified box
        num_people_box = sum(1 for bbox in filtered_bbox if box_x1 <= bbox[0] <= box_x2 and box_y1 <= bbox[1] <= box_y2)

        # Count the total number of people
        num_people_total = len(filtered_bbox)

        # Create a density map with moving crosses based on the detected people and the total count
        density_map = create_density_map_static(frame, filtered_bbox, num_people_box, people_threshold)

        # Encode the image as JPEG
        _, buffer = cv2.imencode('.jpg', density_map)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/static')
def static_density():
    return Response(generate_frames_static(), mimetype='multipart/x-mixed-replace; boundary=frame')


#THERMAL

# Function to create a density map
def create_density_map_thermal(image, bboxes):
    height, width = image.shape[:2]
    density_map = np.zeros((height, width), dtype=np.float32)

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        density_map[y1:y2, x1:x2] += 1

    return density_map

# Threshold for the number of people (adjust as needed)
people_threshold = 5

stream = CamGear(source='https://www.youtube.com/watch?v=gFRtAAmiFbE', stream_mode=True, logging=True).start()
count = 0


def generate_frames_thermal():
    while True:
        frame = stream.read()
        global count
        count += 1
        if count % 10 != 0:
            continue

        frame = cv2.resize(frame, (1020, 600))
        bbox, label, conf = cv.detect_common_objects(frame)

        # Filter the results to include only "person" labels
        person_indices = [i for i, lbl in enumerate(label) if lbl == "person"]
        filtered_bbox = [bbox[i] for i in person_indices]
        filtered_label = ["person"] * len(person_indices)
        filtered_conf = [conf[i] for i in person_indices]

        frame = draw_bbox(frame, filtered_bbox, filtered_label, filtered_conf)

        # Create a density map based on the detected people
        density_map = create_density_map_thermal(frame, filtered_bbox)

        # Normalize the density map
        density_map = cv2.normalize(density_map, None, 0, 255, cv2.NORM_MINMAX)

        # Convert the density map to a color map for visualization
        density_map_color = cv2.applyColorMap(np.uint8(density_map), cv2.COLORMAP_JET)

        # Count the number of people in the density map
        num_people = len(filtered_bbox)

        # Display the alert message on the density map
        alert_message = f"ALARM: {num_people} people detected!"
        cv2.putText(density_map_color, alert_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', density_map_color)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/thermal')
def thermal():
    return Response(generate_frames_thermal(), mimetype='multipart/x-mixed-replace; boundary=frame')
'''

#SIGNALS
colors = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
}
def detect_signal_color(roi):
    # Implement your color detection algorithm here based on the ROI pixels
    # For this example, we assume it's red if the average hue value is within the red range
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    avg_hue = hsv_roi[:, :, 0].mean()
    if 0 <= avg_hue <= 30 or 150 <= avg_hue <= 180:
        return 'red'
    else:
        return 'green'
    
def generate_frames_signals():
    stream = CamGear(source='https://www.youtube.com/watch?v=g7CJ3pm-e7s', stream_mode=True, logging=True).start()
    count = 0
    while True:
        frame = stream.read()
        count += 1
        if count % 10 != 0:
            continue

        frame = cv2.resize(frame, (1020, 600))
        bbox, label, conf = cv.detect_common_objects(frame)
        frame = draw_bbox(frame, bbox, label, conf)

        # Process each detected signal ROI
        for i, roi in enumerate(bbox):
            x1, y1, x2, y2 = roi
            signal_roi = frame[y1:y2, x1:x2]
            signal_color = detect_signal_color(signal_roi)

            # Draw the detected color on the frame
            color_rgb = colors.get(signal_color, (0, 0, 0))  # Default to black if color is not found
            cv2.putText(frame, signal_color.capitalize(), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_rgb, 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/signals')
def signals():
    return Response(generate_frames_signals(), mimetype='multipart/x-mixed-replace; boundary=frame')







if __name__ == '__main__':
    app.run(debug=True)