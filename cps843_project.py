# Import Libraries 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt

def saveImage(image, name):
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, cmap='gray')
    fig.savefig(name)

# Colors
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)

# Initialize num_people
num_people = 0

# Load image/video
image = cv2.imread('site2.jpg')
height, width, _ = image.shape
# Check for errors
if image is None:
    print("Could not read the image.")
    raise SystemExit

# Preprocessing
# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# saveImage(gray, 'gray.png')
# Gaussian blur
gaussian_blur = cv2.GaussianBlur(gray, (9, 9), 0) 
# saveImage(gaussian_blur, 'gaussian_blur.png')

# Object Detection
## YOLO
yolo = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
layer_names = yolo.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers().flatten()]
classes = open('coco.names').read().strip().split('\n')
blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
yolo.setInput(blob)
outs = yolo.forward(output_layers)

# Bounding Boxes
# Show information on the screen
class_ids = []
confidences = []
boxes = []
for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)
## Apply Non-max suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
## Draw bounding boxes for YOLO detections -- now only draws rectangles around people
for i in indices.flatten():
    box = boxes[i]
    label = str(classes[class_ids[i]])
    if label == "person":  # Check if the detected class is a person
        num_people = num_people + 1
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(image, (x, y), (x + w, y + h), green, 2)
        cv2.putText(image, f'Person {num_people}', (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, red, 3)

# Display Count
# PutText
cv2.putText(image, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, red, 2)
cv2.putText(image, f'Total People Detected : {num_people}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, red, 2)
print(f"Objects detected by YOLO: {num_people}")

# Output
# Show Frame
saveImage(image, 'result.png')
cv2.waitKey(0)

# Cleanup
# Release and Destroy
# Release only for video
cv2.destroyAllWindows()