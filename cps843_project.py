# Import Libraries 
import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import argparse

def saveImage(image, name):
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, cmap='gray')
    fig.savefig(name)

def process_image(input, output):
    # Colors
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)

    # Initialize num_people
    num_people = 0

    # Load image/video
    image = cv2.imread(input)
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
            #cv2.putText(image, f'Person {num_people}', (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 2, red, 2)

    # Display Count
    # PutText
    cv2.putText(image, 'Status : Detecting ', (40,40), cv2.FONT_HERSHEY_DUPLEX, 0.8, red, 2)
    cv2.putText(image, f'Total People Detected : {num_people}', (40,70), cv2.FONT_HERSHEY_DUPLEX, 0.8, red, 2)
    print(f"Objects detected by YOLO: {num_people}")

    # Output
    # Show Frame
    saveImage(image, output)
    cv2.waitKey(0)

    # Cleanup
    # Release and Destroy
    # Release only for video
    cv2.destroyAllWindows()

def process_video(input, output):
    # Colors
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)

    # Load video
    cap = cv2.VideoCapture(input, cv2.CAP_MSMF)
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Check for errors
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can also try other codecs like 'XVID'
    out = cv2.VideoWriter(output, fourcc, 20.0, (width, height))

    # Variables for controlling video playback
    paused = False
    current_frame = 0

    while cap.isOpened():
        # Read a frame from the video
        ret, frame = cap.read()

        if not paused:
            current_frame += 1

        if ret:
            # Initialize num_people for each frame
            num_people = 0

            # Preprocessing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gaussian_blur = cv2.GaussianBlur(gray, (9, 9), 0)

            # Object Detection (YOLO)
            yolo = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
            layer_names = yolo.getLayerNames()
            output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers().flatten()]
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False) # can be changed to (608, 608), this increases input size of object.
            yolo.setInput(blob)
            outs = yolo.forward(output_layers)

            # Bounding Boxes
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if confidence > 0.3 and class_id == 0:  # Check if the detected class is a person
                        num_people += 1
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

            # Apply Non-max suppression
            if boxes and confidences:
                indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.4)

                # Check if indices is a tuple, and convert it to a NumPy array
                indices = np.array(indices) if isinstance(indices, tuple) else indices
            
                # Draw bounding boxes for YOLO detections
                num_people = len(indices)
                for i in indices.flatten():
                    box = boxes[i]
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), green, 2)
                    
            # Display Count
            cv2.putText(frame, 'Status : Detecting ', (40, 40), cv2.FONT_HERSHEY_DUPLEX, 0.8, red, 2)
            cv2.putText(frame, f'Total People Detected : {num_people}', (40, 70), cv2.FONT_HERSHEY_DUPLEX, 0.8, red, 2)

            # Show the frame
            cv2.imshow('Frame', frame)

            # Write the frame to the output video file
            out.write(frame)

            # Press P to pause/resume the video
            key = cv2.waitKey(60)  # Decreased wait time
            if key == ord('p'):
                paused = not paused

            # Press Q on the keyboard to exit
            elif key == ord('q'):
                break

            # Press S to skip frames (10 frames per press)
            elif key == ord('s'):
                current_frame += 10

            # Set the video capture to the specified frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)

        else:
            break
        
    # Release the VideoWriter object
    out.release()

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

    print(f"Total People Detected in the video: {num_people}")

def main():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser(description="A script that processes image or video files.")

    # Add arguments
    parser.add_argument("-i", "--image", help="Specify if input is an image", action="store_true")
    parser.add_argument("-v", "--video", help="Specify if input is a video", action="store_true")
    parser.add_argument("input_file", help="Input file path")
    parser.add_argument("-o", "--output", help="Output file path", action="store_true")
    parser.add_argument("output_file", help="Output file path")

    # Parse the command line arguments
    args = parser.parse_args()

    if args.image and not args.video:
        process_image(args.input_file, args.output_file)
    elif args.video and not args.image:
        process_video(args.input_file, args.output_file)
    else:
        print("Please specify either -i for image or -v for video processing.")

if __name__ == "__main__":
    main()