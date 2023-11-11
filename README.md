# site-surveillance-cv

## Setup the virtual environment
```python3 -m venv venv```

## Enable virtual environment
```source venv/bin/activate```

## Install python packages
```pip install -r requirements.txt```

## Copy the HAAR Cascade files to root of project for easy access in code
```
mkdir haarcascades
cp venv/lib/cv2/data/* ./haarcascades
```

## Download YOLO files to root of project for easy access in code
```
wget https://pjreddie.com/media/files/yolov3.weights
wget https://opencv-tutorial.readthedocs.io/en/latest/_downloads/10e685aad953495a95c17bfecd1649e5/yolov3.cfg
wget https://opencv-tutorial.readthedocs.io/en/latest/_downloads/a9fb13cbea0745f3d11da9017d1b8467/coco.names
```

## Run the code
### Analyze image
```python3 cps843_project.py -i <image_to_analyze> -o <output_image>```
### Analyze video
```python3 cps843_project.py -v <video_to_analyze> -o <output_video>```