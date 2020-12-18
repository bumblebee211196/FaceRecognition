# FaceRecognition

A simple Face Recognition application using python, opencv and face-recognition. For detail explanation check out my [blog].

This was possible only due to the simple and clear explanation by Adrian Rosebrock's [blog](https://www.pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/)

## How to run?

### 1. Create virtual environment

```shell
python3 -m venv venv
```

#### macOS
```shell
source venv/bin/activate
```

#### Windows
```shell
.\venv\Scripts\activate
```

### 2. Install required dependencies

```shell
pip3 install -r requirements.txt
```

### 3. Execute

#### a. To detect faces from Images
```shell
python3 -m face_recognition_images -i /path/to/image/file.jpg -m hog
```

#### b. To detect faces from Videos

###### i. To detect faces from video files
```shell
python3 -m detect_faces_from_video -v /path/to/video/file.mov -m hog
```

###### ii. To detect faces realtime using camera
```shell
python3 -m detect_faces_from_video -m hog
```
