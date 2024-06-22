# Ultralytics YOLOv5 Installation and Custom Model Training

[![Ubuntu](https://img.shields.io/badge/Ubuntu-22.04-orange?style=flat-square&logo=ubuntu&logoColor=white)](https://ubuntu.com/)
[![Windows](https://img.shields.io/badge/Windows-11-blue?style=flat-square&logo=windows&logoColor=white)](https://www.microsoft.com/windows/)
[![Python](https://img.shields.io/badge/Python-3.12-blue?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8.3-FF5733?style=flat-square&logo=python&logoColor=white)](https://matplotlib.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26.4-4C65AF?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-brightgreen?style=flat&logo=opencv&logoColor=white)](https://opencv.org/)
[![Nvidia](https://img.shields.io/badge/Nvidia-GPU-76B900?style=flat-square&logo=nvidia&logoColor=white)](https://www.nvidia.com/)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv5-0099E5?style=flat-square&logo=yolo&logoColor=white)](https://ultralytics.com/)

## Table of Contents

1. [Installing YOLOv5](#installing-yolov5)
2. [Labeling Images using LabelImg](#labeling-images-using-labelimg)
3. [Fine-Tuning a Custom Model](#fine-tuning-a-custom-model)
4. [References](#references)

## Installing YOLOv5

YOLOv5 is an unofficial extension of YOLOv4. Follow the steps below to install and set up YOLOv5.

1. **Clone the YOLOv5 repository:**
    ```sh
    git clone https://github.com/ultralytics/yolov5.git
    ```
2. **Navigate to the YOLOv5 directory:**
    ```sh
    cd yolov5
    ```
3. **Install the required dependencies:**
    ```sh
    pip install -r requirements.txt
    ```

## Labeling Images using LabelImg

LabelImg is a graphical image annotation tool used to label images for object detection.

1. **LabelImg for Windows:**
    - Download from [LabelImg releases](https://github.com/HumanSignal/labelImg/releases/tag/v1.8.1).

2. **Label your images:**
    - Open the images you want to label in LabelImg.
    - Draw bounding boxes around the objects of interest.
    - Save the annotations in YOLO format.

3. **Save your labeled images and annotations in a directory structure as follows:**
    ```
    dataset/
        test/
            images/
            labels/
        train/
            images/
            labels/
        valid/
            images/
            labels/
    ```

## Fine-Tuning a Custom Model

To fine-tune a custom YOLOv5 model, follow these steps:

1. **Create your dataset:**
    - Ensure your dataset is structured correctly with images and labels in separate subdirectories for training,
      validation, and testing.

2. **Create a `custom_data.yaml` file inside the `./data/yolov5/data/` directory:**
   ```yaml
   path: ../data/<dataset_name>
   train: train/images
   val: valid/images
   test: test/images
   nc: <number_of_classes>
   names: [ "Class1", "Class2", ..., "ClassN" ]
   ```
   Replace `<dataset_name>` and `<number_of_classes>` with the dataset name and the number of object classes and list
   the names of the classes.

3. **Navigate to the YOLOv5 directory:**
    ```sh
    cd yolov5
    ```
4. **Train the model:**
    - From scratch:
   ```sh
    python train.py --img 640 --batch 16 --epochs 100 --data custom_data.yaml --weights yolov5s.pt --workers 2
    ```
    - Fine-Tuning:
   ```sh
    python train.py --img 640 --batch 16 --epochs 100 --data custom_data.yaml --weights yolov5s.pt --workers 2 --freeze 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
    ```

5. **Use the custom-trained model:**
    ```python
   import torch
   model = torch.hub.load(repo_or_dir="ultralytics/yolov5", model="custom", path="yolov5/runs/train/exp2/weights/best.pt", force_reload=True)
   model.eval() 
   ```

6. **Test:**
    ```sh
    python detect.py --source <image/video path> --weights <path-to-weights>
    ```

## References:

- [YOLOv5 Custom Training Tutorial](https://youtu.be/tFNJGim3FXw?si=ddAqr0_a3LRprrmn)
- [YOLOv4 Custom Training Tutorial pt.1](https://youtu.be/sKDysNtnhJ4?si=ZgbfuZbKLUEplbKT)
- [YOLOv4 Custom Training Tutorial pt.2](https://youtu.be/-NEB5P-SLi0?si=8WIVPUN_sXoTQORX)
- [YOLOv4 configuration files](https://youtu.be/dxUvidAfEuE?si=ARByEFYkeo3EvAmw)

