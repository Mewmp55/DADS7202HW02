# DADS7202HW02

## This work is to try to create an object detection model using Tensorflow, following the steps of [TensorFlow 2 Object Detection API tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html) and YouTube Chanel [Krish Naik](https://www.youtube.com/watch?v=XoMiveY_1Z4) video.

### *Prepare DATASET*
- Prepare the images you want to use in the folder.

   <img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196252482-65f86535-9a44-4862-95d2-8c8587fb10bb.png">

- Go to the roboflow website.
- Sign up for a new account then sign in.
- Create new project and select project type as object detection.

   <img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196253940-498d7780-cbde-4fd4-855c-c44ff6b354d8.png">

- Upload image to project

   <img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196254333-26f139ce-e1d9-47d9-a04f-bde69b2a242d.png">

- Finishing upload

   <img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196255979-447657ef-52f8-415e-91e5-7a3bca21078a.png">

- Annotate images
  - Click on the image to annotate, then drag a frame around the object's area and classify it.
  - Go to the next image and repeat previous step to all images.
  
   <img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196256383-e0a00bc0-78f0-4293-bd19-f2fbfc39ee3a.png">

- Add images to dataset
  - Click the Add n images to Dataset at the top right of the website.
  - Choose the method adjust required value ,then click add images.
 
   <img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196257692-58a054ec-7793-4116-879d-3d38a442be8d.png">

- Generate new version of Dataset
  - Click the generate button at the bottom left of the screen.
  - In Section 3.Preprocessing Can be used to resize the image, and in Section 4. It can be used to do Augmentation and then generate a new dataset immediately in Section 5.

   <img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196258333-54abf1fe-a431-4845-b135-143d18e7ffd9.png">

- Export dataset

   <img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196258535-d99d1a9b-ecd7-42d6-8611-cdca47147589.png">

### *Install Tensorflow*
```
!pip install tensorflow-gpu
```
---output---

```
import tensorflow as tf
print(tf.__version__)
```
---output---

```
!nvidia-smi -L
```
---output---

### *Cloning TFOD 2.0 Github to drive*

- Mounting Google Drive

```
from google.colab import drive
drive.mount('/content/drive')
```
---output---

- Go to the folder symbol on the left side of the screen.
- Create the folder you want to clone the repository into.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196510070-ba902de7-9c4b-4f42-9003-ae7fefddeedf.png">

- Change directory to your folder.

```
cd /content/drive/MyDrive/DADS7202
```
---output---

```
!git clone https://github.com/tensorflow/models.git
```
---output---

- Cloning Tensorflow github repository
- In the created folder, you will see a new folder.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196510281-b8e23663-5209-4b88-957e-5438d5b4bbab.png">

- In the **models folder** ,go to research folder ,then go to the **object_detection** folder and download files **export_tflite_graph_tf2.py**, **exporter_main_v2.py** and **model_main_tf2.py**

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196510913-9edc9d28-423c-4e76-ae88-cc794618cd15.png">

### *COCO API installation*

- Change directory to research folder

```
cd /content/drive/MyDrive/DADS7202/models/research
```
---output---

- Install coco API

```
!protoc object_detection/protos/*.proto --python_out=.
```
```
!git clone https://github.com/cocodataset/cocoapi.git
```
---output---
```
cd cocoapi/PythonAPI
```
---output---
```
!make
```
---output---
```
cp -r pycocotools /content/drive/MyDrive/DADS7202/models/research
```
---output---

### *Object Detection API installation*

- Back to **research** folder

```
cd /content/drive/MyDrive/DADS7202/models/research
```
---output---

- Installing the object detection package

```
cp object_detection/packages/tf2/setup.py .
```
---output---


> #python -m pip install --use-feature=2020-resolver .

```
!python -m pip install .
```
---output---

- Test Installation

```
!python object_detection/builders/model_builder_tf2_test.py
```
---output---

### *Training Custom Object Detector*
Preparing the Workspace

- In the DADS7202 folder (or other created folder in cloning TFOD github step), create a workspace folder and a subfolder, as shown below.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196512776-d4ddc051-f929-4215-b592-744c34821783.png">

- In the **annotations** folder right click + new file create *label_map.pbtxt*
  - double click on label_map.pbtxt and edit the label
  
  <img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196512933-eea5a62b-1a52-45f5-b059-84c77e1fba5d.png">

- In the **test** and **train** folders, upload the images to use train and test the model. In this work uses JPG+XML files.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196513043-0177688f-0c77-428a-ad49-785537b3acd5.png">

- Change directory to pre-trained-models folder

```
cd /content/drive/MyDrive/DADS7202/workspace/training_demo/pre-trained-models
```
---output---

- Download Pre-Trained Model which are listed in [TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md).
- In this work we try to use Pre-Train model as **SSD ResNet101 V1 FPN 640x640 (RetinaNet101)**.

```
!wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz
```
---output---

- Extracted our pre-trained model and The pre-trained-model folder should look like this.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196516271-a753f502-a217-41ec-b39a-a6ae2258592a.png">

```
!tar -xvf ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz
```
---output---

- In the **training-demo** folder, upload the previously downloaded files **export_tflite_graph_tf2.py**, **exporter_main_v2.py** and **model_main_tf2.py**.
  - This step is for easier to call this script.
  - Able to call the script in research folder without download and re-upload step.
  
  <img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196516574-10db4ccb-d2dd-48a7-b4a3-f62f01dc9d53.png">

- Download **partition_dataset.py** and **generate_tfrecord.py**
  - Go to [TensorFlow 2 Object Detection API tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html).
  - Download **Partition Dataset script**, then partition the Dataset. (In this work we skip this step because we preprocessing dataset on [Roboflow](https://roboflow.com/) already.)
  - Download **Generate TensorFlow Records script**.
  - Upload file into **training_demo** folder.
    <img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196516987-a58c2eb9-a3a3-48eb-8617-2ab2ab39d39d.png">

- Create TensorFlow Records.
  - Change directory to **training_demo**.
  - Run **generate_tfrecord.py** script to create tensorflow records.
  - Check the train folder, test folder, label_map.pbtxt and the ourput path before running.
  
```
cd /content/drive/MyDrive/DADS7202/workspace/training_demo
```
---output---

> Create train data:
```
!python generate_tfrecord.py -x /content/drive/MyDrive/DADS7202/workspace/training_demo/images/train -l /content/drive/MyDrive/DADS7202/workspace/training_demo/annotations/label_map.pbtxt -o /content/drive/MyDrive/DADS7202/workspace/training_demo/annotations/train.record
```

> Create test data:
```
!python generate_tfrecord.py -x /content/drive/MyDrive/DADS7202/workspace/training_demo/images/test -l //content/drive/MyDrive/DADS7202/workspace/training_demo/annotations/label_map.pbtxt -o /content/drive/MyDrive/DADS7202/workspace/training_demo/annotations/test.record
```
---output---
