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

In the **models folder** ,go to research folder ,then go to the **object_detection** folder and download files **export_tflite_graph_tf2.py**, **exporter_main_v2.py** and **model_main_tf2.py**

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196510913-9edc9d28-423c-4e76-ae88-cc794618cd15.png">

### *COCO API installation*
