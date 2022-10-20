# DADS7202HW02

## **Introduction**
This project is for **Image Object Detection** on Durian Dataset, using  Google's Machine Learning library, Tensorflow, Two Deep Learning Convolutional Neural Network models, which are Resnet 101, and Faster R CNN.

According to these models, pre-trained on the Common Objects in Context (COCO) dataset, which doesn’t contains Durian pictures. Therefore this project required finetuning and training the models on Durian Dataset.

And another objective of this project is to compare by using precision scores between the models, pre-trained, and after finetuning both the Resnet 101 model and Faster R CNN model. 

## **Step 1: Prepare Dataset**  
Because this model is the Durian image detection, we must prepare the Durian dataset. The scope of the desired durian data set is Durian with rind full and no peeling off. There are two methods for preparing a collection of images.

<details>
<summary>Details</summary>

1. Scraping durian images from Google Images using the library, which will search for images based on the keywords "Durian" from Google and download them. Then the searchable image will be automatically divided into train and test folders with class folders.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196753569-3b3b3ad7-00d2-4f16-aaa9-b82ed28d3b76.png">
  
Ref: [https://www.google.com/search?q=Durian&rlz=1C1YTUH_thTH1010TH1010&sxsrf=ALiCzsZX2GVa-8q5oyA4choFQc_7X9Ir7A:1666211681021&source=lnms&tbm=isch&sa=X&ved=2ahUKEwi_s_iaku36AhVDSWwGHSm2CmMQ_AUoAXoECAIQAw&biw=1536&bih=731&dpr=1.25](https://www.google.com/search?q=Durian&rlz=1C1YTUH_thTH1010TH1010&sxsrf=ALiCzsZX2GVa-8q5oyA4choFQc_7X9Ir7A:1666211681021&source=lnms&tbm=isch&sa=X&ved=2ahUKEwi_s_iaku36AhVDSWwGHSm2CmMQ_AUoAXoECAIQAw&biw=1536&bih=731&dpr=1.25)

```python
import os
import shutil
import copy
import time

import cv2
from google.colab.patches import cv2_imshow
from PIL import Image

from skimage import io
import requests
from google_images_download import google_images_download

import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms

import matplotlib.pyplot as plt
```

```python
!pip install git+https://github.com/Joeclinton1/google-images-download.git
```

```python
def collect_data(query, number = 50, train_ratio=0.7) :
  # Remove spaces
  query = query.replace(' ','')
  classes = query.split(',')
  
  # Search and download images from google.
  response = google_images_download.googleimagesdownload()
  arguments = {'keywords' : query, 
              'limit' : number, 
              'silent_mode' : True,
              'format' : 'jpg',
              'output_directory' : 'data'}
  paths = response.download(arguments)

  # Create a folder to divide between training set and test set.
  if not os.path.isdir('data/train') :
    os.mkdir('data/train')
  if not os.path.isdir('data/test') :
    os.mkdir('data/test')
  for x in classes :
    if not os.path.isdir('data/train/'+x) :
      os.mkdir('data/train/'+x)
  
  # To divide between training set and test set.
  n_train = int(train_ratio*number)
  for x in classes :
    files = os.listdir('data/' + x)
    for i in range(n_train) :
      shutil.move('data/' + x + '/' + files[i], 'data/train/' + x + '/')
    shutil.move('data/' + x, 'data/test/')
  
  print('Complete')
```

2. Download Durian images from the internet.  
Once the Durian image data set from both methods has been obtained, select the images to be used in Annotate.  

3. Prepare the images you want to use in the folder.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196252482-65f86535-9a44-4862-95d2-8c8587fb10bb.png">

</details>

---

## **Step 2: Images Annotation And Data Augmentation**
We use the Roboflow website to annotate images and augmentation at this stage. The steps to do it are as follows.

<details>
<summary>Details</summary>

- From the folder you have prepared.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196252482-65f86535-9a44-4862-95d2-8c8587fb10bb.png">

- Go to the [Roboflow](https://roboflow.com/) website.
- Sign up for a new account then sign in.
- Create new project and select project type as object detection.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196253940-498d7780-cbde-4fd4-855c-c44ff6b354d8.png">

- Upload image to project.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196254333-26f139ce-e1d9-47d9-a04f-bde69b2a242d.png">

- Finishing upload.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196255979-447657ef-52f8-415e-91e5-7a3bca21078a.png">

- Annotate images.
  - Click on the image to annotate, then drag a frame around the object's area and classify it.
  - Go to the next image and repeat previous step to all images.
  
   <img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196256383-e0a00bc0-78f0-4293-bd19-f2fbfc39ee3a.png">

- Add images to dataset.
  - Click the **Add n images to Dataset** at the top right of the website.
  - Choose the method adjust required value, then click add images.
 
   <img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196257692-58a054ec-7793-4116-879d-3d38a442be8d.png">

- Generate new version of dataset.
  - Click the generate button at the bottom left of the screen.
  - In Section 3.Preprocessing Can be used to resize the image, and in Section 4. It can be used to do Augmentation and then generate a new dataset immediately in Section 5.

   <img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196258333-54abf1fe-a431-4845-b135-143d18e7ffd9.png">

- Export dataset.

   <img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196258535-d99d1a9b-ecd7-42d6-8611-cdca47147589.png">

</details>

---

## **Step 3: Prepare the environment**

Get the environment is `GPU 0: A100-SXM4-40GB (UUID: GPU-97cd2fcd-6af8-7668-6823-d5e2473eb828)`.

<details>
<summary>Details</summary>

```python
!nvidia-smi -L
```

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196757909-82128152-46e5-468e-829f-0052efdabeef.png">

```python
import sys
print( f"Python {sys.version}\n" )

import numpy as np
print( f"NumPy {np.__version__}\n" )

import matplotlib.pyplot as plt
%matplotlib inline

import tensorflow as tf
print( f"TensorFlow {tf.__version__}" )
print( f"tf.keras.backend.image_data_format() = {tf.keras.backend.image_data_format()}" )

# Count the number of GPUs as detected by tensorflow
gpus = tf.config.list_physical_devices('GPU')
print( f"TensorFlow detected { len(gpus) } GPU(s):" )
for i, gpu in enumerate(gpus):
  print( f".... GPU No. {i}: Name = {gpu.name} , Type = {gpu.device_type}" )
```

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196757486-ae014929-9763-456f-8d1d-90cfb5cbabaa.png">

</details>

---

## **STEP 4: Installation and setup**  
Before we start, we need to Install and set up the prerequisites that are essential to proceed towards object detection.  

<details>
<summary>Install Tensorflow</summary>

```python
!pip install tensorflow-gpu
```

Check `Tensorflow version is 2.9.2` :
```python
import tensorflow as tf
print(tf.__version__)
```

</details>

<details>
<summary>Cloning TFOD 2.0 Github to drive</summary>

We create a project directory named ‘DADS7202’, this folder contains materials of TensorFlow Object detection models which download and extract from TensorFlow Model Garden repository [Tensorflow models](https://github.com/tensorflow/models), we cloned this repository to the local machine, a new folder named ‘models’ is in the project directory. 

1. Mounting Google Drive.

```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Go to the folder symbol on the left side of the screen.
3. Create the folder you want to clone the repository into.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196510070-ba902de7-9c4b-4f42-9003-ae7fefddeedf.png">

4. Change directory to your folder.

```python
cd /content/drive/MyDrive/DADS7202
```

```python
!git clone https://github.com/tensorflow/models.git
```

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196786008-0f7faa81-2da7-4883-8d6a-192e2f5e024b.png">

5. Cloning Tensorflow github repository.
6. In the created folder, you will see a new folder.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196510281-b8e23663-5209-4b88-957e-5438d5b4bbab.png">

7. In the **models folder**, go to **research** folder, then go to the **object_detection** folder and download files **export_tflite_graph_tf2.py**, **exporter_main_v2.py** and **model_main_tf2.py**

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196510913-9edc9d28-423c-4e76-ae88-cc794618cd15.png">

</details>

<details>
<summary>COCO API installation</summary>
  
We need to install COCO API separately because it doesn't go directly with the Object Detection API.

1. Change directory to **research** folder.

```python
cd /content/drive/MyDrive/DADS7202/models/research
```

2. Install COCO API.

```python
!protoc object_detection/protos/*.proto --python_out=.
```
```python
!git clone https://github.com/cocodataset/cocoapi.git
```

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196786387-03ed5c8f-9c55-4fd2-9114-9d9d715f15c9.png">

```python
cd cocoapi/PythonAPI
```
```python
!make
```
```python
cp -r pycocotools /content/drive/MyDrive/DADS7202/models/research
```

</details>

<details>
<summary>Object Detection API installation</summary>

1. Back to **research** folder.

```python
cd /content/drive/MyDrive/DADS7202/models/research
```

2. Installing the object detection package.

```python
cp object_detection/packages/tf2/setup.py .
```
> #python -m pip install --use-feature=2020-resolver .  
```python
!python -m pip install .
```

3. Test Installation</summary>

```python
!python object_detection/builders/model_builder_tf2_test.py
```

</details>

<details>
<summary>Preparing the Workspace</summary>

Transforming each dataset is required (training, validation, and testing). We should transform our dataset into the [TFRecord format](https://www.tensorflow.org/tutorials/load_data/tfrecord?hl=en), which is a simple format for storing a sequence of binary records.
  
1. In the DADS7202 folder (or other created folder in cloning TFOD github step), create a workspace folder and a subfolder, as shown below.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196512776-d4ddc051-f929-4215-b592-744c34821783.png">

We will use the workspace folder to store all of the model-related attributes, including data. 

The TensorFlow Object Detection API needs A Label Map file is a simple .txt file, It links labels to some integer values, for training and detection purposes.
  
2. In the **annotations** folder right click + new file create **label_map.pbtxt**
  - Double click on label_map.pbtxt and edit the label.
  
  <img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196512933-eea5a62b-1a52-45f5-b059-84c77e1fba5d.png">

3. In the **test** and **train** folders, upload the images to use train and test the model. In this work uses JPG+XML files.

  <img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196513043-0177688f-0c77-428a-ad49-785537b3acd5.png">

4. Change directory to pre-trained-models folder.

```python
cd /content/drive/MyDrive/DADS7202/workspace/training_demo/pre-trained-models
```

</details>

---

## **STEP 5: Training Custom Object Detector**

### **Model: SSD ResNet101 V1 FPN 640x640 (RetinaNet101)**

RetinaNet-101 Feature Pyramid Net Trained on MS-COCO Data, is a single-stage object detection model that goes straight from image pixels to bounding box coordinates and class probabilities. It is able to exceed the accuracy of the best two-stage detectors while offering comparable speed performance to that of the single-stage detectors. The model architecture is based on a Feature Pyramid Network on top of a feedforward ResNet-101 backbone. The model has been trained using a new loss function, "Focal Loss", which addresses the imbalance between foreground and background classes that arises within single-stage detectors.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196790605-ccae3683-0ebc-44b9-844f-a990f18d437c.png">

Ref: [https://resources.wolframcloud.com/NeuralNetRepository/resources/RetinaNet-101-Feature-Pyramid-Net-Trained-on-MS-COCO-Data/](https://resources.wolframcloud.com/NeuralNetRepository/resources/RetinaNet-101-Feature-Pyramid-Net-Trained-on-MS-COCO-Data/)

<details>
<summary>Details</summary>

1. Download Pre-Trained Model which are listed in [TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) because in this work we try to use Pre-Train model as **SSD ResNet101 V1 FPN 640x640 (RetinaNet101)**.

To use this model, we will start from extract pre-trained-model folder.
```python
!wget http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz
```

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196790933-e98f4fa9-5d0c-43e7-b918-44fe99776960.png">

2. Extracted our pre-trained model and The **pre-trained-model** folder should look like this.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196516271-a753f502-a217-41ec-b39a-a6ae2258592a.png">

```python
!tar -xvf ssd_resnet101_v1_fpn_640x640_coco17_tpu-8.tar.gz
```

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196791279-76569959-6662-4007-9c41-d7992c0e147e.png">

3. In the **training-demo** folder, upload the previously downloaded files **export_tflite_graph_tf2.py**, **exporter_main_v2.py** and **model_main_tf2.py**.
  - This step is for easier to call this script.
  - Able to call the script in research folder without download and re-upload step.
  
  <img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196516574-10db4ccb-d2dd-48a7-b4a3-f62f01dc9d53.png">

4. Download **partition_dataset.py** and **generate_tfrecord.py**
  - Go to [TensorFlow 2 Object Detection API tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html).
  - Download **Partition Dataset script**, then partition the Dataset. (In this work we skip this step because we preprocessing dataset on [Roboflow](https://roboflow.com/) already.)
  - Download **Generate TensorFlow Records script**.
  - Upload file into **training_demo** folder.
    
  <img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196516987-a58c2eb9-a3a3-48eb-8617-2ab2ab39d39d.png">

5. Create TensorFlow Records.
  - Change directory to **training_demo**.
  - Run **generate_tfrecord.py** script to create tensorflow records.
  - Check the train folder, test folder, label_map.pbtxt and the ourput path before running.
  
```python
cd /content/drive/MyDrive/DADS7202/workspace/training_demo
```

> Create train data:
```python
!python generate_tfrecord.py -x /content/drive/MyDrive/DADS7202/workspace/training_demo/images/train -l /content/drive/MyDrive/DADS7202/workspace/training_demo/annotations/label_map.pbtxt -o /content/drive/MyDrive/DADS7202/workspace/training_demo/annotations/train.record
```

> Create test data:
```python
!python generate_tfrecord.py -x /content/drive/MyDrive/DADS7202/workspace/training_demo/images/test -l //content/drive/MyDrive/DADS7202/workspace/training_demo/annotations/label_map.pbtxt -o /content/drive/MyDrive/DADS7202/workspace/training_demo/annotations/test.record
```

6. The annotations folder should be look like this.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196517704-15e14d48-0aa9-4872-8ca0-1a3ca4df162e.png">

7. In **models** folder **(inside training_demo folder)** create a new directory named **my_ssd_resnet101_v1_fpn** and download **pipeline.config** from **pre-train-models/ssd_resnet101...**, then re-upload to the newly created directory. Our **training_demo** should now look like this:

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196517998-799e438a-4ee8-4836-89de-79ed5746e519.png">

8. Configure the Training Pipeline.
  - Double click into pipeline.config in model/my_ssd_resnet101_v1_fpn
  - **Looking at line 3, let's change the number of different label classes.**
  - Line 6, 7 can set image resizer height and width.
  - **Line 131 to set batch size**.
  - Line 136 to set augmentation options.
  - **Line 161 change the Path to checkpoint of pre-trained model**.
  - **Line 152, 162 change number of step**.
  - **Line 167 change fine tune checkpoint type to detection**.
  - **Line 168 set it to false**.
  - **Line 172 change Path to label map file**.
  - **Line 174 change Path to training TFRecord file**.
  - **Line 182 change Path to label map file**.
  - **Line 186 change Path to testing TFRecord**.

This model pre-trained on the Common Objects in Context (COCO) dataset, which don’t contains Durian pictures. Therefore this project required configuring and training the models on Durian Dataset.

The TensorFlow Object Detection API allows model configuration via the pipeline.config file that goes along with the pre-trained model. 

For This Model, we play around with different setups to test things out and get the best model performance. `As the following model parameters`:

- Num_classes (int) : 1 :arrow_right: Because we only detect Durian, we set up as 1. 
- Batch_size (int) : 8 :arrow_right: The batch size number must be divisible by 2 and due to the constrain of available memory, so we choose 8. 
- Fine_tune_checkpoint (str): put a path to the pre-trained model of ResNet101 V1 FPN 640x640 (RetinaNet101)  model checkpoint. 
- Fine_tune_checkpoint_type (str): set to detection because we want to train a detection model.
- Use_bfloat16 (boolean): set to ‘false’ cause we are not going to train a model on a TPU.
- Label_map_path (str): provide a path to the **label_map.pbtxt** was created previously. 
- Train_input_reader (str): set a path to training TFRecord file train_input_reader.
- Eval_input_reader(str): set a path to testing TFRecord file.

### *For The Steps of Training. We try several numbers, Begin from 5,000 steps.*
*<b>Number of steps: 2,000 , 5,000 , 10,000</b>* 
> Cause it depends on our resources as memory and GPU

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196518462-62833952-1c11-4428-8d16-06d8fc907e26.png">
  
</details>

### **Model: Faster R-CNN ResNet50 V1 640x640**

Faster_R-CNN-ResNet50_V1 is a single-stage object detection model and the architecture of this model is complex because it has several moving parts.  

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196794462-950de22a-3f18-4c62-9a88-db5bb382f3d1.png">

Ref: [https://tryolabs.com/blog/2018/01/18/faster-r-cnn-down-the-rabbit-hole-of-modern-object-detection](https://tryolabs.com/blog/2018/01/18/faster-r-cnn-down-the-rabbit-hole-of-modern-object-detection)


<details>
<summary>Details</summary>

1. Download Pre-Trained Model which are listed in [TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md) because in this work we try to use Pre-Train model as **Faster R-CNN ResNet50 V1 640x640.**.

To use this model, we will start from extract pre-trained-model folder.
```python
!wget http://download.tensorflow.org/models/object_detection/tf2/20200711/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz
```
  
<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196795446-7a18d794-7905-4587-9caf-8f30fa3a436f.png">

2. Extracted our pre-trained model and The **pre-trained-model** folder should look like this.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196734840-e66baa86-6e5f-410b-8547-95d7d3ed3b1d.png">

```
!tar -xvf faster_rcnn_resnet50_v1_640x640_coco17_tpu-8.tar.gz
```

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196795600-e572a3d7-ca22-42cc-9c1a-0eaacf80be22.png">

3. In the **training-demo** folder, upload the previously downloaded files **export_tflite_graph_tf2.py**, **exporter_main_v2.py** and **model_main_tf2.py** .
  - This step is for easier to call this script.
  - Able to call the script in research folder without download and re-upload step.
  
  <img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196741775-b9f116b9-3fe3-4c4c-b2f4-5fe6e6f3f198.png">

4. Download **partition_dataset.py** and **generate_tfrecord.py**
  - Go to [TensorFlow 2 Object Detection API tutorial](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html).
  - Download **Partition Dataset script**, then partition the dataset. (In this work we skip this step because we preprocessing dataset on [Roboflow](https://roboflow.com/) already.)
  - Download **Generate TensorFlow Records script**.
  - Upload file into **training_demo** folder.

  <img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196742294-9c8c173a-dd8b-4c18-adea-73c72310a4e4.png">

5. Create TensorFlow Records.
  - Change directory to **training_demo**.
  - Run **generate_tfrecord.py** script to create tensorflow records.
  - Check the train folder, test folder, label_map.pbtxt and the ourput path before running.
  
```
cd /content/drive/MyDrive/DADS7202/workspace/training_demo
```

> Create train data:
```
!python generate_tfrecord.py -x /content/drive/MyDrive/DADS7202/workspace/training_demo/images/train -l /content/drive/MyDrive/DADS7202/workspace/training_demo/annotations/label_map.pbtxt -o /content/drive/MyDrive/DADS7202/workspace/training_demo/annotations/train.record
```

> Create test data:
```
!python generate_tfrecord.py -x /content/drive/MyDrive/DADS7202/workspace/training_demo/images/test -l //content/drive/MyDrive/DADS7202/workspace/training_demo/annotations/label_map.pbtxt -o /content/drive/MyDrive/DADS7202/workspace/training_demo/annotations/test.record
```

6. The annotations folder should be look like this.

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196743145-046586bf-b85b-4aa0-b38c-3e79788437dd.png">

7. In **models** folder **(inside training_demo folder)** create a new directory named **Faster_R-CNN_ResNet50_V1** and download **pipeline.config** from **pre-train-models/faster_rcnn...**, then re-upload to the newly created directory. Our **training_demo** should now look like this:

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196743456-8a7b9c17-ab62-4530-99ed-ca8c8b1c6305.png">

8. Configure the Training Pipeline.
  - Double click into pipeline.config in **models/Faster_R-CNN_ResNet50_V1**
  - **Looking at line 10, let's change the number of different label classes.**
  - Line 13,14 can set image resizer height and width.
  - **Line 93 to set batch size.**
  - **Line 97,103 change number of step.**
  - **Line 113 change the Path to checkpoint of pre-trained model.**
  - **Line 114 change fine tune checkpoint type to detection.**
  - **Line 122 set it to false.**
  - **Line 126 change Path to label map file.**
  - **Line 128 change Path to training TFRecord file.**
  - **Line 139 change Path to label map file.**
  - **Line 143 change Path to testing TFRecord.**

### *For This Model, we tune the parameters same as SSD ResNet101 V1 FPN 640x640 (RetinaNet101), except Number of steps:*
*<b>Number of steps: 1,000, 2,000</b>*
  
  <img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196743916-572614fa-9738-44c4-967c-64cac14cab6d.png">

</details>

---

## **STEP 6:  Evaluating the Model**

### **Model: SSD ResNet101 V1 FPN 640x640 (RetinaNet101)**

table compare between pre-trained-model and after fine-tune model

Model (640x640) | Step | Batch_size | mAP  (.50) | mAP  (.50: .95) | Time (sec.)
:----: | :----: | :----: | :----: | :----: | :----: 
SSD ResNet101 V1 FPN (tuning) | 5,000 | 8 |  |  |  
SSD ResNet101 V1 FPN (tuning) | 10,000 | 8 |  |  |  


<details>
<summary>Details</summary>
  
## **Evaluating the Model**

1. Set metric type.

```python
from object_detection.protos import eval_pb2
eval_config = eval_pb2.EvalConfig()
eval_config.metrics_set.extend(['coco_detection_metrics'])
```

2. Change directory to training_demo.

```python
cd /content/drive/MyDrive/DADS7202/workspace/training_demo
```

3. Model evaluate using Tensorboard.

```python
!python model_main_tf2.py --model_dir=/content/drive/MyDrive/DADS7202/workspace/training_demo/models/my_ssd_resnet101_v1_fpn --pipeline_config_path=/content/drive/MyDrive/DADS7202/workspace/training_demo/models/my_ssd_resnet101_v1_fpn/pipeline.config --checkpoint_dir=/content/drive/MyDrive/DADS7202/workspace/training_demo/models/my_ssd_resnet101_v1_fpn
```
---output---

```
%load_ext tensorboard
%tensorboard --logdir=/content/drive/MyDrive/DADS7202/workspace/training_demo/models/my_ssd_resnet101_v1_fpn
```
---output---

## **Inferencing Trained Models.**

1. In exported-models folder create my_model folder.
2. Export the model to */content/drive/MyDrive/DADS7202/workspace/training_demo/exported-models/my_model*

```python
!python exporter_main_v2.py --input_type image_tensor --pipeline_config_path /content/drive/MyDrive/DADS7202/workspace/training_demo/models/my_ssd_resnet101_v1_fpn/pipeline.config --trained_checkpoint_dir /content/drive/MyDrive/DADS7202/workspace/training_demo/models/my_ssd_resnet101_v1_fpn --output_directory /content/drive/MyDrive/DADS7202/workspace/training_demo/exported-models/my_model
```
---output---
  
3. Inferencing trained model.

```python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2
import argparse
from google.colab.patches import cv2_imshow

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# PROVIDE PATH TO IMAGE DIRECTORY
IMAGE_PATHS = '/content/drive/MyDrive/DADS7202/workspace/training_demo/images/test/1_jpg.rf.24fda645c9751b1f97ca006a4c164020.jpg'

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = '/content/drive/MyDrive/DADS7202/workspace/training_demo/pre-trained-models/ssd_resnet101_v1_fpn_640x640_coco17_tpu-8'

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = '/content/drive/MyDrive/DADS7202/workspace/training_demo/annotations/label_map.pbtxt'

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(0.60)

# LOAD THE MODEL

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# LOAD LABEL MAP DATA FOR PLOTTING

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))




print('Running inference for {}... '.format(IMAGE_PATHS), end='')

image = cv2.imread(IMAGE_PATHS)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_expanded = np.expand_dims(image_rgb, axis=0)

# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
input_tensor = tf.convert_to_tensor(image)
# The model expects a batch of images, so add an axis with `tf.newaxis`.
input_tensor = input_tensor[tf.newaxis, ...]

# input_tensor = np.expand_dims(image_np, 0)
detections = detect_fn(input_tensor)

# All outputs are batches tensors.
# Convert to numpy arrays, and take index [0] to remove the batch dimension.
# We're only interested in the first num_detections.
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
               for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

image_with_detections = image.copy()

# SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
viz_utils.visualize_boxes_and_labels_on_image_array(
      image_with_detections,
      detections['detection_boxes'],
      detections['detection_classes'],
      detections['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=0.6,
      agnostic_mode=False)

print('Done')
# DISPLAYS OUTPUT IMAGE
cv2_imshow(image_with_detections)
# CLOSES WINDOW ONCE KEY IS PRESSED
```

> <b>Pre-trained-model SSD_resnet101_v1_fpn without training.</b>
  
`
Loading model...Done! Took 39.04430317878723 seconds
Running inference for /content/drive/MyDrive/DADS7202/workspace/training_demo/images/test/1_jpg.rf.24fda645c9751b1f97ca006a4c164020.jpg... Done
`
  
<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196521072-a133a268-62ad-418a-8372-93456b26da4d.png">

```python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2
import argparse
from google.colab.patches import cv2_imshow

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# PROVIDE PATH TO IMAGE DIRECTORY
IMAGE_PATHS = '/content/drive/MyDrive/DADS7202/workspace/training_demo/images/test/1_jpg.rf.24fda645c9751b1f97ca006a4c164020.jpg'

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = '/content/drive/MyDrive/DADS7202/workspace/training_demo/exported-models/my_model'

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = '/content/drive/MyDrive/DADS7202/workspace/training_demo/annotations/label_map.pbtxt'

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(0.60)

# LOAD THE MODEL

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# LOAD LABEL MAP DATA FOR PLOTTING

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))




print('Running inference for {}... '.format(IMAGE_PATHS), end='')

image = cv2.imread(IMAGE_PATHS)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_expanded = np.expand_dims(image_rgb, axis=0)

# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
input_tensor = tf.convert_to_tensor(image)
# The model expects a batch of images, so add an axis with `tf.newaxis`.
input_tensor = input_tensor[tf.newaxis, ...]

# input_tensor = np.expand_dims(image_np, 0)
detections = detect_fn(input_tensor)

# All outputs are batches tensors.
# Convert to numpy arrays, and take index [0] to remove the batch dimension.
# We're only interested in the first num_detections.
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
               for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

image_with_detections = image.copy()

# SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
viz_utils.visualize_boxes_and_labels_on_image_array(
      image_with_detections,
      detections['detection_boxes'],
      detections['detection_classes'],
      detections['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=0.6,
      agnostic_mode=False)

print('Done')
# DISPLAYS OUTPUT IMAGE
cv2_imshow(image_with_detections)
# CLOSES WINDOW ONCE KEY IS PRESSED
```

> <b>With 5,000 steps of training, the results are still unsatisfactory. After this, try 10,000 training steps.</b>
  
`
Loading model...Done! Took 20.713525533676147 seconds
Running inference for /content/drive/MyDrive/DADS7202/workspace/training_demo/images/test/1_jpg.rf.24fda645c9751b1f97ca006a4c164020.jpg... Done
`
  
<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196521564-44488043-7a83-4159-b702-4e7aa03443a4.png">

```python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2
import argparse
from google.colab.patches import cv2_imshow

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# PROVIDE PATH TO IMAGE DIRECTORY
IMAGE_PATHS = '/content/drive/MyDrive/DADS7202/workspace/training_demo/images/test/1_jpg.rf.24fda645c9751b1f97ca006a4c164020.jpg'

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = '/content/drive/MyDrive/DADS7202/workspace/training_demo/exported-models/my_model2'

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = '/content/drive/MyDrive/DADS7202/workspace/training_demo/annotations/label_map.pbtxt'

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(0.60)

# LOAD THE MODEL

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# LOAD LABEL MAP DATA FOR PLOTTING

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))




print('Running inference for {}... '.format(IMAGE_PATHS), end='')

image = cv2.imread(IMAGE_PATHS)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_expanded = np.expand_dims(image_rgb, axis=0)

# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
input_tensor = tf.convert_to_tensor(image)
# The model expects a batch of images, so add an axis with `tf.newaxis`.
input_tensor = input_tensor[tf.newaxis, ...]

# input_tensor = np.expand_dims(image_np, 0)
detections = detect_fn(input_tensor)

# All outputs are batches tensors.
# Convert to numpy arrays, and take index [0] to remove the batch dimension.
# We're only interested in the first num_detections.
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
               for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

image_with_detections = image.copy()

# SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
viz_utils.visualize_boxes_and_labels_on_image_array(
      image_with_detections,
      detections['detection_boxes'],
      detections['detection_classes'],
      detections['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=0.6,
      agnostic_mode=False)

print('Done')
# DISPLAYS OUTPUT IMAGE
cv2_imshow(image_with_detections)
# CLOSES WINDOW ONCE KEY IS PRESSED
```
  
> <b>By increasing the steps to 10000, the results look better.</b>
  
`
Loading model...Done! Took 21.366928339004517 seconds
Running inference for /content/drive/MyDrive/DADS7202/workspace/training_demo/images/test/1_jpg.rf.24fda645c9751b1f97ca006a4c164020.jpg... Done
`

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196521952-64671dc6-717c-4ef9-a38a-93bed9b177e5.png">
   
</details>

### **Model: Faster R-CNN ResNet50 V1 640x640**

table compare between pre-trained-model and after fine-tune model

Model (640x640) | Step | Batch_size | mAP  (.50) | mAP  (.50: .95) | Time (sec.)
:----: | :----: | :----: | :----: | :----: | :----:
Faster R-CNN ResNet50 V1 (tuning) | 1,000 | 8 |  |  |  
Faster R-CNN ResNet50 V1 (tuning) | 2,000 | 8 |  |  |  

<details>
<summary>Details</summary>
  
## **Evaluating the Model**

1. Set metric type.

```python
from object_detection.protos import eval_pb2
eval_config = eval_pb2.EvalConfig()
eval_config.metrics_set.extend(['coco_detection_metrics'])
```
  
2. Change directory to training_demo

```python
cd /content/drive/MyDrive/DADS7202/workspace/training_demo
```

3. Model evaluate using Tensorboard

```python
!python model_main_tf2.py --model_dir=/content/drive/MyDrive/DADS7202/workspace/training_demo/models/Faster_R-CNN_ResNet50_V1 --pipeline_config_path=/content/drive/MyDrive/DADS7202/workspace/training_demo/models/Faster_R-CNN_ResNet50_V1/pipeline.config --checkpoint_dir=/content/drive/MyDrive/DADS7202/workspace/training_demo/models/Faster_R-CNN_ResNet50_V1
```
---output---
  
```python
%load_ext tensorboard
%tensorboard --logdir=/content/drive/MyDrive/DADS7202/workspace/training_demo/models/Faster_R-CNN_ResNet50_V1
```
---output---
  
## **Inferencing Trained Models**
  
1. In exported-models folder create my_model folder
2. Export the model to */content/drive/MyDrive/DADS7202/workspace/training_demo/exported-models/my_model*
  
```python
!python exporter_main_v2.py --input_type image_tensor --pipeline_config_path /content/drive/MyDrive/DADS7202/workspace/training_demo/models/Faster_R-CNN_ResNet50_V1/pipeline.config --trained_checkpoint_dir /content/drive/MyDrive/DADS7202/workspace/training_demo/models/Faster_R-CNN_ResNet50_V1 --output_directory /content/drive/MyDrive/DADS7202/workspace/training_demo/exported-models/my_model3_faster_R_CNN_1000
```
---output---
  
3. Inferencing trained model

```python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2
import argparse
from google.colab.patches import cv2_imshow

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# PROVIDE PATH TO IMAGE DIRECTORY
IMAGE_PATHS = '/content/drive/MyDrive/DADS7202/workspace/training_demo/images/test/1_jpg.rf.24fda645c9751b1f97ca006a4c164020.jpg'

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = '/content/drive/MyDrive/DADS7202/workspace/training_demo/pre-trained-models/faster_rcnn_resnet50_v1_640x640_coco17_tpu-8'

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = '/content/drive/MyDrive/DADS7202/workspace/training_demo/annotations/label_map.pbtxt'

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(0.60)

# LOAD THE MODEL

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# LOAD LABEL MAP DATA FOR PLOTTING

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))




print('Running inference for {}... '.format(IMAGE_PATHS), end='')

image = cv2.imread(IMAGE_PATHS)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_expanded = np.expand_dims(image_rgb, axis=0)

# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
input_tensor = tf.convert_to_tensor(image)
# The model expects a batch of images, so add an axis with `tf.newaxis`.
input_tensor = input_tensor[tf.newaxis, ...]

# input_tensor = np.expand_dims(image_np, 0)
detections = detect_fn(input_tensor)

# All outputs are batches tensors.
# Convert to numpy arrays, and take index [0] to remove the batch dimension.
# We're only interested in the first num_detections.
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
               for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

image_with_detections = image.copy()

# SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
viz_utils.visualize_boxes_and_labels_on_image_array(
      image_with_detections,
      detections['detection_boxes'],
      detections['detection_classes'],
      detections['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=0.5,
      agnostic_mode=False)

print('Done')
# DISPLAYS OUTPUT IMAGE
cv2_imshow(image_with_detections)
# CLOSES WINDOW ONCE KEY IS PRESSED
```

> <b>Pre-trained-model Faster R-CNN without training.</b>

`
Loading model...Done! Took 9.365224361419678 seconds
Running inference for /content/drive/MyDrive/DADS7202/workspace/training_demo/images/test/1_jpg.rf.24fda645c9751b1f97ca006a4c164020.jpg... Done
`

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196767560-f524d772-dea6-4c40-8a72-a8129f59b175.png">


```python

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2
import argparse
from google.colab.patches import cv2_imshow

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# PROVIDE PATH TO IMAGE DIRECTORY
IMAGE_PATHS = '/content/drive/MyDrive/DADS7202/workspace/training_demo/images/test/1_jpg.rf.24fda645c9751b1f97ca006a4c164020.jpg'

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = '/content/drive/MyDrive/DADS7202/workspace/training_demo/exported-models/my_model3_faster_R_CNN_1000'

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = '/content/drive/MyDrive/DADS7202/workspace/training_demo/annotations/label_map.pbtxt'

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(0.60)

# LOAD THE MODEL

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# LOAD LABEL MAP DATA FOR PLOTTING

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))




print('Running inference for {}... '.format(IMAGE_PATHS), end='')

image = cv2.imread(IMAGE_PATHS)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_expanded = np.expand_dims(image_rgb, axis=0)

# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
input_tensor = tf.convert_to_tensor(image)
# The model expects a batch of images, so add an axis with `tf.newaxis`.
input_tensor = input_tensor[tf.newaxis, ...]

# input_tensor = np.expand_dims(image_np, 0)
detections = detect_fn(input_tensor)

# All outputs are batches tensors.
# Convert to numpy arrays, and take index [0] to remove the batch dimension.
# We're only interested in the first num_detections.
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
               for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

image_with_detections = image.copy()

# SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
viz_utils.visualize_boxes_and_labels_on_image_array(
      image_with_detections,
      detections['detection_boxes'],
      detections['detection_classes'],
      detections['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=0.5,
      agnostic_mode=False)

print('Done')
# DISPLAYS OUTPUT IMAGE
cv2_imshow(image_with_detections)
# CLOSES WINDOW ONCE KEY IS PRESSED
```

> <b>Fater R-CNN as 1,000 Step.</b>

`
Loading model...Done! Took 8.798496007919312 seconds
Running inference for /content/drive/MyDrive/DADS7202/workspace/training_demo/images/test/1_jpg.rf.24fda645c9751b1f97ca006a4c164020.jpg... Done
`

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196767967-65d71df4-0692-4e87-b51b-260b9801db1b.png">

```python
  
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import cv2
import argparse
from google.colab.patches import cv2_imshow

# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# PROVIDE PATH TO IMAGE DIRECTORY
IMAGE_PATHS = '/content/drive/MyDrive/DADS7202/workspace/training_demo/images/test/1_jpg.rf.24fda645c9751b1f97ca006a4c164020.jpg'

# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = '/content/drive/MyDrive/DADS7202/workspace/training_demo/exported-models/my_model4_faster_R_CNN_2000'

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = '/content/drive/MyDrive/DADS7202/workspace/training_demo/annotations/label_map.pbtxt'

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(0.60)

# LOAD THE MODEL

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))

# LOAD LABEL MAP DATA FOR PLOTTING

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    return np.array(Image.open(path))




print('Running inference for {}... '.format(IMAGE_PATHS), end='')

image = cv2.imread(IMAGE_PATHS)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_expanded = np.expand_dims(image_rgb, axis=0)

# The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
input_tensor = tf.convert_to_tensor(image)
# The model expects a batch of images, so add an axis with `tf.newaxis`.
input_tensor = input_tensor[tf.newaxis, ...]

# input_tensor = np.expand_dims(image_np, 0)
detections = detect_fn(input_tensor)

# All outputs are batches tensors.
# Convert to numpy arrays, and take index [0] to remove the batch dimension.
# We're only interested in the first num_detections.
num_detections = int(detections.pop('num_detections'))
detections = {key: value[0, :num_detections].numpy()
               for key, value in detections.items()}
detections['num_detections'] = num_detections

# detection_classes should be ints.
detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

image_with_detections = image.copy()

# SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
viz_utils.visualize_boxes_and_labels_on_image_array(
      image_with_detections,
      detections['detection_boxes'],
      detections['detection_classes'],
      detections['detection_scores'],
      category_index,
      use_normalized_coordinates=True,
      max_boxes_to_draw=200,
      min_score_thresh=0.5,
      agnostic_mode=False)

print('Done')
# DISPLAYS OUTPUT IMAGE
cv2_imshow(image_with_detections)
# CLOSES WINDOW ONCE KEY IS PRESSED
```

> <b>Fater R-CNN as 2,000 Step.</b>

`
Loading model...Done! Took 8.783716917037964 seconds
Running inference for /content/drive/MyDrive/DADS7202/workspace/training_demo/images/test/1_jpg.rf.24fda645c9751b1f97ca006a4c164020.jpg... Done
`

<img width="500" alt="image" src="https://user-images.githubusercontent.com/97492504/196771728-79acf142-cf3a-40ac-b14b-460f21b6ba66.png">

</details>

---

## **Dissussion**

---


## **Conclusion**

Model (640x640) | Step | Batch_size | mAP  (.50) | mAP  (.50: .95) | Time (sec.)
:----: | :----: | :----: | :----: | :----: | :----:
SSD ResNet101 V1 FPN | 5,000 | 8 | 0.784264 | 0.390598 | 0.343 | 
SSD ResNet101 V1 FPN | 10,000 | 8 | 0.940915 | 0.395898 | 0.339 | 
Faster R-CNN ResNet50 V1 | 1,000 | 8 | 0.951654 | 0.449221 | 0.253 | 
Faster R-CNN ResNet50 V1 | 2,000 | 8 | 0.933264 | 0.499617 | 0.249 | 

---

## **Member**  
6410412002  Mr. Kittipat Pattarajariya  
6410412004  Miss Chonthanya Yosbuth  
6410412010  Mr. Saran Ditjarern  
6410412018  Miss Panumas sitthikarn  

---

## **Reference**  

---

## **End Credit**  
This work is part of the DADS7202 Deep Learning in Master degree of Science at Faculty of Applied Statistics National Institute of Development Administration.

---
