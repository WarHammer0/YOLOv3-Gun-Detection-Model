# YOLOv3-Gun-Detection-Model

My motivation is to be proactive and address the open issues faced during an aftermath of a shootout. I conceptualize a shootout countermeasure system which detects handheld weapons before the shooter tries to use them. My design will leverage on the current infrastructure e.g. surveillance cameras, web cams etc. which are already deployed and are situated in the public places like schools, restaurants, and other public buildings. Detection and location of a handheld weapon will be based on object detection and classification using image/video feeds (using a visible camera or an infrared one). My current implementation relies heavily on visible imagery, but a similar approach could be used for short or long wave infrared camera systems for detecting concealed weapons. Since time between weapon detection and taking an action is of quintessential importance for an effective countermeasure plan, the system requirements need to have a near real time detection methodology.


Summary of my system requirements:

a)	Should be able to detect a weapon in a scene, in near real time basis.

b)	Should have a small footprint in terms of memory usage, CPU consumption.

c)	Should have ability to be custom trained as and when needed.

d)	Should be able to work on Open Source software as much as possible in order to have smaller production cost and stay intellectual property free.

e)	Should be deployable on the current surveillance systems available in the market e.g webcams, IP cameras etc.

f)	Should have zero false negative detections because, we never want to miss an actual weapon in the scene. In other words, Recall should be 1.

g)	Should have small amount of false positive detections, as we do not want to act when there is no need to do so. In other words, system’s precision should be greater than 95%.   

DATA DESCRIPTION


To train a neural network, make successful detections and then classify them appropriately, I needed a large imagery data set of commonly available handheld weapons. In addition to that, I also needed ground truth, in other words, we needed labelled images which clearly show a region in the image with a bounding box around the weapon. 

Our dataset was downloaded from the following locations:

a)	Soft Computing and Intelligent Information Systems: We used 3000 gun images from here: https://sci2s.ugr.es/sites/default/files/files/TematicWebSites/WeaponsDetection/BasesDeDatos/WeaponS.zip

b)	For testing and validation, we used 608 images (out of which, 304 are gun images) from here: https://sci2s.ugr.es/sites/default/files/files/TematicWebSites/WeaponsDetection/BasesDeDatos/Test.zip

YOLO is implemented on DARKNET which is a C based implementation of CNNs (for REF: https://pjreddie.com/media/files/papers/YOLOv3.pdf)

To get best of both the worlds, I used hybrid deep learning technique, a mix of transfer and classical training from scratch approaches:
1)	I used YOLO’s C version (for faster execution) with a pretrained model which did not have “gun” as a class.
2)	I trained the model from scratch after creating a new class called “gun” and using our image dataset meant for training as described in Section 2.2 a) above. I used Google COLAB’s PRO version for faster training using NVIDIA TESLA P100 GPGPU.
(REF: https://medium.com/@quangnhatnguyenle/how-to-train-yolov3-on-google-colab-to-detect-custom-objects-e-g-gun-detection-d3a1ee43eda1)
3)	As per the guidance provided by YOLO makers here: https://github.com/AlexeyAB/darknet#when-should-i-stop-training, We stopped our model training process when we witnessed that average loss (error) was low enough (<= 0.6) and maintained its value across different batches and iterations. 
4)	After successful training, I used NVIDIA CUDA (Compute Unified Device Architecture), cuDNN (CUDA Deep Neural Network library), OpenCV (Open Source Computer Vision) libraries to run our inference code (code to utilize the trained model for static or live gun detection on an image or webcam feed) written in Python and executed in Anaconda’s SPYDER framework. 
5)	OpenCV (https://opencv.org/) backend was used to read/write images and video files.


To get complete access to the Data and the Trained Model, Please download the Folder named, "YOLOv3GunDEtectionModel" from the following link:
https://drive.google.com/drive/folders/142znBd5PZHXqrTWXC1WOQzyyWiLhgn5_?usp=sharing

Please go through the Project Report for complete instructions and all the software and hardware requirements.


