# In[1]:

# Importing necessary libraries
import os
import cv2 as cv
import argparse
import sys
import numpy as np
import os.path
import keras
from keras.models import load_model
from keras.utils.vis_utils import plot_model
import seaborn as sns
import matplotlib.pyplot as plt

# In[2]:

# Initialize the parameters
confThreshold = 0.5  #Confidence threshold
nmsThreshold = 0.4   #Non-maximum suppression threshold

# for image input
inpWidth = 416       #Width of network's input image
inpHeight = 416      #Height of network's input image

# for webcam input
inpWidthLIVE = 640       #Width of network's input image
inpHeightLIVE = 480      #Height of network's input image

TrueImageList = []
FalseImageList = []

# In[3]:

# Set the commandline argument parser
parser = argparse.ArgumentParser(description='Object Detection using YOLO version 3 in OPENCV')
parser.add_argument('--image', help='Path to image file.')
parser.add_argument('--video', help='Path to video file.')
parser.add_argument('--live', help='live webcam input')
parser.add_argument('--test', help='test flag for scanning a directory for images to test')
parser.add_argument('--test_dirT', help='test flag has to be turned ON and this directory points to the location\
                    of images that DO have a weapon')
parser.add_argument('--test_dirF', help='test flag has to be turned ON and this directory points to the location\
                    of images that do NOT have weapon')
parser.add_argument('--classes', help='Path to YOLO v3 classes name file.', required=True)
parser.add_argument('--config', help='Path to YOLO v3 config file.', required=True)
parser.add_argument('--model', help='Path to YOLO v3 WEIGHT/MODEL file.', required=True)


# Parse the arguments from commandline
args = parser.parse_args()

# some sanity checks
if (not(args.image) and not(args.video) and not(args.live) and not(args.test)):
    print("ERROR: Please provide input file via --image OR --video OR --live OR --test and restart this application")
    sys.exit()
    
if (args.test):
    if (not(args.test_dirT) and not(args.test_dirF)):
        print("ERROR: test flag is turned ON, Please provide input test directory via --test_dirT --test_dirF \
              and restart this application")
        sys.exit()
    else:
        print('')
        print ('Running in test mode and the ACTUAL WEAPON directory is set to: ', args.test_dirT)
        print ('Running in test mode and the NON existent WEAPON directory is set to: ', args.test_dirF)
        print('')
        
# In[4]:

# Convert C based YOLOv3 weights (trained model) to KERAS for evaluation
runfile("convert.py", args = "../cfg/yolov3_custom_train.cfg  \
                              ../model/Trained/yolov3_custom_train_final.weights \
                              ../model/Trained/yolov3_custom_train_final.h5")

# In[5]:

# USE converted KERAS model and make a network plot
YOLOv3_classifier = load_model('../model/Trained/yolov3_custom_train_final.h5', compile=False)

# plot_model(YOLOv3_classifier, to_file='../results/plots/classifier_plot.png', show_shapes=True, 
#             show_layer_names=True)


# In[6]:

# Load names of classes
classesFile = args.classes
classes = None

with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = args.config
modelWeights = args.model


# In[7]:

# Let us use CUDA and CuDNN from NVIDIA to use GPGPU and accelerate our Inference time (msec)
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

# In[8]:

# Some helper function defintions
# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
    
    label = '%.2f' % conf
        
    # Get the label for the class name and its confidence
    if classes:
        assert(classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

# Remove the bounding boxes with low confidence using non-maxima suppression
def WeaponDetector(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    WeaponDetected = False

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
        WeaponDetected = True
        
    return WeaponDetected

# In[9]:

# Process inputs (OpenCV)
winName = 'Object detection in OpenCV/YOLOv3: Group 7 (ISTM:6290-10)'
cv.namedWindow(winName, cv.WINDOW_NORMAL)
outputFile = "webcam_yolo_v3_out.avi"

# let us start making settings based on the mode command line arguments are setup
if (args.image):
    # Open the image file
    print ("Taking input from a SINGLE Image file...")
    if not os.path.isfile(args.image):
        print("Input image file ", args.image, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.image)
    outputFile = args.image[:-4]+'_yolo_out.jpg'
    
elif (args.video):
    # Open the video file
    print ("Taking input from a SINGLE Video file...")
    if not os.path.isfile(args.video):
        print("Input video file ", args.video, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(args.video)
    outputFile = args.video[:-4]+'_yolo_out_.avi'
    
elif (args.test):
    # detect images in test directory (test_dir).
    print ("Taking inputs from a directory comatining multiple image files (For Testing)...")
    for (root, dirs, files) in os.walk(args.test_dirT):
        if files:
            for image in files:
                TrueImageList.append(root + "/" + image)
                
    for (root, dirs, files) in os.walk(args.test_dirF):
        if files:
            for image in files:
                FalseImageList.append(root + "/" + image)
                
else:
    # Webcam input
    print ("Taking live video input from PC's webcam")
    cap = cv.VideoCapture(0)

# In[10]]:

# Get the video writer initialized to save the output video
if (not args.image and not args.test):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M','J','P','G'), 30, 
                                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

if (args.test):
    TestImageIndexer = 0
    DictIndexer = 0
    TrueImageListParsed = False
    FalseImageListParsed = False
    TestWeaponDetected = False
    TestResultsList = np.zeros(shape=(len(TrueImageList) + len(FalseImageList),2) ).astype('bool')
    
# In[11]:

# RUN in loop for Inference Engine
while cv.waitKey(1) < 0:
    
    if (args.test):
        if (TrueImageListParsed == False):
            if (TestImageIndexer < len(TrueImageList)):
                cap = cv.VideoCapture(TrueImageList[TestImageIndexer])
            else:
                TrueImageListParsed = True
                TestImageIndexer = 0
                continue
        elif (FalseImageListParsed == False):
            if (TestImageIndexer < len(FalseImageList)):
                cap = cv.VideoCapture(FalseImageList[TestImageIndexer])
            else:
                FalseImageListParsed = True
                continue        
        
    # get frame from the video
    hasFrame, frame = cap.read()
    
    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        # Release device
        cap.release()
        break

    # Create a 4D blob from a frame.
    if (args.live):
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidthLIVE, inpHeightLIVE), [0,0,0], 1, crop=False)
    else:
        blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    TestWeaponDetected = WeaponDetector(frame, outs)
    
    if (args.test):
        # build a 2D list with first column of Image Name and second is Weapon Detected flag
        if (TrueImageListParsed == False):
            TestResultsList[TestImageIndexer][0] = True
            TestResultsList[TestImageIndexer][1] = TestWeaponDetected
        elif (FalseImageListParsed == False):
            TestResultsList[TestImageIndexer + len(TrueImageList)][0] = False
            TestResultsList[TestImageIndexer + len(TrueImageList)][1] = TestWeaponDetected
            
        # let us increment the Unit TEst Image indexer value now, for next run
        TestImageIndexer = TestImageIndexer + 1
        
    # Put efficiency information. The function getPerfProfile returns the overall time for 
    # inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (10, 28), cv.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 0))

    # Write the frame with the detection boxes
    if (args.image):
        cv.imwrite(outputFile, frame.astype(np.uint8))
    elif (args.video):
        vid_writer.write(frame.astype(np.uint8))

    if (not args.test):
        cv.imshow(winName, frame)


# In[12]:

# Parsed stuff upstairs, now let us see if we need post processing (e.g. Unit Test results parsing)
if (args.test):
    ActualTrue = len(TrueImageList)
    ActualTrueDetectedTrue = np.count_nonzero(TestResultsList[0:len(TrueImageList),1])
    ActualTrueDetectedFalse = len(TrueImageList) - ActualTrueDetectedTrue

    ActualFalse = len(FalseImageList)
    ActualFalseDetectedTrue = np.count_nonzero(TestResultsList[len(TrueImageList):,1])
    ActualFalseDetectedFalse = len(FalseImageList) - ActualFalseDetectedTrue
    
    # Let us calculate ACCURACY
    Accuracy = 100*((ActualTrueDetectedTrue + ActualFalseDetectedFalse)/(ActualTrue + ActualFalse))
    print ('')
    print( "Accuracy: ", round(Accuracy,2), "%")

    # Let us calculate RECALL
    Recall = ((ActualTrueDetectedTrue)/(ActualTrueDetectedTrue + ActualTrueDetectedFalse))
    print ('')
    print( "Recall: ", round(Recall,2))    

    # Let us calculate PRECISION
    Precision = ((ActualTrueDetectedTrue)/(ActualTrueDetectedTrue + ActualFalseDetectedTrue))
    print ('')
    print( "Precision: ", round(Precision,2))    

    # Let us calculate F1 SCORE
    F1Score = 2*((Precision*Recall)/(Precision + Recall))
    print ('')
    print( "F1Score: ", round(F1Score,2))    
    
    # Plot Seaborn confusion Matrix
    confusion_matrix = np.empty((2,2))
    confusion_matrix[0,0] = ActualTrueDetectedTrue
    confusion_matrix[0,1] = ActualTrueDetectedFalse
    confusion_matrix[1,0] = ActualFalseDetectedTrue
    confusion_matrix[1,1] = ActualFalseDetectedFalse

    fig, ax = plt.subplots()
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    sns.set(font_scale=1.4)
    sns.heatmap(confusion_matrix, annot=True, fmt='g')
    plt.xlabel('Weapon Detected', fontsize = 16)
    plt.ylabel('Actual Weapon')
    

# In[13]:

# EXIT Gracefully
if (not(args.live)):
    # Exit out gracefully
    cv.waitKey(5000)

# Release device
cap.release()
cv.destroyAllWindows()