# YOLOv3-Gun-Detection-Model

Our team’s motivation is to be proactive and address these open issues faced during an aftermath of a shootout. We conceptualize a shootout countermeasure system which detects handheld weapons before the shooter tries to use them. Our design will leverage on the current infrastructure e.g. surveillance cameras, web cams etc. which are already deployed and are situated in the public places like schools, restaurants, and other public buildings. Detection and location of a handheld weapon will be based on object detection and classification using image/video feeds (using a visible camera or an infrared one). Our current implementation relies heavily on visible imagery, but a similar approach could be used for short or long wave infrared camera systems for detecting concealed weapons. Since time between weapon detection and taking an action is of quintessential importance for an effective countermeasure plan, our system requirements are to have a near real time detection methodology.


Summary of our system requirements:

a)	Should be able to detect a weapon in a scene, in near real time basis.

b)	Should have a small footprint in terms of memory usage, CPU consumption.

c)	Should have ability to be custom trained as and when needed.

d)	Should be able to work on Open Source software as much as possible in order to have smaller production cost and stay intellectual property free.

e)	Should be deployable on the current surveillance systems available in the market e.g webcams, IP cameras etc.

f)	Should have zero false negative detections because, we never want to miss an actual weapon in the scene. In other words, Recall should be 1.

g)	Should have small amount of false positive detections, as we do not want to act when there is no need to do so. In other words, system’s precision should be greater than 95%.   


