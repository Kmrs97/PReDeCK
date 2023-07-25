# PReDeCK
PReDeCK- A framework built upon NeurASP, for Part Of Relation Detection on detected objects in an image, using Commonsense Knowledge 
<br>
## Install NeurASP
Follow the instruction found [here](https://github.com/azreasoners/NeurASP/tree/master).
Instead of installing the packages that are listed there, install the provided conda enviroment running the following command:
```
conda env create -n predeck --file env.yml
conda activate predeck
```
Replace the files ```neurasp.py``` and ```mvpp.py``` with the ones provided in this repository. 

In ```NeurASP``` folder, insert the file ```string_parser.py``` and the folder ```ConceptNet```.

In ```NeurASP/examples/```  insert the folder ```PReDeCK```.
In ```NeurASP/examples/PReDeCK``` do the following:
1. Install YOLOv5 according to the [instructions](https://github.com/ultralytics/yolov5).
2. Download the [dataset](https://universe.roboflow.com/pascalpart/pascal-part-fquij) and copy the test set in the folder.

Need to explain how to install neurasp enviroment 
<br>
Need to provide a link to the files my folder semanticPascalPart->Instead provide ground truth json 
<br>
And to the test set-Roboflow
<br>
Install yolo

