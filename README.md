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
2. Download the [dataset](https://universe.roboflow.com/pascalpart/pascal-part-fquij) and copy the test set (only the images) in the folder.

Fill the required fields in  ```config.yaml``` file, accordingly. 
1. Specify the path of the images.
2. What experiments to run
3. Enable/disable evaluation process
4. Specify the output path
5. Specify the deployable model following the example
6. Set the size of the images (in our dataset is 640)


Need to provide a link to the files my folder semanticPascalPart->Instead provide ground truth json 


