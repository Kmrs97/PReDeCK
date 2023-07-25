# PReDeCK
PReDeCK- A framework built upon NeurASP, for Part Of Relation Detection on detected objects in an image, using Commonsense Knowledge 
<br>
## Install NeurASP
* Follow the instruction found [here](https://github.com/azreasoners/NeurASP/tree/master).
* Instead of installing the packages that are listed there, install the provided conda enviroment running the following command:
```
conda env create -n predeck --file env.yml
conda activate predeck
```
* Replace the files ```neurasp.py``` and ```mvpp.py``` with the ones provided in this repository. 

* In ```NeurASP``` folder, insert the file ```string_parser.py``` and the folder ```ConceptNet```.

* In ```NeurASP/examples/```  insert the folder ```PReDeCK```.

* In ```NeurASP/examples/PReDeCK``` do the following:
1. Install YOLOv5 according to the [instructions](https://github.com/ultralytics/yolov5).
2. Download the [dataset](https://universe.roboflow.com/pascalpart/pascal-part-fquij) and copy the test set (only the images) in the folder.

* Fill the required fields in  ```config.yaml``` file, accordingly. 
1. Specify the path of the images.
2. What experiments to run
3. Enable/disable evaluation process
4. Specify the output path
5. Specify the deployable model following the example
6. Set the size of the images (in our dataset is 640)


* Run the system with the command stated below:
```python run.py```

## Commonsense Knowledge exportation via ConceptNet

We provide all the necessary files to pull information from ConceptNet. 

* First, you have to initialize a local graph through Neo4j, and insert the credentials in the ```.py``` files, when needed.

* Then, by running ```python neo_concept_categories.py```, the local graph that includes the requested information is ready.

* To represent the knowledge derived from ConceptNet in ASP, run ```python create_CNrules_file.py```

To avoid, the previous steps, we have provided the facts in ASP in the file ```CNrules.txt```.

In general, the method is quite adaptable and can be configured according to the needs of the user.
