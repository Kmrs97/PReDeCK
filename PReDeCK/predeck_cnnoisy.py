import sys
sys.path.append('../../')

import torch
import json
from dataGen import termPath2dataList
from network import Net
from neurasp import NeurASP
import string_parser as pstr
import os.path
from os import path

def infer(img_path,img_size,model,output):
    print("Running noisy ConceptNet Experiment...")
    dprogram = r'''
    nn(label(1,I,B),["Aeroplane", "Animal_Wing", "Arm", "Artifact_Wing", "Beak", "Bicycle", "Bird", "Boat", "Body", "Bodywork", "Bottle", "Bus", "Cap", "Car", "Cat", "Chain_wheel", "Chair", "Coach", "Cow", "Dog", "Door", "Ear", "Ebrow", "Engine", "Eye", "Foot", "Hair", "Hand", "Handlebar", "Head", "Headlight", "Hoof", "Horn", "Horse", "Leg", "License_plate", "Locomotive", "Mirror", "Motorbike", "Mouth", "Muzzle", "Neck", "Nose", "Person", "Plant", "Pot", "Pottedplant", "Saddle", "Screen", "Sheep", "Sofa", "Stern", "Tail", "Torso", "Train", "Tvmonitor", "Wheel", "Window", "Diningtable","Other"]) :- box(I,B,X1,Y1,X2,Y2,C).

    '''

    aspProgram = r'''
    


    '''
    # %partOf/3 inverse of hasA/3
    #  partOf(L1,L2,C):-hasA(L2,L1,C).

    # %partOf/3 is inverse of hasContext/3
    # partOf(X,Y,C):-hasContext(X,Y,C).

    # %Transitivity
    # partOf(L1,L3,C1):-
    #     partOf(L1,L2,C1),
    #     partOf(L2,L3,C2),
    #     C1=C2.

    # %Symmetricity
    # isSynonymWith(X,Y,C):-isSynonymWith(Y,X,C).

    # %Inference rules

    # partOf(X,Y,C1):-
    #     partOf(X,Z,C1),
    #     isA(Y,Z,C2),
    #     C1=C2.

    # partOf(Z,Y,C):-
    #     partOf(X,Y,C),
    #     isA(X,Z,C).

    # partOf(X,Y,C):-
    #     partOf(X,Z,C),
    #     isA(Z,Y,C).

    # partOf(Z, Y,C1) :-
    #     partOf(X, Y,C1),
    #     isSynonymWith(X,Z,C2),
    #     C1=C2.

    # partOf(X,Z,C1) :-
    #     partOf(X, Y,C1),
    #     isSynonymWith(Y,Z,C2),
    #     C1=C2.

    objects_relationships_rules=r'''
partOf("Sprocket","Bicycle","Artifact").
partOf("Handlebar","Bicycle","Artifact").
partOf("Chain_wheel","Bicycle","Artifact").
partOf("Saddle","Bicycle","Artifact").
partOf("Wheel","Bicycle","Artifact").
partOf("Bicycle","Bicycle","Artifact").
partOf("Engine","Bicycle","Artifact").
partOf("Horn","Bicycle","Artifact").
partOf("Window","Bicycle","Artifact").
partOf("Bodywork","Bicycle","Artifact").
partOf("Leg","Bicycle","Artifact").
partOf("Train","Bicycle","Artifact").
partOf("Locomotive","Bicycle","Artifact").
partOf("Wing","Bicycle","Artifact").
partOf("Mouth","Bicycle","Artifact").
partOf("Stern","Bicycle","Artifact").
partOf("Door","Bicycle","Artifact").
partOf("Headlight","Bicycle","Artifact").
partOf("Foot","Bicycle","Artifact").
partOf("Artifact_Wing","Bicycle","Artifact").
partOf("Tail","Bicycle","Artifact").
partOf("Car","Bicycle","Artifact").
partOf("Mouth","Bottle","Artifact").
partOf("Leg","Bottle","Artifact").
partOf("Foot","Bottle","Artifact").
partOf("Wing","Bird","Animal").
partOf("Beak","Bird","Animal").
partOf("Animal_Wing","Bird","Animal").
partOf("Nose","Bird","Animal").
partOf("Body","Bird","Animal").
partOf("Head","Bird","Animal").
partOf("Tail","Bird","Animal").
partOf("Torso","Bird","Animal").
partOf("Ear","Bird","Animal").
partOf("Muzzle","Bird","Animal").
partOf("Mouth","Bird","Animal").
partOf("Eye","Bird","Animal").
partOf("Leg","Bird","Animal").
partOf("Neck","Bird","Animal").
partOf("Foot","Bird","Animal").
partOf("Window","Car","Artifact").
partOf("Horn","Car","Animal").
partOf("Horn","Car","Artifact").
partOf("Engine","Car","Artifact").
partOf("Locomotive","Car","Artifact").
partOf("Wing","Car","Artifact").
partOf("Bodywork","Car","Artifact").
partOf("Wheel","Car","Artifact").
partOf("Train","Car","Artifact").
partOf("Bicycle","Car","Artifact").
partOf("Artifact_Wing","Car","Artifact").
partOf("Headlight","Car","Artifact").
partOf("Stern","Car","Artifact").
partOf("Mouth","Car","Artifact").
partOf("Leg","Car","Artifact").
partOf("Handlebar","Car","Artifact").
partOf("Sprocket","Car","Artifact").
partOf("Chain_wheel","Car","Artifact").
partOf("Saddle","Car","Artifact").
partOf("Tail","Car","Artifact").
partOf("Door","Car","Artifact").
partOf("Foot","Car","Artifact").
partOf("Car","Car","Artifact").
partOf("Window","Bus","Artifact").
partOf("Engine","Bus","Artifact").
partOf("Horn","Bus","Artifact").
partOf("Wing","Bus","Artifact").
partOf("Locomotive","Bus","Artifact").
partOf("Wheel","Bus","Artifact").
partOf("Bodywork","Bus","Artifact").
partOf("Train","Bus","Artifact").
partOf("Artifact_Wing","Bus","Artifact").
partOf("Sprocket","Bus","Artifact").
partOf("Stern","Bus","Artifact").
partOf("Bicycle","Bus","Artifact").
partOf("Handlebar","Bus","Artifact").
partOf("Leg","Bus","Artifact").
partOf("Mouth","Bus","Artifact").
partOf("Headlight","Bus","Artifact").
partOf("Chain_wheel","Bus","Artifact").
partOf("Saddle","Bus","Artifact").
partOf("Foot","Bus","Artifact").
partOf("Tail","Bus","Artifact").
partOf("Door","Bus","Artifact").
partOf("Car","Bus","Artifact").
partOf("Hair","Cow","Animal").
partOf("Tail","Cow","Animal").
partOf("Nose","Cow","Animal").
partOf("Body","Cow","Animal").
partOf("Head","Cow","Animal").
partOf("Torso","Cow","Animal").
partOf("Ear","Cow","Animal").
partOf("Muzzle","Cow","Animal").
partOf("Mouth","Cow","Animal").
partOf("Eye","Cow","Animal").
partOf("Leg","Cow","Animal").
partOf("Neck","Cow","Animal").
partOf("Foot","Cow","Animal").
partOf("Beak","Cow","Animal").
partOf("Wing","Cow","Animal").
partOf("Animal_Wing","Cow","Animal").
partOf("Hair","Cat","Animal").
partOf("Nose","Cat","Animal").
partOf("Hair","Cat","Body").
partOf("Nose","Cat","Body").
partOf("Body","Cat","Animal").
partOf("Head","Cat","Animal").
partOf("Torso","Cat","Animal").
partOf("Ear","Cat","Animal").
partOf("Muzzle","Cat","Animal").
partOf("Mouth","Cat","Animal").
partOf("Eye","Cat","Animal").
partOf("Leg","Cat","Animal").
partOf("Neck","Cat","Animal").
partOf("Foot","Cat","Animal").
partOf("Beak","Cat","Animal").
partOf("Tail","Cat","Animal").
partOf("Wing","Cat","Animal").
partOf("Animal_Wing","Cat","Animal").
partOf("Nose","Horse","Animal").
partOf("Body","Horse","Animal").
partOf("Head","Horse","Animal").
partOf("Torso","Horse","Animal").
partOf("Ear","Horse","Animal").
partOf("Muzzle","Horse","Animal").
partOf("Mouth","Horse","Animal").
partOf("Eye","Horse","Animal").
partOf("Leg","Horse","Animal").
partOf("Neck","Horse","Animal").
partOf("Foot","Horse","Animal").
partOf("Beak","Horse","Animal").
partOf("Wing","Horse","Animal").
partOf("Tail","Horse","Animal").
partOf("Animal_Wing","Horse","Animal").
partOf("Nose","Dog","Animal").
partOf("Nose","Dog","Body").
partOf("Head","Dog","Animal").
partOf("Body","Dog","Animal").
partOf("Torso","Dog","Animal").
partOf("Foot","Dog","Animal").
partOf("Neck","Dog","Animal").
partOf("Leg","Dog","Animal").
partOf("Eye","Dog","Animal").
partOf("Mouth","Dog","Animal").
partOf("Muzzle","Dog","Animal").
partOf("Ear","Dog","Animal").
partOf("Beak","Dog","Animal").
partOf("Hair","Dog","Animal").
partOf("Tail","Dog","Animal").
partOf("Wing","Dog","Animal").
partOf("Animal_Wing","Dog","Animal").
partOf("Nose","Car","Artifact").
partOf("Mouth","Car","Artifact").
partOf("Mouth","Car","Artifact").
partOf("Nose","Car","Artifact").
partOf("Mouth","Car","Artifact").
partOf("Foot","Car","Artifact").
partOf("Body","Car","Artifact").
partOf("Arm","Car","Artifact").
partOf("Leg","Car","Artifact").
partOf("Head","Car","Artifact").
partOf("Torso","Car","Artifact").
partOf("Neck","Car","Artifact").
partOf("Ear","Car","Artifact").
partOf("Muzzle","Car","Artifact").
partOf("Torso","Car","Artifact").
partOf("Neck","Car","Artifact").
partOf("Leg","Car","Artifact").
partOf("Head","Car","Artifact").
partOf("Eye","Car","Artifact").
partOf("Stern","Car","Artifact").
partOf("Tail","Car","Artifact").
partOf("Eye","Car","Artifact").
partOf("Eyebrow","Car","Artifact").
partOf("Ear","Car","Artifact").
partOf("Foot","Car","Artifact").
partOf("Hand","Car","Artifact").
partOf("Ebrow","Car","Artifact").
partOf("Beak","Car","Artifact").
partOf("Wing","Car","Artifact").
partOf("Tail","Car","Artifact").

partOf("Nose","Bicycle","Artifact").
partOf("Mouth","Bicycle","Artifact").
partOf("Mouth","Bicycle","Artifact").
partOf("Nose","Bicycle","Artifact").
partOf("Mouth","Bicycle","Artifact").
partOf("Foot","Bicycle","Artifact").
partOf("Body","Bicycle","Artifact").
partOf("Arm","Bicycle","Artifact").
partOf("Leg","Bicycle","Artifact").
partOf("Head","Bicycle","Artifact").
partOf("Torso","Bicycle","Artifact").
partOf("Neck","Bicycle","Artifact").
partOf("Ear","Bicycle","Artifact").
partOf("Muzzle","Bicycle","Artifact").
partOf("Torso","Bicycle","Artifact").
partOf("Neck","Bicycle","Artifact").
partOf("Leg","Bicycle","Artifact").
partOf("Head","Bicycle","Artifact").
partOf("Eye","Bicycle","Artifact").
partOf("Stern","Bicycle","Artifact").
partOf("Tail","Bicycle","Artifact").
partOf("Eye","Bicycle","Artifact").
partOf("Eyebrow","Bicycle","Artifact").
partOf("Ear","Bicycle","Artifact").
partOf("Foot","Bicycle","Artifact").
partOf("Hand","Bicycle","Artifact").
partOf("Ebrow","Bicycle","Artifact").
partOf("Beak","Bicycle","Artifact").
partOf("Wing","Bicycle","Artifact").
partOf("Tail","Bicycle","Artifact").

partOf("Animal_Wing","Bicycle","Animal").
partOf("Bodywork","Motorbike","Artifact").
partOf("Train","Motorbike","Artifact").
partOf("Wheel","Motorbike","Artifact").
partOf("Locomotive","Motorbike","Artifact").
partOf("Wing","Motorbike","Artifact").
partOf("Sprocket","Motorbike","Artifact").
partOf("Handlebar","Motorbike","Artifact").
partOf("Headlight","Motorbike","Artifact").
partOf("Engine","Motorbike","Artifact").
partOf("Horn","Motorbike","Artifact").
partOf("Window","Motorbike","Artifact").
partOf("Chain_wheel","Motorbike","Artifact").
partOf("Saddle","Motorbike","Artifact").
partOf("Artifact_Wing","Motorbike","Artifact").
partOf("Stern","Motorbike","Artifact").
partOf("Bicycle","Motorbike","Artifact").
partOf("Door","Motorbike","Artifact").
partOf("Leg","Motorbike","Artifact").
partOf("Mouth","Motorbike","Artifact").
partOf("Foot","Motorbike","Artifact").
partOf("Tail","Motorbike","Artifact").
partOf("Car","Motorbike","Artifact").
partOf("Sprocket","Train","Artifact").
partOf("Handlebar","Train","Artifact").
partOf("Chain_wheel","Train","Artifact").
partOf("Saddle","Train","Artifact").
partOf("Engine","Train","Artifact").
partOf("Horn","Train","Artifact").
partOf("Window","Train","Artifact").
partOf("Wheel","Train","Artifact").
partOf("Leg","Train","Artifact").
partOf("Mouth","Train","Artifact").
partOf("Bicycle","Train","Artifact").
partOf("Wing","Train","Artifact").
partOf("Locomotive","Train","Artifact").
partOf("Stern","Train","Artifact").
partOf("Door","Train","Artifact").
partOf("Bodywork","Train","Artifact").
partOf("Train","Train","Artifact").
partOf("Tail","Train","Artifact").
partOf("Foot","Train","Artifact").
partOf("Car","Train","Artifact").
partOf("Artifact_Wing","Train","Artifact").
partOf("Headlight","Train","Artifact").
partOf("Nose","Sheep","Animal").
partOf("Body","Sheep","Animal").
partOf("Head","Sheep","Animal").
partOf("Torso","Sheep","Animal").
partOf("Ear","Sheep","Animal").
partOf("Muzzle","Sheep","Animal").
partOf("Mouth","Sheep","Animal").
partOf("Eye","Sheep","Animal").
partOf("Leg","Sheep","Animal").
partOf("Neck","Sheep","Animal").
partOf("Foot","Sheep","Animal").
partOf("Beak","Sheep","Animal").
partOf("Wing","Sheep","Animal").
partOf("Tail","Sheep","Animal").
partOf("Animal_Wing","Sheep","Animal").
partOf("Wing","Aeroplane","Artifact").
partOf("Tail","Aeroplane","Artifact").
partOf("Artifact_Wing","Aeroplane","Artifact").
partOf("Stern","Aeroplane","Artifact").
partOf("Car","Aeroplane","Artifact").
partOf("Door","Aeroplane","Artifact").
partOf("Window","Aeroplane","Artifact").
partOf("Horn","Aeroplane","Artifact").
partOf("Engine","Aeroplane","Artifact").
partOf("Locomotive","Aeroplane","Artifact").
partOf("Bodywork","Aeroplane","Artifact").
partOf("Wheel","Aeroplane","Artifact").
partOf("Mouth","Aeroplane","Artifact").
partOf("Sprocket","Aeroplane","Artifact").
partOf("Handlebar","Aeroplane","Artifact").
partOf("Train","Aeroplane","Artifact").
partOf("Saddle","Aeroplane","Artifact").
partOf("Chain_wheel","Aeroplane","Artifact").
partOf("Bicycle","Aeroplane","Artifact").
partOf("Leg","Aeroplane","Artifact").
partOf("Foot","Aeroplane","Artifact").
partOf("Headlight","Aeroplane","Artifact").

    

    %define boxes that contain objects
    objectBox(B,L,C):-label(_,_,B,L),object(L,C).

    %define boxes that contain parts
    partBox(B,L,C):-label(_,_,B,L),part(L,C).


    %define boxes that have a Spatial partOf relation
    spatial_partOf(B1,Lbl1,B2,Lbl2):-over90(B1,B2),
                    partBox(B1,Lbl1,C),
                    objectBox(B2,Lbl2,C),
                    partOf(Lbl1,Lbl2,C).



                                                                        
    '''


    hashtag='#'
    spatial_rules=r'''

    % Define the area of the bounding boxes
    area(B, A) :- box(_,B, Xmin,Ymin, Xmax, Ymax,_), 
                A = (Xmax - Xmin) * (Ymax - Ymin).

    %Find if two bounding boxes overlap
    overlap(B1,B2):-box(_,B1,Xmin1,Ymin1,Xmax1,Ymax1,_),
                    box(_,B2,Xmin2,Ymin2,_,_,_),
                    B1!=B2,
                    Xmin2>=Xmin1,
                    Xmin2<=Xmax1,
                    Ymin2>=Ymin1,
                    Ymin2<=Ymax1.

    overlap(B1,B2):-box(_,B1,Xmin1,Ymin1,Xmax1,Ymax1,_),
                    box(_,B2,Xmin2,_,_,Ymax2,_),
                    B1!=B2,
                    Xmin2>=Xmin1,
                    Xmin2<=Xmax1,
                    Ymax2>=Ymin1,
                    Ymax2<=Ymax1.

    overlap(B1,B2):-box(_,B1,Xmin1,Ymin1,Xmax1,Ymax1,_),
                    box(_,B2,Xmin2,Ymin2,Xmax2,Ymax2,_),
                    B1!=B2,
                    Xmin1<Xmax2,
                    Xmin2<Xmax1,
                    Ymin1<Ymax2,
                    Ymin2<Ymax1.                

    % Define the over90 predicate
    %Find if two bounding boxes overlap with over 90% coverage
    over90(Bmin,Bmax) :- box(_,B1,Xmin1,Ymin1,Xmax1,Ymax1,_),
                    box(_,B2,Xmin2,Ymin2,Xmax2,Ymax2,_),
                    overlap(B1,B2),
                    area(B1, A1), area(B2, A2),
                    Amin='''+hashtag+r'''min{A1;A2},
                    Amax='''+hashtag+r'''max{A1;A2},
                    area(Bmin,Amin),
                    area(Bmax,Amax),
                    Ymax='''+hashtag+r'''min{Ymax1;Ymax2},
                    Ymin='''+hashtag+r'''max{Ymin1;Ymin2},
                    Xmax='''+hashtag+r'''min{Xmax1;Xmax2},
                    Xmin='''+hashtag+r'''max{Xmin1;Xmin2},
                    Aovl=(Ymax - Ymin) * (Xmax- Xmin),
                    90 <= ((100*Aovl)/ Amin).

                
    '''


    constraints=r'''

    '''
        

    # file = open('CNrules.txt', 'r')
    # rules = file.readlines()

    # for r in rules:
    #     objects_relationships_rules+=r


    m = Net()
    nnMapping = {'label': m}

    termPath = 'img ./'+img_path

    # set the classes that we consider
    domain =["Aeroplane", "Animal_Wing", "Arm", "Artifact_Wing", "Beak", "Bicycle", "Bird", "Boat", "Body", "Bodywork", "Bottle", "Bus", "Cap", "Car", "Cat", "Chain_wheel", "Chair", "Coach", "Cow", "Dog", "Door", "Ear", "Ebrow", "Engine", "Eye", "Foot", "Hair", "Hand", "Handlebar", "Head", "Headlight", "Hoof", "Horn", "Horse", "Leg", "License_plate", "Locomotive", "Mirror", "Motorbike", "Mouth", "Muzzle", "Neck", "Nose", "Person", "Plant", "Pot", "Pottedplant", "Saddle", "Screen", "Sheep", "Sofa", "Stern", "Tail", "Torso", "Train", "Tvmonitor", "Wheel", "Window", "Diningtable","Other"]
    objects=["Aeroplane","Bicycle","Bird","Boat","Bottle","Bus","Car", "Cat","Chair","Cow", "Dog","Horse"
    ,"Motorbike","Person","Pottedplant","Sheep", "Sofa","Train", "Tvmonitor", "Diningtable"]

    animals=["Beak","Bird","Cat","Cow","Dog","Foot","Hoof","Horn","Horse","Muzzle",
	"Neck","Leg","Saddle","Sheep","Tail","Torso","Head","Eye","Nose","Ear", "Animal_Wing"]
    body=["Body","Ear","Ebrow","Eye","Foot","Hair","Hand","Head","Leg","Mouth","Neck",
	"Nose","Torso","Arm"]
    artifacts=["Aeroplane","Bicycle","Boat","Bodywork","Bottle","Bus","Cap",
	"Car","Chair","Coach","Door","Engine","Handlebar","Headlight","Locomotive","Mirror",
	"Motorbike","Pot","Saddle","Screen","Sofa","Stern","Tail","Train","Tvmonitor","Wheel"
	,"Window","Diningtable","Body","License_plate","Person","Artifact_Wing", "Chain_wheel"]

    #skip, they dont contribute to the data colection
    # person=["Arm","Person"]
    # plant=["Plant","Pottedplant"]

    parts=[x for x in domain if x not in objects and x!="Other"]

  # Rules to separate objects and parts
    for lbl in animals:
        if lbl in objects:
             rule='object("'+lbl+'","Animal").'
        elif lbl in parts:
            rule='part("'+lbl+'","Animal").'
        objects_relationships_rules+=rule+'\n'
    
    for lbl in body:
        if lbl in objects:
             rule='object("'+lbl+'","Body").'
        elif lbl in parts:
            rule='part("'+lbl+'","Body").'
        objects_relationships_rules+=rule+'\n'

    for lbl in artifacts:
        if lbl in objects:
             rule='object("'+lbl+'","Artifact").'
        elif lbl in parts:
            rule='part("'+lbl+'","Artifact").'
        objects_relationships_rules+=rule+'\n'


    
    
        
  
   

    aspProgram=spatial_rules+objects_relationships_rules+constraints
    # +hashtag+r'''show box_partOf/4.'''

    print(aspProgram)


    image_ids,factsList, dataList =termPath2dataList(termPath, img_size, domain,model,"pascal") ##coco OR pascal
    # print(image_ids)
    # print(factsList)
    # print("\n")
    # print(dataList)


    json_dict={}
    for idx, facts in enumerate(factsList):
        print(image_ids[idx])
        json_dict[image_ids[idx]]=[]
        NeurASPobj = NeurASP(dprogram + facts, nnMapping, optimizers=None)
        # Find the most probable stable model
        models,_ = NeurASPobj.infer(dataDic=dataList[idx], obs='', mvpp=aspProgram + facts,postProcessing=False,stable_models=1)
        # print('\nInfernece Result on Data {}:'.format(idx+1))
        # print(models)

        ##For all stable models
        # for i in range(len(models)):
        #     # print(models[i])
        #     # print('\n')
        #     d=pstr.parse_string(models[i])
        #     json_dict[image_ids[idx]].append(d)

        ##Only for the first stable model
        d={}
        if len(models)>0:
            d=pstr.parse_string(models[0])
            json_dict[image_ids[idx]].append(d)
        
    #write json with the inferences
    #     
    json_object=json.dumps(json_dict,indent=4)
    if path.exists(output)==False:
        os.mkdir(output)
    opath=output+'/noisy_conceptnet'
    if path.exists(opath)==False:
        os.mkdir(opath) 
    with open(opath+"/NCN_inferences_JSON.json", "w") as outfile:
        outfile.write(json_object)
    outfile.close()
