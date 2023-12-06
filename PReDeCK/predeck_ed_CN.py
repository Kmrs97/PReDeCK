import sys
sys.path.append('../../')

import torch
import json
from dataGen import termPath2dataList
from network import Net
from neurasp import NeurASP
from examples.yolov5_connection import parse_dataset as prd
from ConceptNet import partOf_relation as pr
import string_parser as pstr
import os.path
from os import path

import errors_handling as erh

def infer(img_path,img_size,model,output):
    dprogram = r'''
    nn(label(1,I,B),["Aeroplane", "Animal_Wing", "Arm", "Artifact_Wing", "Beak", "Bicycle", "Bird", "Boat", "Body", "Bodywork", "Bottle", "Bus", "Cap", "Car", "Cat", "Chain_wheel", "Chair", "Coach", "Cow", "Dog", "Door", "Ear", "Ebrow", "Engine", "Eye", "Foot", "Hair", "Hand", "Handlebar", "Head", "Headlight", "Hoof", "Horn", "Horse", "Leg", "License_plate", "Locomotive", "Mirror", "Motorbike", "Mouth", "Muzzle", "Neck", "Nose", "Person", "Plant", "Pot", "Pottedplant", "Saddle", "Screen", "Sheep", "Sofa", "Stern", "Tail", "Torso", "Train", "Tvmonitor", "Wheel", "Window", "Diningtable","Other"]) :- box(I,B,X1,Y1,X2,Y2,C).

    '''

    aspProgram = r'''


    '''

    cn_knowledge=r'''
    
partOf("Sprocket","Bicycle","Artifact").
partOf("Handlebar","Bicycle","Artifact").
partOf("Saddle","Bicycle","Artifact").
partOf("Wheel","Bicycle","Artifact").
partOf("Chain_wheel","Bicycle","Artifact").
partOf("Leg","Bicycle","Artifact").
partOf("Mouth","Bicycle","Artifact").
partOf("Foot","Bicycle","Artifact").
partOf("Mouth","Bottle","Artifact").
partOf("Wing","Bird","Animal").
partOf("Beak","Bird","Animal").
partOf("Animal_Wing","Bird","Animal").
partOf("Nose","Bird","Animal").
partOf("Body","Bird","Animal").
partOf("Head","Bird","Animal").
partOf("Tail","Bird","Animal").
partOf("Ear","Bird","Animal").
partOf("Muzzle","Bird","Animal").
partOf("Mouth","Bird","Animal").
partOf("Eye","Bird","Animal").
partOf("Torso","Bird","Animal").
partOf("Leg","Bird","Animal").
partOf("Neck","Bird","Animal").
partOf("Foot","Bird","Animal").
partOf("Window","Car","Artifact").
partOf("Horn","Car","Animal").
partOf("Horn","Car","Artifact").
partOf("Engine","Car","Artifact").
partOf("Bodywork","Car","Artifact").
partOf("Wheel","Car","Artifact").
partOf("Headlight","Car","Artifact").
partOf("Mouth","Car","Artifact").
partOf("Leg","Car","Artifact").
partOf("Door","Car","Artifact").
partOf("Foot","Car","Artifact").
partOf("Window","Bus","Artifact").
partOf("Engine","Bus","Artifact").
partOf("Horn","Bus","Artifact").
partOf("Wheel","Bus","Artifact").
partOf("Bodywork","Bus","Artifact").
partOf("Leg","Bus","Artifact").
partOf("Mouth","Bus","Artifact").
partOf("Headlight","Bus","Artifact").
partOf("Foot","Bus","Artifact").
partOf("Door","Bus","Artifact").
partOf("Hair","Cow","Animal").
partOf("Tail","Cow","Animal").
partOf("Nose","Cow","Animal").
partOf("Body","Cow","Animal").
partOf("Head","Cow","Animal").
partOf("Ear","Cow","Animal").
partOf("Muzzle","Cow","Animal").
partOf("Mouth","Cow","Animal").
partOf("Eye","Cow","Animal").
partOf("Torso","Cow","Animal").
partOf("Leg","Cow","Animal").
partOf("Neck","Cow","Animal").
partOf("Foot","Cow","Animal").
partOf("Hair","Cat","Animal").
partOf("Nose","Cat","Animal").
partOf("Hair","Cat","Body").
partOf("Nose","Cat","Body").
partOf("Body","Cat","Animal").
partOf("Head","Cat","Animal").
partOf("Ear","Cat","Animal").
partOf("Muzzle","Cat","Animal").
partOf("Mouth","Cat","Animal").
partOf("Eye","Cat","Animal").
partOf("Torso","Cat","Animal").
partOf("Leg","Cat","Animal").
partOf("Neck","Cat","Animal").
partOf("Foot","Cat","Animal").
partOf("Tail","Cat","Animal").
partOf("Nose","Horse","Animal").
partOf("Body","Horse","Animal").
partOf("Head","Horse","Animal").
partOf("Ear","Horse","Animal").
partOf("Muzzle","Horse","Animal").
partOf("Mouth","Horse","Animal").
partOf("Eye","Horse","Animal").
partOf("Torso","Horse","Animal").
partOf("Leg","Horse","Animal").
partOf("Neck","Horse","Animal").
partOf("Foot","Horse","Animal").
partOf("Nose","Dog","Animal").
partOf("Nose","Dog","Body").
partOf("Head","Dog","Animal").
partOf("Body","Dog","Animal").
partOf("Foot","Dog","Animal").
partOf("Neck","Dog","Animal").
partOf("Leg","Dog","Animal").
partOf("Torso","Dog","Animal").
partOf("Eye","Dog","Animal").
partOf("Mouth","Dog","Animal").
partOf("Muzzle","Dog","Animal").
partOf("Ear","Dog","Animal").
partOf("Hair","Dog","Animal").
partOf("Tail","Dog","Animal").
partOf("Nose","Person","Animal").
partOf("Mouth","Person","Animal").
partOf("Mouth","Person","Body").
partOf("Nose","Person","Body").
partOf("Mouth","Person","Artifact").
partOf("Arm","Person","Body").
partOf("Leg","Person","Body").
partOf("Head","Person","Body").
partOf("Torso","Person","Body").
partOf("Neck","Person","Body").
partOf("Eye","Person","Body").
partOf("Eyebrow","Person","Body").
partOf("Ear","Person","Body").
partOf("Foot","Person","Body").
partOf("Hand","Person","Body").
partOf("Ebrow","Person","Body").
partOf("Nose","Sheep","Animal").
partOf("Body","Sheep","Animal").
partOf("Head","Sheep","Animal").
partOf("Ear","Sheep","Animal").
partOf("Muzzle","Sheep","Animal").
partOf("Mouth","Sheep","Animal").
partOf("Eye","Sheep","Animal").
partOf("Torso","Sheep","Animal").
partOf("Leg","Sheep","Animal").
partOf("Neck","Sheep","Animal").
partOf("Foot","Sheep","Animal").



    '''
    objects_relationships_rules=r'''


    %define boxes that contain objects
    objectBox(B,L,Conf,C):-label(_,_,B,L),object(L,C),box(_,B,_,_,_,_,Conf).

    %define boxes that contain parts
    partBox(B,L,Conf,C):-label(_,_,B,L),part(L,C),box(_,B,_,_,_,_,Conf).


    %define boxes that have a Spatial partOf relation
    spatial_partOf(B1,Lbl1,B2,Lbl2):-over90(B1,B2),
                    partBox(B1,Lbl1,_,_),
                    objectBox(B2,Lbl2,_,_).


    %Rule to infer the Actual Part Of between two boxes,considering spatial AND semantic relation                              
    semantic_partOf(B1,L1,B2,L2):-spatial_partOf(B1,L1,B2,L2),
                                   partBox(B1,L1,_,C),
                                   objectBox(B2,L2,_,C),
                                   partOf(L1,L2,C).

                                                                                                  
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
                        box(_,B2,Xmin2,Ymin2,Xmax2,Ymax2,_),
                        B1!=B2,
                        Xmin1<Xmax2,
                        Xmin2<Xmax1,
                        Ymin1<Ymax2,
                        Ymin2<Ymax1.

        overlap(B1,B2):-box(_,B1,Xmin1,Ymin1,Xmax1,Ymax1,_),
                        box(_,B2,Xmin2,_,_,Ymax2,_),
                        B1!=B2,
                        Xmin2>=Xmin1,
                        Xmin2<=Xmax1,
                        Ymax2>=Ymin1,
                        Ymax2<=Ymax1.

               

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
                        75 <= ((100*Aovl)/ Amin).

                    
        '''

    
    error_detection_ruleset=r'''
        %%PRELIMINARIES

        highConfidence(B):-box(_,B,_,_,_,_,C),
                        C >= 400.

        hasNumberOfAccParts(Y,N):-N='''+hashtag+r'''count {(X,Y):semantic_partOf(X,_,Y,_),partBox(X,_,C,_),C>=400},objectBox(Y,_,_,_).

        context(B):-hasNumberOfAccParts(B,N), N >= 2.

        single(B):-partBox(B,_,_,_), not semantic_partOf(B,_,_,_).

        knownPart(B):-partBox(B,L,_,_),partOf(L,_,_).
        knownObject(B):-objectBox(B,L,_,_),partOf(_,L,_).

        %%RATIONALE

        errorCaseA(B1):-spatial_partOf(B1,_,B2,_),
                    knownPart(B1),
                    knownObject(B2),
                    highConfidence(B2),
                    not highConfidence(B1),
                    single(B1).

        errorCaseB(B2):-spatial_partOf(B1,_,B2,_),
                    knownPart(B1),
                    knownObject(B2),
                    highConfidence(B1),
                    not highConfidence(B2),
                    not context(B2),
                    single(B1).

        errorCaseAB(B1,B2):-spatial_partOf(B1,_,B2,_),
            knownPart(B1),
            knownObject(B2),
            not highConfidence(B1),
            not highConfidence(B2),
            not context(B2),
            single(B1).

        errorCaseC(B1):-spatial_partOf(B1,_,B2,_),
            knownPart(B1),
            knownObject(B2),
            highConfidence(B1),
            1{highConfidence(B2);context(B2)},
            single(B1).

        errorCaseC(B1):-single(B1),
                        knownPart(B1),
                        highConfidence(B1),
                        not spatial_partOf(B1,_,_,_).
        
        
    '''
    m = Net()
    nnMapping = {'label': m}

    termPath = 'img ./'+img_path

    # set the classes that we consider
    domain =["Aeroplane", "Animal_Wing", "Arm", "Artifact_Wing", "Beak", "Bicycle", "Bird", "Boat", "Body", "Bodywork", "Bottle", "Bus", "Cap", "Car", "Cat", "Chain_wheel", "Chair", "Coach", "Cow", "Dog", "Door", "Ear", "Ebrow", "Engine", "Eye", "Foot", "Hair", "Hand", "Handlebar", "Head", "Headlight", "Hoof", "Horn", "Horse", "Leg", "License_plate", "Locomotive", "Mirror", "Motorbike", "Mouth", "Muzzle", "Neck", "Nose", "Person", "Plant", "Pot", "Pottedplant", "Saddle", "Screen", "Sheep", "Sofa", "Stern", "Tail", "Torso", "Train", "Tvmonitor", "Wheel", "Window", "Diningtable","Other"]
    objects=["Aeroplane","Bicycle","Bird","Boat","Bottle","Bus","Car", "Cat","Chair","Cow", "Dog","Horse"
    ,"Motorbike","Person","Pottedplant","Sheep", "Sofa","Train", "Tvmonitor", "Diningtable"]

    animals=["Beak","Bird","Cat","Cow","Dog","Foot","Hoof","Horn","Horse","Muzzle",
	"Neck","Leg","Saddle","Sheep","Tail","Torso","Head","Eye","Nose","Ear","Animal_Wing",'Person']
    body=["Body","Ear","Ebrow","Eye","Foot","Hair","Hand","Head","Leg","Mouth","Neck",
	"Nose","Torso","Arm","Person"]
    artifacts=["Aeroplane","Bicycle","Boat","Bodywork","Bottle","Bus","Cap",
	"Car","Chair","Coach","Door","Engine","Handlebar","Headlight","Locomotive","Mirror",
	"Motorbike","Pot","Saddle","Screen","Sofa","Stern","Tail","Train","Tvmonitor","Wheel"
	,"Window","Diningtable","Body","License_plate","Artifact_Wing","Chain_wheel"]

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

    
    aspProgram=spatial_rules+objects_relationships_rules+error_detection_ruleset+cn_knowledge
   
    
    print(aspProgram)

    image_ids,factsList, dataList =termPath2dataList(termPath, img_size, domain,model,"pascal")
    # print(image_ids)
    # print(factsList)
    # print("\n")
    # print(dataList)


    json_dict={}
    error_dict={}
    for idx, facts in enumerate(factsList):
        print(image_ids[idx])
        json_dict[image_ids[idx]]=[]
        NeurASPobj = NeurASP(dprogram + facts, nnMapping, optimizers=None)
        # Find the most probable stable model
        models,top5classes= NeurASPobj.infer(dataDic=dataList[idx], obs='', mvpp=aspProgram + facts,postProcessing=False,stable_models=1)
        models[0],error_dict=erh.resolve_errors(image_ids[idx],error_dict,models[0],top5classes,aspProgram,"CN")

        # print('\nInfernece Result on Data {}:'.format(idx+1))
        # print(models)
        # probs_json=pstr.parse_probs(top5classes)
        # if path.exists(output)==False:
        #     os.mkdir(output)
        # opath=output+'/imgBoxProbabilities'
        # if path.exists(opath)==False:
        #     os.mkdir(opath) 
        # with open(opath+"/"+str(image_ids[idx])+".json", "w") as outfile:
        #     outfile.write(probs_json)
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
    
    # print(error_dict)
    #write json with the inferences
    json_object=json.dumps(json_dict,indent=4)
    if path.exists(output)==False:
        os.mkdir(output)
    opath=output+'/CN_ErrorDetection'
    if path.exists(opath)==False:
        os.mkdir(opath) 
    with open(opath+"/CNED_inferences_JSON.json", "w") as outfile:
        outfile.write(json_object)
    outfile.close()
    error_json=json.dumps(error_dict,indent=4)
    with open(opath+"/detected_errors.json", "w") as outfile:
        outfile.write(error_json)
    outfile.close()