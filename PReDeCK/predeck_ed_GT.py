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

    ontology_ground_truth=r'''
    
    %%PartOf

    partOf("Eye","Cow").
    partOf("Head","Cow").
    partOf("Leg","Cow").
    partOf("Neck","Cow").
    partOf("Torso","Cow").

    partOf("Eye","Bird").
    partOf("Head","Bird").
    partOf("Leg","Bird").
    partOf("Neck","Bird").
    partOf("Torso","Bird").

    partOf("Eye","Cat").
    partOf("Head","Cat").
    partOf("Leg","Cat").
    partOf("Neck","Cat").
    partOf("Torso","Cat").

    partOf("Eye","Person").
    partOf("Head","Person").
    partOf("Leg","Person").
    partOf("Neck","Person").
    partOf("Torso","Person").

    partOf("Eye","Horse").
    partOf("Head","Horse").
    partOf("Leg","Horse").
    partOf("Neck","Horse").
    partOf("Torso","Horse").

    partOf("Eye","Sheep").
    partOf("Head","Sheep").
    partOf("Leg","Sheep").
    partOf("Neck","Sheep").
    partOf("Torso","Sheep").

    partOf("Eye","Dog").
    partOf("Head","Dog").
    partOf("Leg","Dog").
    partOf("Neck","Dog").
    partOf("Torso","Dog").


    partOf("Animal_Wing","Bird").
    partOf("Beak","Bird").
    partOf("Tail","Bird").

    partOf("Ear","Cat").
    partOf("Tail","Cat").

    partOf("Ear","Cow").
    partOf("Horn","Cow").
    partOf("Muzzle","Cow").
    partOf("Tail","Cow").

    partOf("Ear","Sheep").
    partOf("Horn","Sheep").
    partOf("Muzzle","Sheep").
    partOf("Tail","Sheep").

    partOf("Ear","Dog").
    partOf("Muzzle","Dog").
    partOf("Tail","Dog").
    partOf("Nose","Dog").

    partOf("Ear","Horse").
    partOf("Muzzle","Horse").
    partOf("Tail","Horse").
    partOf("Hoof","Horse").

    partOf("Arm","Person").
    partOf("Ear","Person").
    partOf("Ebrow","Person").
    partOf("Foot","Person").
    partOf("Hair","Person").
    partOf("Hand","Person").
    partOf("Mouth","Person").
    partOf("Nose","Person").

    partOf("Body","Bottle").
    partOf("Cap","Bottle").

    partOf("Plant","Pottedplant").
    partOf("Pot","Pottedplant").
    partOf("Screen","Tvmonitor").

    partOf("Engine","Aeroplane").
    partOf("Artifact_Wing","Aeroplane").
    partOf("Wheel","Aeroplane").
    partOf("Body","Aeroplane").
    partOf("Stern","Aeroplane").

    partOf("Chain_wheel","Bicycle").
    partOf("Handlebar","Bicycle").
    partOf("Headlight","Bicycle").
    partOf("Saddle","Bicycle").
    partOf("Wheel","Bicycle").

    partOf("Bodywork","Bus").
    partOf("Door","Bus").
    partOf("Headlight","Bus").
    partOf("License_plate","Bus").
    partOf("Mirror","Bus").
    partOf("Wheel","Bus").
    partOf("Window","Bus").

    partOf("Bodywork","Car").
    partOf("Door","Car").
    partOf("Headlight","Car").
    partOf("License_plate","Car").
    partOf("Mirror","Car").
    partOf("Wheel","Car").
    partOf("Window","Car").


    partOf("Headlight","Motorbike").
    partOf("Saddle","Motorbike").
    partOf("Wheel","Motorbike").
    partOf("Handlebar","Motorbike").

    partOf("Coach","Train").
    partOf("Headlight","Train").
    partOf("Locomotive","Train").


    '''
    objects_relationships_rules=r'''


    %define boxes that contain objects
    objectBox(B,L,Conf):-label(_,_,B,L),object(L),box(_,B,_,_,_,_,Conf).

    %define boxes that contain parts
    partBox(B,L,Conf):-label(_,_,B,L),part(L),box(_,B,_,_,_,_,Conf).


    %define boxes that have a Spatial partOf relation
    spatial_partOf(B1,Lbl1,B2,Lbl2):-over90(B1,B2),
                    partBox(B1,Lbl1,_),
                    objectBox(B2,Lbl2,_).


    %Rule to infer the Actual Part Of between two boxes,considering spatial AND semantic relation                              
    semantic_partOf(B1,L1,B2,L2):-spatial_partOf(B1,L1,B2,L2),
                                   partOf(L1,L2).
                                                                                                  
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

    # error_detection_ruleset=r'''
    # minConfidence(B1,B2,B1):- box(_,B1,_,_,_,_,C1),
    #                        box(_,B2,_,_,_,_,C2),
    #                        B1!=B2,
    #                        C1<C2,
    #                        C1<=450.

    #  :-spatial_partOf(B1,L1,B2,L2),
    #       not semantic_partOf(B1,_,_,_),
    #       partOf(L1,_),
    #       partOf(_,L2),
    #       box(_,B1,_,_,_,_,_),
    #       box(_,B2,_,_,_,_,_),
    #       minConfidence(B1,B2,_).

    # :-spatial_partOf(B1,L1,B2,L2),
    #       not semantic_partOf(B1,_,_,_),
    #       partOf(L1,_),
    #       partOf(_,L2),
    #       box(_,B1,_,_,_,_,_),
    #       box(_,B2,_,_,_,_,_),
    #       minConfidence(B2,B1,_).

  
    # '''
    
    error_detection_ruleset=r'''
        %%PRELIMINARIES

        highConfidence(B):-box(_,B,_,_,_,_,C),
                        C >= 400.

        hasNumberOfAccParts(Y,N):-N='''+hashtag+r'''count {(X,Y):semantic_partOf(X,_,Y,_),partBox(X,_,C),C>=400},objectBox(Y,_,_).

        context(B):-hasNumberOfAccParts(B,N), N >= 2.

        single(B):-partBox(B,_,_), not semantic_partOf(B,_,_,_).

        knownPart(B):-partBox(B,L,_),partOf(L,_).
        knownObject(B):-objectBox(B,L,_),partOf(_,L).

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

    # set the  classes that we consider
    domain =["Aeroplane", "Animal_Wing", "Arm", "Artifact_Wing", "Beak", "Bicycle", "Bird", "Boat", "Body", "Bodywork", "Bottle", "Bus", "Cap", "Car", "Cat", "Chain_wheel", "Chair", "Coach", "Cow", "Dog", "Door", "Ear", "Ebrow", "Engine", "Eye", "Foot", "Hair", "Hand", "Handlebar", "Head", "Headlight", "Hoof", "Horn", "Horse", "Leg", "License_plate", "Locomotive", "Mirror", "Motorbike", "Mouth", "Muzzle", "Neck", "Nose", "Person", "Plant", "Pot", "Pottedplant", "Saddle", "Screen", "Sheep", "Sofa", "Stern", "Tail", "Torso", "Train", "Tvmonitor", "Wheel", "Window", "Diningtable","Other"]
    objects=["Aeroplane","Bicycle","Bird","Boat","Bottle","Bus","Car", "Cat","Chair","Cow", "Dog","Horse"
             ,"Motorbike","Person","Pottedplant","Sheep", "Sofa","Train", "Tvmonitor", "Diningtable"]

    parts=[x for x in domain if x not in objects and x!="Other"]

    # Rules to separate objects and parts
    for o in objects:
        rule='object("'+o+'").'
        objects_relationships_rules+=rule+'\n'

    for p in parts:
        rule='part("'+p+'").'
        objects_relationships_rules+=rule+'\n'
    
    aspProgram=spatial_rules+objects_relationships_rules+error_detection_ruleset+ontology_ground_truth
   
    
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
        models[0],error_dict=erh.resolve_errors(image_ids[idx],error_dict,models[0],top5classes,aspProgram,'Ontology')

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
    opath=output+'/gtKnowledge_ErrorDetection'
    if path.exists(opath)==False:
        os.mkdir(opath) 
    with open(opath+"/GTED_inferences_JSON.json", "w") as outfile:
        outfile.write(json_object)
    outfile.close()
    error_json=json.dumps(error_dict,indent=4)
    with open(opath+"/detected_errors.json", "w") as outfile:
        outfile.write(error_json)
    outfile.close()