import sys
sys.path.append('../../')

import torch
import json
from dataGen import termPath2dataList
from network import Net
from neurasp import NeurASP
import os.path
from os import path
import string_parser as pstr

def infer(img_path,img_size,model,output):
    dprogram = r'''
    nn(label(1,I,B),["Aeroplane", "Animal_Wing", "Arm", "Artifact_Wing", "Beak", "Bicycle", "Bird", "Boat", "Body", "Bodywork", "Bottle", "Bus", "Cap", "Car", "Cat", "Chain_wheel", "Chair", "Coach", "Cow", "Dog", "Door", "Ear", "Ebrow", "Engine", "Eye", "Foot", "Hair", "Hand", "Handlebar", "Head", "Headlight", "Hoof", "Horn", "Horse", "Leg", "License_plate", "Locomotive", "Mirror", "Motorbike", "Mouth", "Muzzle", "Neck", "Nose", "Person", "Plant", "Pot", "Pottedplant", "Saddle", "Screen", "Sheep", "Sofa", "Stern", "Tail", "Torso", "Train", "Tvmonitor", "Wheel", "Window", "Diningtable","Other"]) :- box(I,B,X1,Y1,X2,Y2,C).

    '''

    aspProgram = r'''


    '''

    objects_relationships_rules=r'''


    %define boxes that have a Spatial partOf relation
    isPartOf(B1,L1,B2,L2):-candidatePartOf(B1,B2),
                                    label(_,_,B1,L1),
                                    label(_,_,B2,L2).
                                                                    
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

    % Define the candidatePartOf predicate
    %Find if two bounding boxes overlap with over 90% coverage
    candidatePartOf(Bmin,Bmax) :- box(_,B1,Xmin1,Ymin1,Xmax1,Ymax1,_),
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

    m = Net()
    nnMapping = {'label': m}

    termPath = 'img ./'+img_path
    
    # set the classes that we consider
    domain =["Aeroplane", "Animal_Wing", "Arm", "Artifact_Wing", "Beak", "Bicycle", "Bird", "Boat", "Body", "Bodywork", "Bottle", "Bus", "Cap", "Car", "Cat", "Chain_wheel", "Chair", "Coach", "Cow", "Dog", "Door", "Ear", "Ebrow", "Engine", "Eye", "Foot", "Hair", "Hand", "Handlebar", "Head", "Headlight", "Hoof", "Horn", "Horse", "Leg", "License_plate", "Locomotive", "Mirror", "Motorbike", "Mouth", "Muzzle", "Neck", "Nose", "Person", "Plant", "Pot", "Pottedplant", "Saddle", "Screen", "Sheep", "Sofa", "Stern", "Tail", "Torso", "Train", "Tvmonitor", "Wheel", "Window", "Diningtable","Other"]
    aspProgram=spatial_rules+objects_relationships_rules

 

    print(aspProgram)

    image_ids,factsList, dataList =termPath2dataList(termPath, img_size, domain,model,"pascal") ##coco OR pascal
    

    json_dict={}
    for idx, facts in enumerate(factsList):
        print(image_ids[idx])
        json_dict[image_ids[idx]]=[]
        NeurASPobj = NeurASP(dprogram + facts, nnMapping, optimizers=None)
        # Find the most probable stable model
        models = NeurASPobj.infer(dataDic=dataList[idx], obs='', mvpp=aspProgram + facts,postProcessing=False,stable_models=1)
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
    json_object=json.dumps(json_dict,indent=4)
    if path.exists(output)==False:
        os.mkdir(output)
    opath=output+'/baseline'
    if path.exists(opath)==False:
        os.mkdir(opath) 
    with open(opath+"/BL_inferences_JSON.json", "w") as outfile:
        outfile.write(json_object)
    outfile.close()
