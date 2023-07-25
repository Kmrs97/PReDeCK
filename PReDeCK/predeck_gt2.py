import sys
sys.path.append('../../')

import torch
import json
from dataGen import termPath2dataList
from network import Net
from neurasp import NeurASP
from examples.PReDeCK import parse_dataset as prd
import string_parser as pstr
import os.path
from os import path

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


    %Transitivity

    %define boxes that have a Spatial partOf relation
    isPartOf(B1,L1,B2,L2):-candidatePartOf(B1,B2),
                                label(_,_,B1,L1),
                                label(_,_,B2,L2),
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

    constraints=r'''

   

  
    '''
    m = Net()
    nnMapping = {'label': m}

    termPath = 'img ./'+img_path

    # set the  classes that we consider
    domain =["Aeroplane", "Animal_Wing", "Arm", "Artifact_Wing", "Beak", "Bicycle", "Bird", "Boat", "Body", "Bodywork", "Bottle", "Bus", "Cap", "Car", "Cat", "Chain_wheel", "Chair", "Coach", "Cow", "Dog", "Door", "Ear", "Ebrow", "Engine", "Eye", "Foot", "Hair", "Hand", "Handlebar", "Head", "Headlight", "Hoof", "Horn", "Horse", "Leg", "License_plate", "Locomotive", "Mirror", "Motorbike", "Mouth", "Muzzle", "Neck", "Nose", "Person", "Plant", "Pot", "Pottedplant", "Saddle", "Screen", "Sheep", "Sofa", "Stern", "Tail", "Torso", "Train", "Tvmonitor", "Wheel", "Window", "Diningtable","Other"]
    
    aspProgram=spatial_rules+objects_relationships_rules+constraints+ontology_ground_truth

    print(aspProgram)

    image_ids,factsList, dataList =termPath2dataList(termPath, img_size, domain,model,"pascal")

    json_dict={}
    for idx, facts in enumerate(factsList):
        print(image_ids[idx])
        json_dict[image_ids[idx]]=[]
        NeurASPobj = NeurASP(dprogram + facts, nnMapping, optimizers=None)
        # Find the most probable stable model
        models = NeurASPobj.infer(dataDic=dataList[idx], obs='', mvpp=aspProgram + facts,postProcessing=False,stable_models=1)
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
    json_object=json.dumps(json_dict,indent=4)
    if path.exists(output)==False:
        os.mkdir(output)
    opath=output+'/groundtruth2'
    if path.exists(opath)==False:
        os.mkdir(opath) 
    with open(opath+"/GT2_inferences_JSON.json", "w") as outfile:
        outfile.write(json_object)
    outfile.close()