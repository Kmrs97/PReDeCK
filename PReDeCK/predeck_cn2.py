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
    dprogram = r'''
    nn(label(1,I,B),["Aeroplane", "Animal_Wing", "Arm", "Artifact_Wing", "Beak", "Bicycle", "Bird", "Boat", "Body", "Bodywork", "Bottle", "Bus", "Cap", "Car", "Cat", "Chain_wheel", "Chair", "Coach", "Cow", "Dog", "Door", "Ear", "Ebrow", "Engine", "Eye", "Foot", "Hair", "Hand", "Handlebar", "Head", "Headlight", "Hoof", "Horn", "Horse", "Leg", "License_plate", "Locomotive", "Mirror", "Motorbike", "Mouth", "Muzzle", "Neck", "Nose", "Person", "Plant", "Pot", "Pottedplant", "Saddle", "Screen", "Sheep", "Sofa", "Stern", "Tail", "Torso", "Train", "Tvmonitor", "Wheel", "Window", "Diningtable","Other"]) :- box(I,B,X1,Y1,X2,Y2,C).

    '''

    aspProgram = r'''
  



    '''

##Uncomment for reasoning to enrich the knowledge domain
##(It is very time-consuming so we materialize the output facts of this process
# and explicitly inserted the system)
    
#  %partOf/3 inverse of hasA/3
#     partOf(L1,L2,C):-hasA(L2,L1,C).

#     %partOf/3 is inverse of hasContext/3
#     partOf(X,Y,C):-hasContext(X,Y,C).

#     %Transitivity
#     partOf(L1,L3,C1):-
#         partOf(L1,L2,C1),
#         partOf(L2,L3,C2),
#         C1=C2.

#     %Symmetricity
#     isSynonymWith(X,Y,C):-isSynonymWith(Y,X,C).

#     %Inference rules

#     partOf(X,Y,C1):-
#         partOf(X,Z,C1),
#         isA(Y,Z,C2),
#         C1=C2.

#     partOf(Z,Y,C):-
#         partOf(X,Y,C),
#         isA(X,Z,C).

#     partOf(X,Y,C):-
#         partOf(X,Z,C),
#         isA(Z,Y,C).

#     partOf(Z, Y,C1) :-
#         partOf(X, Y,C1),
#         isSynonymWith(X,Z,C2),
#         C1=C2.

#     partOf(X,Z,C1) :-
#         partOf(X, Y,C1),
#         isSynonymWith(Y,Z,C2),
#         C1=C2.


    materialized_knowledge=r'''
        partOf("Sprocket","Bicycle","Artifact").
        partOf("Handlebar","Bicycle","Artifact").
        partOf("Saddle","Bicycle","Artifact").
        partOf("Chain_wheel","Bicycle","Artifact").
        partOf("Wheel","Bicycle","Artifact").
        partOf("Bicycle","Bicycle","Artifact").
        partOf("Wing","Bicycle","Artifact").
        partOf("Window","Bicycle","Artifact").
        partOf("Door","Bicycle","Artifact").
        partOf("Pot","Bicycle","Artifact").
        partOf("Car","Bicycle","Artifact").
        partOf("Leg","Bicycle","Artifact").
        partOf("Horn","Bicycle","Artifact").
        partOf("Bodywork","Bicycle","Artifact").
        partOf("Artifact_Wing","Bicycle","Artifact").
        partOf("Boat","Bicycle","Artifact").
        partOf("Engine","Bicycle","Artifact").
        partOf("Mirror","Bicycle","Artifact").
        partOf("Person","Bicycle","Artifact").
        partOf("Screen","Bicycle","Artifact").
        partOf("Body","Bicycle","Artifact").
        partOf("Headlight","Bicycle","Artifact").
        partOf("Stern","Bicycle","Artifact").
        partOf("Tail","Bicycle","Artifact").
        partOf("Chair","Bicycle","Artifact").
        partOf("Arm","Bicycle","Artifact").
        partOf("Foot","Bicycle","Artifact").
        partOf("Dining_table","Bicycle","Artifact").
        partOf("Locomotive","Bicycle","Artifact").
        partOf("Train","Bicycle","Artifact").
        partOf("Mouth","Bicycle","Artifact").
        partOf("Cap","Bicycle","Artifact").
        partOf("Bus","Bicycle","Artifact").
        partOf("Coach","Bicycle","Artifact").
        partOf("Mouth","Bottle","Artifact").
        partOf("Cap","Bottle","Artifact").
        partOf("Person","Bottle","Artifact").
        partOf("Stern","Bottle","Artifact").
        partOf("Window","Bottle","Artifact").
        partOf("Tail","Bottle","Artifact").
        partOf("Dining_table","Bottle","Artifact").
        partOf("Pot","Bottle","Artifact").
        partOf("Car","Bottle","Artifact").
        partOf("Wheel","Bottle","Artifact").
        partOf("Door","Bottle","Artifact").
        partOf("Leg","Bottle","Artifact").
        partOf("Foot","Bottle","Artifact").
        partOf("Arm","Bottle","Artifact").
        partOf("Chair","Bottle","Artifact").
        partOf("Horn","Bottle","Artifact").
        partOf("Wing","Bottle","Artifact").
        partOf("Engine","Bottle","Artifact").
        partOf("Mirror","Bottle","Artifact").
        partOf("Bodywork","Bottle","Artifact").
        partOf("Sprocket","Bottle","Artifact").
        partOf("Handlebar","Bottle","Artifact").
        partOf("Screen","Bottle","Artifact").
        partOf("Saddle","Bottle","Artifact").
        partOf("Chain_wheel","Bottle","Artifact").
        partOf("Locomotive","Bottle","Artifact").
        partOf("Artifact_Wing","Bottle","Artifact").
        partOf("Bicycle","Bottle","Artifact").
        partOf("Body","Bottle","Artifact").
        partOf("Boat","Bottle","Artifact").
        partOf("Train","Bottle","Artifact").
        partOf("Headlight","Bottle","Artifact").
        partOf("Bus","Bottle","Artifact").
        partOf("Coach","Bottle","Artifact").
        partOf("Wing","Bird","Animal").
        partOf("Beak","Bird","Animal").
        partOf("Animal_Wing","Bird","Animal").
        partOf("Nose","Bird","Animal").
        partOf("Mouth","Bird","Animal").
        partOf("Tail","Bird","Animal").
        partOf("Body","Bird","Animal").
        partOf("Head","Bird","Animal").
        partOf("Torso","Bird","Animal").
        partOf("Saddle","Bird","Animal").
        partOf("Person","Bird","Animal").
        partOf("Ear","Bird","Animal").
        partOf("Muzzle","Bird","Animal").
        partOf("Eye","Bird","Animal").
        partOf("Leg","Bird","Animal").
        partOf("Neck","Bird","Animal").
        partOf("Dog","Bird","Animal").
        partOf("Cat","Bird","Animal").
        partOf("Hair","Bird","Animal").
        partOf("Foot","Bird","Animal").
        partOf("Horn","Bird","Animal").
        partOf("Car","Bird","Animal").
        partOf("Hoof","Bird","Animal").
        partOf("Sheep","Bird","Animal").
        partOf("Bird","Bird","Animal").
        partOf("Window","Car","Artifact").
        partOf("Horn","Car","Artifact").
        partOf("Wing","Car","Artifact").
        partOf("Engine","Car","Artifact").
        partOf("Mirror","Car","Artifact").
        partOf("Door","Car","Artifact").
        partOf("Bodywork","Car","Artifact").
        partOf("Wheel","Car","Artifact").
        partOf("Car","Car","Artifact").
        partOf("Bicycle","Car","Artifact").
        partOf("Dining_table","Car","Artifact").
        partOf("Locomotive","Car","Artifact").
        partOf("Artifact_Wing","Car","Artifact").
        partOf("Head","Car","Animal").
        partOf("Stern","Car","Artifact").
        partOf("Body","Car","Animal").
        partOf("Nose","Car","Animal").
        partOf("Person","Car","Artifact").
        partOf("Body","Car","Artifact").
        partOf("Screen","Car","Artifact").
        partOf("Headlight","Car","Artifact").
        partOf("Sprocket","Car","Artifact").
        partOf("Handlebar","Car","Artifact").
        partOf("Leg","Car","Artifact").
        partOf("Chair","Car","Artifact").
        partOf("Arm","Car","Artifact").
        partOf("Saddle","Car","Artifact").
        partOf("Train","Car","Artifact").
        partOf("Chain_wheel","Car","Artifact").
        partOf("Torso","Car","Animal").
        partOf("Tail","Car","Artifact").
        partOf("Hair","Car","Animal").
        partOf("Horn","Car","Animal").
        partOf("Foot","Car","Animal").
        partOf("Beak","Car","Animal").
        partOf("Wing","Car","Animal").
        partOf("Cat","Car","Animal").
        partOf("Dog","Car","Animal").
        partOf("Neck","Car","Animal").
        partOf("Leg","Car","Animal").
        partOf("Eye","Car","Animal").
        partOf("Muzzle","Car","Animal").
        partOf("Ear","Car","Animal").
        partOf("Mouth","Car","Animal").
        partOf("Person","Car","Animal").
        partOf("Boat","Car","Artifact").
        partOf("Pot","Car","Artifact").
        partOf("Foot","Car","Artifact").
        partOf("Bus","Car","Artifact").
        partOf("Coach","Car","Artifact").
        partOf("Mouth","Car","Artifact").
        partOf("Tail","Car","Animal").
        partOf("Animal_Wing","Car","Animal").
        partOf("Cap","Car","Artifact").
        partOf("Sheep","Car","Animal").
        partOf("Car","Car","Animal").
        partOf("Hoof","Car","Animal").
        partOf("Saddle","Car","Animal").
        partOf("Bird","Car","Animal").
        partOf("Window","Bus","Artifact").
        partOf("Car","Bus","Artifact").
        partOf("Wing","Bus","Artifact").
        partOf("Horn","Bus","Artifact").
        partOf("Door","Bus","Artifact").
        partOf("Mirror","Bus","Artifact").
        partOf("Engine","Bus","Artifact").
        partOf("Wheel","Bus","Artifact").
        partOf("Bodywork","Bus","Artifact").
        partOf("Person","Bus","Artifact").
        partOf("Leg","Bus","Artifact").
        partOf("Train","Bus","Artifact").
        partOf("Sprocket","Bus","Artifact").
        partOf("Stern","Bus","Artifact").
        partOf("Bicycle","Bus","Artifact").
        partOf("Locomotive","Bus","Artifact").
        partOf("Dining_table","Bus","Artifact").
        partOf("Artifact_Wing","Bus","Artifact").
        partOf("Chair","Bus","Artifact").
        partOf("Screen","Bus","Artifact").
        partOf("Body","Bus","Artifact").
        partOf("Boat","Bus","Artifact").
        partOf("Saddle","Bus","Artifact").
        partOf("Arm","Bus","Artifact").
        partOf("Handlebar","Bus","Artifact").
        partOf("Headlight","Bus","Artifact").
        partOf("Pot","Bus","Artifact").
        partOf("Mouth","Bus","Artifact").
        partOf("Bus","Bus","Artifact").
        partOf("Coach","Bus","Artifact").
        partOf("Foot","Bus","Artifact").
        partOf("Tail","Bus","Artifact").
        partOf("Chain_wheel","Bus","Artifact").
        partOf("Cap","Bus","Artifact").
        partOf("Hair","Cow","Animal").
        partOf("Wing","Cow","Animal").
        partOf("Hoof","Cow","Animal").
        partOf("Horn","Cow","Animal").
        partOf("Animal_Wing","Cow","Animal").
        partOf("Tail","Cow","Animal").
        partOf("Cat","Cow","Animal").
        partOf("Nose","Cow","Animal").
        partOf("Body","Cow","Animal").
        partOf("Head","Cow","Animal").
        partOf("Car","Cow","Animal").
        partOf("Leg","Cow","Animal").
        partOf("Dog","Cow","Animal").
        partOf("Sheep","Cow","Animal").
        partOf("Torso","Cow","Animal").
        partOf("Foot","Cow","Animal").
        partOf("Beak","Cow","Animal").
        partOf("Mouth","Cow","Animal").
        partOf("Person","Cow","Animal").
        partOf("Neck","Cow","Animal").
        partOf("Eye","Cow","Animal").
        partOf("Muzzle","Cow","Animal").
        partOf("Ear","Cow","Animal").
        partOf("Saddle","Cow","Animal").
        partOf("Bird","Cow","Animal").
        partOf("Nose","Cat","Animal").
        partOf("Body","Cat","Animal").
        partOf("Head","Cat","Animal").
        partOf("Torso","Cat","Animal").
        partOf("Person","Cat","Animal").
        partOf("Mouth","Cat","Animal").
        partOf("Ear","Cat","Animal").
        partOf("Muzzle","Cat","Animal").
        partOf("Eye","Cat","Animal").
        partOf("Leg","Cat","Animal").
        partOf("Neck","Cat","Animal").
        partOf("Dog","Cat","Animal").
        partOf("Cat","Cat","Animal").
        partOf("Wing","Cat","Animal").
        partOf("Beak","Cat","Animal").
        partOf("Foot","Cat","Animal").
        partOf("Horn","Cat","Animal").
        partOf("Hair","Cat","Animal").
        partOf("Animal_Wing","Cat","Animal").
        partOf("Tail","Cat","Animal").
        partOf("Hoof","Cat","Animal").
        partOf("Car","Cat","Animal").
        partOf("Sheep","Cat","Animal").
        partOf("Saddle","Cat","Animal").
        partOf("Bird","Cat","Animal").
        partOf("Hoof","Horse","Animal").
        partOf("Nose","Horse","Animal").
        partOf("Body","Horse","Animal").
        partOf("Head","Horse","Animal").
        partOf("Torso","Horse","Animal").
        partOf("Hair","Horse","Animal").
        partOf("Person","Horse","Animal").
        partOf("Mouth","Horse","Animal").
        partOf("Ear","Horse","Animal").
        partOf("Muzzle","Horse","Animal").
        partOf("Eye","Horse","Animal").
        partOf("Leg","Horse","Animal").
        partOf("Neck","Horse","Animal").
        partOf("Dog","Horse","Animal").
        partOf("Cat","Horse","Animal").
        partOf("Wing","Horse","Animal").
        partOf("Beak","Horse","Animal").
        partOf("Foot","Horse","Animal").
        partOf("Horn","Horse","Animal").
        partOf("Animal_Wing","Horse","Animal").
        partOf("Tail","Horse","Animal").
        partOf("Car","Horse","Animal").
        partOf("Sheep","Horse","Animal").
        partOf("Saddle","Horse","Animal").
        partOf("Bird","Horse","Animal").
        partOf("Tail","Dog","Animal").
        partOf("Head","Dog","Animal").
        partOf("Body","Dog","Animal").
        partOf("Nose","Dog","Animal").
        partOf("Muzzle","Dog","Animal").
        partOf("Ear","Dog","Animal").
        partOf("Torso","Dog","Animal").
        partOf("Hair","Dog","Animal").
        partOf("Horn","Dog","Animal").
        partOf("Foot","Dog","Animal").
        partOf("Beak","Dog","Animal").
        partOf("Wing","Dog","Animal").
        partOf("Cat","Dog","Animal").
        partOf("Dog","Dog","Animal").
        partOf("Neck","Dog","Animal").
        partOf("Leg","Dog","Animal").
        partOf("Eye","Dog","Animal").
        partOf("Mouth","Dog","Animal").
        partOf("Person","Dog","Animal").
        partOf("Animal_Wing","Dog","Animal").
        partOf("Car","Dog","Animal").
        partOf("Hoof","Dog","Animal").
        partOf("Sheep","Dog","Animal").
        partOf("Saddle","Dog","Animal").
        partOf("Bird","Dog","Animal").
        partOf("Mouth","Person","Body").
        partOf("Mouth","Person","Animal").
        partOf("Eye","Person","Body").
        partOf("Eye","Person","Animal").
        partOf("Nose","Person","Body").
        partOf("Nose","Person","Animal").
        partOf("Eyebrow","Person","Body").
        partOf("Person","Person","Person").
        partOf("Body","Person","Body").
        partOf("Horn","Person","Animal").
        partOf("Foot","Person","Animal").
        partOf("Body","Person","Animal").
        partOf("Torso","Person","Body").
        partOf("Ebrow","Person","Body").
        partOf("Dog","Person","Animal").
        partOf("Cat","Person","Animal").
        partOf("Dog","Person","Body").
        partOf("Cat","Person","Body").
        partOf("Wing","Person","Animal").
        partOf("Head","Person","Animal").
        partOf("Leg","Person","Artifact").
        partOf("Hair","Person","Body").
        partOf("Person","Person","Animal").
        partOf("Hair","Person","Animal").
        partOf("Person","Person","Body").
        partOf("Arm","Person","Body").
        partOf("Leg","Person","Body").
        partOf("Head","Person","Body").
        partOf("Neck","Person","Body").
        partOf("Ear","Person","Body").
        partOf("Foot","Person","Body").
        partOf("Hand","Person","Body").
        partOf("Ear","Person","Animal").
        partOf("Muzzle","Person","Animal").
        partOf("Car","Person","Animal").
        partOf("Torso","Person","Animal").
        partOf("Neck","Person","Animal").
        partOf("Leg","Person","Animal").
        partOf("Tail","Person","Body").
        partOf("Stern","Person","Body").
        partOf("Animal_Wing","Person","Animal").
        partOf("Door","Person","Artifact").
        partOf("Tail","Person","Animal").
        partOf("Wheel","Person","Artifact").
        partOf("Beak","Person","Animal").
        partOf("Window","Person","Artifact").
        partOf("Foot","Person","Artifact").
        partOf("Hoof","Person","Animal").
        partOf("Sheep","Person","Animal").
        partOf("Screen","Person","Artifact").
        partOf("Car","Person","Artifact").
        partOf("Wing","Person","Artifact").
        partOf("Horn","Person","Artifact").
        partOf("Bicycle","Person","Artifact").
        partOf("Pot","Person","Artifact").
        partOf("Stern","Person","Artifact").
        partOf("Mouth","Person","Artifact").
        partOf("Chair","Person","Artifact").
        partOf("Arm","Person","Artifact").
        partOf("Tail","Person","Artifact").
        partOf("Saddle","Person","Artifact").
        partOf("Sprocket","Person","Artifact").
        partOf("Handlebar","Person","Artifact").
        partOf("Headlight","Person","Artifact").
        partOf("Body","Person","Artifact").
        partOf("Boat","Person","Artifact").
        partOf("Train","Person","Artifact").
        partOf("Chain_wheel","Person","Artifact").
        partOf("Bodywork","Person","Artifact").
        partOf("Mirror","Person","Artifact").
        partOf("Engine","Person","Artifact").
        partOf("Person","Person","Artifact").
        partOf("Dining_table","Person","Artifact").
        partOf("Locomotive","Person","Artifact").
        partOf("Artifact_Wing","Person","Artifact").
        partOf("Bus","Person","Artifact").
        partOf("Coach","Person","Artifact").
        partOf("Cap","Person","Artifact").
        partOf("Saddle","Person","Animal").
        partOf("Bird","Person","Animal").
        partOf("Bodywork","Motorbike","Artifact").
        partOf("Tail","Motorbike","Artifact").
        partOf("Person","Motorbike","Artifact").
        partOf("Wheel","Motorbike","Artifact").
        partOf("Engine","Motorbike","Artifact").
        partOf("Mirror","Motorbike","Artifact").
        partOf("Door","Motorbike","Artifact").
        partOf("Horn","Motorbike","Artifact").
        partOf("Sprocket","Motorbike","Artifact").
        partOf("Handlebar","Motorbike","Artifact").
        partOf("Headlight","Motorbike","Artifact").
        partOf("Window","Motorbike","Artifact").
        partOf("Body","Motorbike","Artifact").
        partOf("Car","Motorbike","Artifact").
        partOf("Screen","Motorbike","Artifact").
        partOf("Wing","Motorbike","Artifact").
        partOf("Saddle","Motorbike","Artifact").
        partOf("Chain_wheel","Motorbike","Artifact").
        partOf("Leg","Motorbike","Artifact").
        partOf("Bicycle","Motorbike","Artifact").
        partOf("Pot","Motorbike","Artifact").
        partOf("Mouth","Motorbike","Artifact").
        partOf("Stern","Motorbike","Artifact").
        partOf("Foot","Motorbike","Artifact").
        partOf("Chair","Motorbike","Artifact").
        partOf("Arm","Motorbike","Artifact").
        partOf("Dining_table","Motorbike","Artifact").
        partOf("Locomotive","Motorbike","Artifact").
        partOf("Artifact_Wing","Motorbike","Artifact").
        partOf("Train","Motorbike","Artifact").
        partOf("Coach","Motorbike","Artifact").
        partOf("Bus","Motorbike","Artifact").
        partOf("Boat","Motorbike","Artifact").
        partOf("Cap","Motorbike","Artifact").
        partOf("Window","Train","Artifact").
        partOf("Sprocket","Train","Artifact").
        partOf("Car","Train","Artifact").
        partOf("Person","Train","Artifact").
        partOf("Horn","Train","Artifact").
        partOf("Wing","Train","Artifact").
        partOf("Engine","Train","Artifact").
        partOf("Mirror","Train","Artifact").
        partOf("Door","Train","Artifact").
        partOf("Boat","Train","Artifact").
        partOf("Wheel","Train","Artifact").
        partOf("Bodywork","Train","Artifact").
        partOf("Screen","Train","Artifact").
        partOf("Body","Train","Artifact").
        partOf("Headlight","Train","Artifact").
        partOf("Bicycle","Train","Artifact").
        partOf("Handlebar","Train","Artifact").
        partOf("Chain_wheel","Train","Artifact").
        partOf("Saddle","Train","Artifact").
        partOf("Stern","Train","Artifact").
        partOf("Train","Train","Artifact").
        partOf("Dining_table","Train","Artifact").
        partOf("Locomotive","Train","Artifact").
        partOf("Artifact_Wing","Train","Artifact").
        partOf("Chair","Train","Artifact").
        partOf("Arm","Train","Artifact").
        partOf("Leg","Train","Artifact").
        partOf("Coach","Train","Artifact").
        partOf("Bus","Train","Artifact").
        partOf("Pot","Train","Artifact").
        partOf("Tail","Train","Artifact").
        partOf("Foot","Train","Artifact").
        partOf("Mouth","Train","Artifact").
        partOf("Cap","Train","Artifact").
        partOf("Horn","Sheep","Animal").
        partOf("Nose","Sheep","Animal").
        partOf("Body","Sheep","Animal").
        partOf("Head","Sheep","Animal").
        partOf("Torso","Sheep","Animal").
        partOf("Car","Sheep","Animal").
        partOf("Person","Sheep","Animal").
        partOf("Mouth","Sheep","Animal").
        partOf("Ear","Sheep","Animal").
        partOf("Muzzle","Sheep","Animal").
        partOf("Eye","Sheep","Animal").
        partOf("Leg","Sheep","Animal").
        partOf("Neck","Sheep","Animal").
        partOf("Dog","Sheep","Animal").
        partOf("Cat","Sheep","Animal").
        partOf("Wing","Sheep","Animal").
        partOf("Beak","Sheep","Animal").
        partOf("Foot","Sheep","Animal").
        partOf("Hair","Sheep","Animal").
        partOf("Animal_Wing","Sheep","Animal").
        partOf("Tail","Sheep","Animal").
        partOf("Hoof","Sheep","Animal").
        partOf("Person","Sheep","Person").
        partOf("Sheep","Sheep","Animal").
        partOf("Saddle","Sheep","Animal").
        partOf("Bird","Sheep","Animal").
        partOf("Wing","Aeroplane","Artifact").
        partOf("Tail","Aeroplane","Artifact").
        partOf("Car","Aeroplane","Artifact").
        partOf("Screen","Aeroplane","Artifact").
        partOf("Body","Aeroplane","Artifact").
        partOf("Artifact_Wing","Aeroplane","Artifact").
        partOf("Wheel","Aeroplane","Artifact").
        partOf("Stern","Aeroplane","Artifact").
        partOf("Window","Aeroplane","Artifact").
        partOf("Leg","Aeroplane","Artifact").
        partOf("Foot","Aeroplane","Artifact").
        partOf("Door","Aeroplane","Artifact").
        partOf("Mirror","Aeroplane","Artifact").
        partOf("Engine","Aeroplane","Artifact").
        partOf("Horn","Aeroplane","Artifact").
        partOf("Bodywork","Aeroplane","Artifact").
        partOf("Bicycle","Aeroplane","Artifact").
        partOf("Boat","Aeroplane","Artifact").
        partOf("Train","Aeroplane","Artifact").
        partOf("Person","Aeroplane","Artifact").
        partOf("Headlight","Aeroplane","Artifact").
        partOf("Handlebar","Aeroplane","Artifact").
        partOf("Sprocket","Aeroplane","Artifact").
        partOf("Chair","Aeroplane","Artifact").
        partOf("Arm","Aeroplane","Artifact").
        partOf("Saddle","Aeroplane","Artifact").
        partOf("Chain_wheel","Aeroplane","Artifact").
        partOf("Pot","Aeroplane","Artifact").
        partOf("Mouth","Aeroplane","Artifact").
        partOf("Dining_table","Aeroplane","Artifact").
        partOf("Locomotive","Aeroplane","Artifact").
        partOf("Coach","Aeroplane","Artifact").
        partOf("Bus","Aeroplane","Artifact").
        partOf("Cap","Aeroplane","Artifact").

    '''
    objects_relationships_rules=r'''
    %define boxes that contain objects
    objectBox(B,L,C):-label(_,_,B,L),object(L,C).

    %define boxes that contain parts
    partBox(B,L,C):-label(_,_,B,L),part(L,C).


    %define boxes that have a Spatial partOf relation
    isPartOf(B1,L1,B2,L2):-candidatePartOf(B1,B2),
                    label(_,_,B1,L1),
                    label(_,_,B2,L2),
                    partOf(L1,L2,_).
                                                                        
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


    aspProgram=spatial_rules+objects_relationships_rules+materialized_knowledge

    image_ids,factsList, dataList =termPath2dataList(termPath, img_size, domain,model,"pascal") ##coco OR pascal
   


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
    #     
    json_object=json.dumps(json_dict,indent=4)
    if path.exists(output)==False:
        os.mkdir(output)
    opath=output+'/conceptnet2'
    if path.exists(opath)==False:
        os.mkdir(opath) 
    with open(opath+"/CN2_inferences_JSON.json", "w") as outfile:
        outfile.write(json_object)
    outfile.close()
