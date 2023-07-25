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
## partOf relations directly accesible from ConceptNet
    aspProgram = r'''
    partOf("Wing","Division").
partOf("Wing","Insect").
partOf("Wing","Airplane").
partOf("Wing","Angel").
partOf("Wing","Bat").
partOf("Wing","Bird").
partOf("Wing","An_airplane").
partOf("Fender","Locomotive").
partOf("Fender","Car").
partOf("Bird","Flock").
partOf("Organ","Body").
partOf("Sprocket","Bicycle").
partOf("Sprocket","Camera").
partOf("Sprocket","Bike").
partOf("Wheel","Bicycle").
partOf("Wheel","Car").
partOf("Wheel","Wheeled_vehicle").
partOf("Wheel","Vehicle").
partOf("Wheel","Skateboard").
partOf("Wheel","Wagon").
partOf("Wheel","Motorcyle").
partOf("Arm","Body").
partOf("Arm","Armchair").
partOf("Arm","Homo").
partOf("Arm","Person").
partOf("Arm","Chair").
partOf("Body","Narration").
partOf("Body","Address").
partOf("Leg","Table").
partOf("Leg","Chair").
partOf("Leg","Body").
partOf("Leg","Hospital_bed").
partOf("Leg","Spinning_wheel").
partOf("Leg","Pair_of_trousers").
partOf("Leg","Tripod").
partOf("Leg","Cot").
partOf("Leg","Grand_piano").
partOf("Leg","Four_poster").
partOf("Leg","The_body").
partOf("Leg","Spider").
partOf("Hand","The_body").
partOf("Hand","Arm").
partOf("Hand","Homo").
partOf("Hand","Timepiece").
partOf("Hand","An_arm").
partOf("Hand","Human").
partOf("Person","Society").
partOf("Chair","Funiture").
partOf("Beak","Bird").
partOf("Mouth","Face").
partOf("Mouth","Bottle").
partOf("Mouth","Jar").
partOf("Mouth","Head").
partOf("Mouth","Bell").
partOf("Nose","Head").
partOf("Nose","Upper_respiratory_tract").
partOf("Nose","Missile").
partOf("Nose","Face").
partOf("Nose","Aircraft").
partOf("Nose","Your_body").
partOf("Chain","Bicycle").
partOf("Vessel","Body").
partOf("Torso","Body").
partOf("Torso","The_body").
partOf("Bodywork","Motor_vehicle").
partOf("Bottle","Bottle_of_wine").
partOf("Casing","Window").
partOf("Bus","The_fleet").
partOf("Car","Airship").
partOf("Car","Elevator").
partOf("Car","Train").
partOf("Cap","Fungus").
partOf("Plant_part","Plant").
partOf("Hood","Plant").
partOf("Hood","Car").
partOf("Coach","Football_team").
partOf("Wheels","Car").
partOf("Train","Gown").
partOf("Horn","Car").
partOf("Horn","Bull").
partOf("Horn","Ram").
partOf("Horn","Goat").
partOf("Horn","Stock_saddle").
partOf("Horn","Auto").
partOf("Roof","Car").
partOf("Roof","Bus").
partOf("Dog","The_animal_kingdom").
partOf("Hair","Your_head").
partOf("Hair","Integumentary_system").
partOf("Hair","Mammal").
partOf("Hair","Human").
partOf("Hair","The_head_of_a_typical_person").
partOf("Legs","Dog").
partOf("Cow","Cattle").
partOf("Cow","Heard").
partOf("Door","Doorway").
partOf("Door","House").
partOf("Door","Transportation_vehicle").
partOf("Window","Building").
partOf("Window","Window_envelope").
partOf("Window","Bus").
partOf("Window","Computer_screen").
partOf("Window","Car").
partOf("Handle","Door").
partOf("Handle","Handlebar").
partOf("Car_door","Car").
partOf("Ear","Vestibular_apparatus").
partOf("Ear","Auditory_system").
partOf("Ear","Corn").
partOf("Ear","Head").
partOf("Head","An_animal").
partOf("Head","Body").
partOf("Head","Nail").
partOf("Head","Abscess").
partOf("Head","Coin").
partOf("Head","Pin").
partOf("Head","People_").
partOf("Head","Screw").
partOf("Head","Skeletal_muscle").
partOf("Head","Ram").
partOf("Head","Bolt").
partOf("Head","Hammer").
partOf("Head","Animal").
partOf("Head","Human_body").
partOf("Head","Match").
partOf("Head","X_bar").
partOf("Head","The_penis").
partOf("Eyebrow","Face").
partOf("Face","Head").
partOf("Eye","Head").
partOf("Eye","Face").
partOf("Eye","Visual_system").
partOf("Eye","Needle").
partOf("Engine","Car_").
partOf("Engine","An_automobile").
partOf("Engine","Motorcycle").
partOf("Engine","Train").
partOf("Engine","Lawnmower").
partOf("Foot","Body").
partOf("Foot","Leg").
partOf("Foot","Homo").
partOf("Foot","Invertebrate").
partOf("Foot","Yard").
partOf("Foot","The_human_body").
partOf("Foot","Mile").
partOf("Foot","Chair").
partOf("Vertebrate_foot","Leg").
partOf("Inch","Foot").
partOf("Side","Torso").
partOf("Handlebar","Bicycle").
partOf("Mind","Person").
partOf("Human_body","Person").
partOf("Brain","Person").
partOf("Brain","Head").
partOf("Headlight","Electrical_system").
partOf("Hoof","Ungulate").
partOf("Boot","Car").
partOf("Automobile_horn","Car").
partOf("Pommel","Saddle").
partOf("Stallion","Horse").
partOf("License_plate","Radio").
partOf("License_plate","Radio").
partOf("Glass","Window").
partOf("Muzzle","Head").
partOf("Neck","Garment").
partOf("Neck","Body").
partOf("Neck","Cello").
partOf("Neck","Guitar").
partOf("Part","Hair").
partOf("Nape","Neck").
partOf("Plant","An_ecosystem").
partOf("Plant","The_garden").
partOf("Plant","Nature").
partOf("Belly","Torso").
partOf("Saddle","Back").
partOf("Saddle","Domestic_fowl").
partOf("Saddle","Shoe").
partOf("Saddle","Cello").
partOf("Back","Chair").
partOf("Back","Torso").
partOf("Bicycle_seat","Bicycle").
partOf("Screen","Screen_door").
partOf("Screen","Monitor").
partOf("Screen","Cathode_ray_tube").
partOf("Screen","Tv").
partOf("Sheep","Flock").
partOf("Stern","Ship").
partOf("Stern","Sailboat").
partOf("Buttocks","Torso").
partOf("Tail","Vertebrate").
partOf("Tail","Coin").
partOf("Tail","Cat").
partOf("Tail","Fuselage").
partOf("Reverse","Car").
partOf("Dock","Tail").
partOf("Trunk","Car").
partOf("Gearing","Engine").
partOf("Steering_wheel","Car").
partOf("Rim","Wheel").
partOf("Windowpane","Window").
partOf("Wing","Division").
partOf("Wing","Insect").
partOf("Wing","Airplane").
partOf("Wing","Angel").
partOf("Wing","Bat").
partOf("Wing","An_airplane").
partOf("Angel","Venezuela").
partOf("Fender","Locomotive").
partOf("Fender","Car").
partOf("Fender","Auto").
partOf("Fender","Automobile").
partOf("Fly","Garment").
partOf("Annex","Building").
partOf("Bird","Flock").
partOf("Organ","Body").
partOf("Sprocket","Camera").
partOf("Sprocket","Bike").
partOf("Wheel","Wheeled_vehicle").
partOf("Wheel","Vehicle").
partOf("Wheel","Skateboard").
partOf("Wheel","Wagon").
partOf("Wheel","Motorcyle").
partOf("Arm","Armchair").
partOf("Arm","Homo").
partOf("Body","Narration").
partOf("Body","Address").
partOf("Leg","Table").
partOf("Leg","Hospital_bed").
partOf("Leg","Spinning_wheel").
partOf("Leg","Pair_of_trousers").
partOf("Leg","Tripod").
partOf("Leg","Cot").
partOf("Leg","Grand_piano").
partOf("Leg","Four_poster").
partOf("Leg","The_body").
partOf("Leg","Spider").
partOf("Armrest","Car_door").
partOf("Sleeve","Garment").
partOf("Hand","The_body").
partOf("Hand","Homo").
partOf("Hand","Timepiece").
partOf("Hand","An_arm").
partOf("Hand","Human").
partOf("Person","Society").
partOf("Chair","Funiture").
partOf("Mouth","Face").
partOf("Mouth","Jar").
partOf("Mouth","Bell").
partOf("Nose","Upper_respiratory_tract").
partOf("Nose","Missile").
partOf("Nose","Face").
partOf("Nose","Aircraft").
partOf("Nose","Your_body").
partOf("Body_part","Human_body").
partOf("Chain","Chain_printer").
partOf("Chain","Molecule").
partOf("Chain","Chain_tongs").
partOf("Chain","Bicycle").
partOf("Animal","An_ecology").
partOf("Animal","Nature").
partOf("Animal","An_ecosystem").
partOf("Vessel","Vascular_system").
partOf("Vessel","Body").
partOf("Address","Letter").
partOf("Torso","The_body").
partOf("Bodywork","Motor_vehicle").
partOf("Bottle","Bottle_of_wine").
partOf("Casing","Window").
partOf("Casing","Pneumatic_tire").
partOf("Casing","Doorway").
partOf("Bus","The_fleet").
partOf("Car","Airship").
partOf("Car","Elevator").
partOf("Cap","Fungus").
partOf("Plant_part","Plant").
partOf("Hood","Airplane").
partOf("Hood","Plant").
partOf("Hood","Car").
partOf("Hood","Auto").
partOf("Hood","Automobile").
partOf("Top","Ship").
partOf("Lid","Jar").
partOf("Seat","Pair_of_trousers").
partOf("Elevator","Horizontal_tail").
partOf("Elevator","Building").
partOf("Coach","Football_team").
partOf("Wheels","Car").
partOf("Auto","Automatic_transmission").
partOf("Train","Gown").
partOf("Horn","Bull").
partOf("Horn","Ram").
partOf("Horn","Goat").
partOf("Horn","Stock_saddle").
partOf("Horn","Auto").
partOf("Roof","Truck").
partOf("Roof","Car").
partOf("Roof","Cave").
partOf("Roof","Bus").
partOf("Roof","Building").
partOf("Roof","House").
partOf("Dog","The_animal_kingdom").
partOf("Hair","Your_head").
partOf("Hair","Integumentary_system").
partOf("Hair","Mammal").
partOf("Hair","Human").
partOf("Hair","The_head_of_a_typical_person").
partOf("Legs","Dog").
partOf("Trainer","Football_team").
partOf("Cow","Cattle").
partOf("Cow","Heard").
partOf("Door","Doorway").
partOf("Door","House").
partOf("Door","Transportation_vehicle").
partOf("Window","Building").
partOf("Window","Window_envelope").
partOf("Window","Computer_screen").
partOf("Doorway","Wall").
partOf("Room","Building").
partOf("Handle","Door").
partOf("Handle","Handcart").
partOf("Handle","Spatula").
partOf("Handle","Faucet").
partOf("Handle","Umbrella").
partOf("Handle","Edge_tool").
partOf("Handle","Carrycot").
partOf("Handle","Carpet_beater").
partOf("Handle","Handlebar").
partOf("Handle","Hammer").
partOf("Handle","Watering_can").
partOf("Handle","Baseball_bat").
partOf("Handle","Coffee_cup").
partOf("Handle","Teacup").
partOf("Handle","Hand_tool").
partOf("Handle","Saucepan").
partOf("Handle","Frying_pan").
partOf("Handle","Racket").
partOf("Handle","Cricket_bat").
partOf("Handle","Aspergill").
partOf("Handle","Cutlery").
partOf("Handle","Cheese_cutter").
partOf("Handle","Briefcase").
partOf("Handle","Coffeepot").
partOf("Handle","Brush").
partOf("Handle","Handbarrow").
partOf("Handle","Ladle").
partOf("Handle","Mug").
partOf("Handle","Baggage").
partOf("Handle","Handset").
partOf("Handle","Coffee_mug").
partOf("Handle","Shovel").
partOf("Handle","An_object").
partOf("House","Street").
partOf("House","Someone_s_assets").
partOf("Car_door","Car").
partOf("Ear","Vestibular_apparatus").
partOf("Ear","Auditory_system").
partOf("Ear","Corn").
partOf("Head","An_animal").
partOf("Head","Nail").
partOf("Head","Abscess").
partOf("Head","Coin").
partOf("Head","Pin").
partOf("Head","People_").
partOf("Head","Screw").
partOf("Head","Skeletal_muscle").
partOf("Head","Ram").
partOf("Head","Bolt").
partOf("Head","Hammer").
partOf("Head","Animal").
partOf("Head","Human_body").
partOf("Head","Match").
partOf("Head","X_bar").
partOf("Head","The_penis").
partOf("Spike","Shoe").
partOf("Eyebrow","Face").
partOf("Face","Watch").
partOf("Face","Playing_card").
partOf("Face","Animal").
partOf("Face","Clock").
partOf("Face","Head").
partOf("Face","Homo").
partOf("Face","Golf_club_head").
partOf("Face","Racket").
partOf("Face","Me").
partOf("Face","The_body").
partOf("Eye","Face").
partOf("Eye","Visual_system").
partOf("Eye","Needle").
partOf("Engine","Car_").
partOf("Engine","An_automobile").
partOf("Engine","Motorcycle").
partOf("Engine","Lawnmower").
partOf("Eyelet","Boot").
partOf("Eyelet","Garment").
partOf("Foot","Homo").
partOf("Foot","Invertebrate").
partOf("Foot","Yard").
partOf("Foot","The_human_body").
partOf("Foot","Mile").
partOf("Foundation","Building").
partOf("Vertebrate_foot","Leg").
partOf("Vertebrate_foot","Vertebrate").
partOf("Yard","Lea").
partOf("Yard","Chain").
partOf("Yard","Sailing_vessel").
partOf("Yard","Perch").
partOf("Yard","Fathom").
partOf("Inch","Foot").
partOf("Mile","League").
partOf("Human","An_ecology").
partOf("Human","Human_society").
partOf("Human","Nature").
partOf("Human","The_natural_world").
partOf("Side","Torso").
partOf("Helm","Ship").
partOf("Rudder","Vessel").
partOf("Rudder","Ship").
partOf("Rudder","An_airplane").
partOf("Rudder","Sailboat").
partOf("Nail","Integumentary_system").
partOf("Nail","Digit").
partOf("Nail","House").
partOf("Nail","Toe").
partOf("Nail","Finger").
partOf("Nail","Tool").
partOf("Point","Needle").
partOf("Point","Pin").
partOf("Mind","Person").
partOf("Pin","Cylinder_lock").
partOf("Heading","Table").
partOf("Obverse","Coin").
partOf("Screw","Ship").
partOf("Screw","Outboard_motor").
partOf("Screw","Inclined_plane").
partOf("Ram","Computer").
partOf("Bolt","Nut_and_bolt").
partOf("Bolt","Rifle").
partOf("Bolt","Lock").
partOf("Hammer","Gunlock").
partOf("Hammer","Piano_action").
partOf("Human_body","Person").
partOf("Human_body","Homo").
partOf("X_bar","Xp").
partOf("Brain","Your_body").
partOf("Brain","Human_body").
partOf("Brain","Person").
partOf("Brain","Central_nervous_system").
partOf("Brain","Head").
partOf("Brain","Nervous_system").
partOf("Class","Society").
partOf("Headlight","Electrical_system").
partOf("Electrical_system","Motor_vehicle").
partOf("Hoof","Ungulate").
partOf("Boot","Car").
partOf("Automobile_horn","Car").
partOf("Bull","Cattle").
partOf("Goat","Tribe").
partOf("Pommel","Pommel_horse").
partOf("Pommel","Saddle").
partOf("Pommel","Haft").
partOf("Pommel","Hilt").
partOf("Stallion","Horse").
partOf("Grand_piano","The_chicago_symphony_orchestra").
partOf("License_plate","Radio").
partOf("License_plate","Radio").
partOf("Glass","Window").
partOf("Glass","Eyeglasses").
partOf("Bell","Funnel").
partOf("Bell","Wind_instrument").
partOf("Bell","Blunderbuss").
partOf("Bell","Trumpet").
partOf("Bell","Bell_tower").
partOf("Bell","Carillon").
partOf("Neck","Garment").
partOf("Neck","Cello").
partOf("Neck","Guitar").
partOf("Part","Unit").
partOf("Part","Whole").
partOf("Part","Hair").
partOf("Part","Meronymy").
partOf("Cello","String_quartet").
partOf("Nape","Neck").
partOf("Front","Cello").
partOf("Organism","An_ecosystem").
partOf("Plant","An_ecosystem").
partOf("Plant","The_garden").
partOf("Plant","Nature").
partOf("Nature","Personality").
partOf("Belly","Vertebrate").
partOf("Belly","Torso").
partOf("Saddle","Back").
partOf("Saddle","Domestic_fowl").
partOf("Saddle","Shoe").
partOf("Saddle","Cello").
partOf("Back","Car_seat").
partOf("Back","Chair").
partOf("Back","Torso").
partOf("Back","My_body").
partOf("Back","Cello").
partOf("Shoe","Person_s_clothing").
partOf("Shoe","Someone_s_clothing").
partOf("Bicycle_seat","Bicycle").
partOf("Screen","Screen_door").
partOf("Screen","Monitor").
partOf("Screen","Cathode_ray_tube").
partOf("Screen","Tv").
partOf("Monitor","The_computer").
partOf("Monitor","Computer_system").
partOf("Monitor","Television").
partOf("Monitor","Computer").
partOf("Sheep","Flock").
partOf("Stern","Ship").
partOf("Stern","Sailboat").
partOf("Buttocks","Torso").
partOf("Buttocks","Normal_human_body").
partOf("Sailboat","Regatta").
partOf("Tail","Vertebrate").
partOf("Tail","Coin").
partOf("Tail","Fuselage").
partOf("Reverse","Coin").
partOf("Reverse","Car").
partOf("Dock","Tail").
partOf("Dock","Seaport").
partOf("Fuselage","Airplane").
partOf("Trunk","An_automobile").
partOf("Trunk","Tree").
partOf("Trunk","Car").
partOf("Trunk","Auto").
partOf("Trunk","Automobile").
partOf("Gearing","Engine").
partOf("String","Cello").
partOf("Steering_wheel","Steering_system").
partOf("Steering_wheel","Car").
partOf("Steering_wheel","Auto").
partOf("Rim","Wheel").
partOf("Panel","Monitor").
partOf("Windowpane","Window").
partOf("Wall","House").
partOf("Wall","Building").


    '''
    
    objects_relationships_rules=r'''

    

    %define boxes that contain objects
    objectBox(B,L):-label(_,_,B,L),object(L).

    %define boxes that contain parts
    partBox(B,L):-label(_,_,B,L),part(L).


   
    
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




    m = Net()
    nnMapping = {'label': m}

    termPath = 'img ./'+img_path

    # set the classes that we consider
    domain =["Aeroplane", "Animal_Wing", "Arm", "Artifact_Wing", "Beak", "Bicycle", "Bird", "Boat", "Body", "Bodywork", "Bottle", "Bus", "Cap", "Car", "Cat", "Chain_wheel", "Chair", "Coach", "Cow", "Dog", "Door", "Ear", "Ebrow", "Engine", "Eye", "Foot", "Hair", "Hand", "Handlebar", "Head", "Headlight", "Hoof", "Horn", "Horse", "Leg", "License_plate", "Locomotive", "Mirror", "Motorbike", "Mouth", "Muzzle", "Neck", "Nose", "Person", "Plant", "Pot", "Pottedplant", "Saddle", "Screen", "Sheep", "Sofa", "Stern", "Tail", "Torso", "Train", "Tvmonitor", "Wheel", "Window", "Diningtable","Other"]
    

    aspProgram+=spatial_rules+objects_relationships_rules
 

    print(aspProgram)


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
    opath=output+'/conceptnet1'
    if path.exists(opath)==False:
        os.mkdir(opath) 
    with open(opath+"/CN1_inferences_JSON.json", "w") as outfile:
        outfile.write(json_object)
    outfile.close()
