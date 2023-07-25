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
    nn(label(1,I,B),["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "Other"]):- box(I,B,X1,Y1,X2,Y2,C).

    '''

    aspProgram = r'''
    isSynonymWith("wing","fender").
    isSynonymWith("wing","fly").
    isSynonymWith("wing","annex").
    isSynonymWith("wing","flank").
    isSynonymWith("wing","forward").
    isSynonymWith("airplane","aeroplane").
    isSynonymWith("fender","wing").
    isSynonymWith("bird","shuttlecock").
    isSynonymWith("bird","dame").
    isSynonymWith("bird","boo").
    isSynonymWith("bird","girl").
    isSynonymWith("bird","chap").
    isSynonymWith("bird","fowl").
    isSynonymWith("bird","time").
    isSynonymWith("bird","broad").
    isSynonymWith("bird","chick").
    isSynonymWith("bird","guy").
    isSynonymWith("bird","lass").
    isSynonymWith("bird","porridge").
    isSynonymWith("flank","wing").
    isSynonymWith("limb","arm").
    isSynonymWith("sprocket","cog").
    isSynonymWith("sprocket","widget").
    isSynonymWith("wheel","bicycle").
    isSynonymWith("wheel","steering_wheel").
    isSynonymWith("wheel","rack").
    isSynonymWith("wheel","roulette_wheel").
    isSynonymWith("wheel","rim").
    isSynonymWith("wheel","breaking_wheel").
    isSynonymWith("bicycle","push_bike").
    isSynonymWith("bicycle","pushbike").
    isSynonymWith("bicycle","bike").
    isSynonymWith("bicycle","two_wheeler").
    isSynonymWith("bicycle","cycle").
    isSynonymWith("bicycle","pedal_cycle").
    isSynonymWith("bicycle","velocipede").
    isSynonymWith("bike","bicycle").
    isSynonymWith("aeroplane","airplane").
    isSynonymWith("arm","branch").
    isSynonymWith("arm","sleeve").
    isSynonymWith("arm","weapon").
    isSynonymWith("arm","beweapon").
    isSynonymWith("body","torso").
    isSynonymWith("body","soundbox").
    isSynonymWith("body","consistency").
    isSynonymWith("body","leotard").
    isSynonymWith("leg","branch").
    isSynonymWith("leg","stage").
    isSynonymWith("leg","peg").
    isSynonymWith("leg","cathetus").
    isSynonymWith("leg","limb").
    isSynonymWith("leg","prop").
    isSynonymWith("branch","arm").
    isSynonymWith("hand","handwriting").
    isSynonymWith("hand","hired_hand").
    isSynonymWith("hand","pass").
    isSynonymWith("hand","bridge_player").
    isSynonymWith("hand","manus").
    isSynonymWith("hand","on_hand").
    isSynonymWith("hand","at_hand").
    isSynonymWith("chair","president").
    isSynonymWith("chair","professorship").
    isSynonymWith("chair","moderate").
    isSynonymWith("chair","electric_chair").
    isSynonymWith("chair","stool").
    isSynonymWith("beak","peck").
    isSynonymWith("beak","honker").
    isSynonymWith("beak","bill").
    isSynonymWith("mouth","sass").
    isSynonymWith("mouth","mouthpiece").
    isSynonymWith("mouth","talk").
    isSynonymWith("nose","nuzzle").
    isSynonymWith("nose","nozzle").
    isSynonymWith("nose","scent").
    isSynonymWith("nose","pry").
    isSynonymWith("honker","beak").
    isSynonymWith("bill","beak").
    isSynonymWith("motorbike","minibike").
    isSynonymWith("motorbike","motorcycle").
    isSynonymWith("cycle","bicycle").
    isSynonymWith("fowl","bird").
    isSynonymWith("boat","gravy_boat").
    isSynonymWith("boat","watercraft").
    isSynonymWith("boat","ship").
    isSynonymWith("boat","craft").
    isSynonymWith("watercraft","boat").
    isSynonymWith("ship","boat").
    isSynonymWith("torso","trunk").
    isSynonymWith("bottle","nursing_bottle").
    isSynonymWith("bottle","feeding_bottle").
    isSynonymWith("bottle","baby_s_bottle").
    isSynonymWith("bottle","flask").
    isSynonymWith("bottle","balls").
    isSynonymWith("feeding_bottle","bottle").
    isSynonymWith("baby_s_bottle","bottle").
    isSynonymWith("flask","bottle").
    isSynonymWith("bus","busbar").
    isSynonymWith("bus","bus_topology").
    isSynonymWith("bus","autobus").
    isSynonymWith("bus","electrical_bus").
    isSynonymWith("car","cable_car").
    isSynonymWith("car","rail_car").
    isSynonymWith("car","coach").
    isSynonymWith("car","carriage").
    isSynonymWith("car","car_nicobarese").
    isSynonymWith("car","railroad_car").
    isSynonymWith("car","automobile").
    isSynonymWith("car","auto").
    isSynonymWith("car","railcar").
    isSynonymWith("car","carload").
    isSynonymWith("car","gondola").
    isSynonymWith("car","motor_car").
    isSynonymWith("autobus","bus").
    isSynonymWith("cap","capital").
    isSynonymWith("cap","hood").
    isSynonymWith("cap","crownwork").
    isSynonymWith("cap","detonator").
    isSynonymWith("cap","ceiling").
    isSynonymWith("cap","lid").
    isSynonymWith("hood","head").
    isSynonymWith("top","head").
    isSynonymWith("top","cap").
    isSynonymWith("lid","cap").
    isSynonymWith("machine","car").
    isSynonymWith("machine","engine").
    isSynonymWith("coach","passenger_car").
    isSynonymWith("coach","bus").
    isSynonymWith("coach","carriage").
    isSynonymWith("railroad_car","car").
    isSynonymWith("automobile","car").
    isSynonymWith("auto","car").
    isSynonymWith("railcar","car").
    isSynonymWith("train","coach").
    isSynonymWith("train","aim").
    isSynonymWith("train","gearing").
    isSynonymWith("train","string").
    isSynonymWith("train","caravan").
    isSynonymWith("train","trail").
    isSynonymWith("train","discipline").
    isSynonymWith("train","educate").
    isSynonymWith("train","prepare").
    isSynonymWith("horn","cornet").
    isSynonymWith("horn","french_horn").
    isSynonymWith("horn","automobile_horn").
    isSynonymWith("horn","horn_peninsula").
    isSynonymWith("horn","horn_region").
    isSynonymWith("horn","horn_of_africa").
    isSynonymWith("gondola","car").
    isSynonymWith("motor_car","car").
    isSynonymWith("cat","kat").
    isSynonymWith("cat","vomit").
    isSynonymWith("cat","computerized_tomography").
    isSynonymWith("cat","cat_o__nine_tails").
    isSynonymWith("cat","guy").
    isSynonymWith("cat","big_cat").
    isSynonymWith("cat","saber_toothed_cat").
    isSynonymWith("cat","pantherine_cat").
    isSynonymWith("cat","feliform").
    isSynonymWith("cat","panther").
    isSynonymWith("cat","feline_cat").
    isSynonymWith("cat","ct").
    isSynonymWith("dog","domestic_dog").
    isSynonymWith("dog","andiron").
    isSynonymWith("dog","frump").
    isSynonymWith("dog","frank").
    isSynonymWith("dog","bloke").
    isSynonymWith("dog","chase").
    isSynonymWith("dog","pawl").
    isSynonymWith("dog","cad").
    isSynonymWith("dog","fellow").
    isSynonymWith("dog","canine").
    isSynonymWith("dog","canis_familiaris").
    isSynonymWith("dog","hound").
    isSynonymWith("dog","cur").
    isSynonymWith("dog","guy").
    isSynonymWith("dog","broon").
    isSynonymWith("dog","click").
    isSynonymWith("dog","stud").
    isSynonymWith("dog","soldier").
    isSynonymWith("dog","pooch").
    isSynonymWith("dog","chap").
    isSynonymWith("hair","haircloth").
    isSynonymWith("hair","hair_s_breadth").
    isSynonymWith("ct","cat").
    isSynonymWith("trainer","coach").
    isSynonymWith("wagon","car").
    isSynonymWith("manager","coach").
    isSynonymWith("cow","overawe").
    isSynonymWith("cow","abash").
    isSynonymWith("cow","daunt").
    isSynonymWith("cow","bovine").
    isSynonymWith("cow","frighten").
    isSynonymWith("cow","intimidate").
    isSynonymWith("cow","heifer").
    isSynonymWith("cow","bastard").
    isSynonymWith("cow","discourage").
    isSynonymWith("cow","bitch").
    isSynonymWith("cow","dishearten").
    isSynonymWith("overawe","cow").
    isSynonymWith("canine","dog").
    isSynonymWith("domestic_dog","dog").
    isSynonymWith("canis_familiaris","dog").
    isSynonymWith("broon","dog").
    isSynonymWith("pooch","dog").
    isSynonymWith("door","doorway").
    isSynonymWith("door","car_door").
    isSynonymWith("window","windowpane").
    isSynonymWith("handle","hand").
    isSynonymWith("ear","auricle").
    isSynonymWith("ear","head").
    isSynonymWith("ear","attention").
    isSynonymWith("ear","hearing").
    isSynonymWith("ear","spike").
    isSynonymWith("head","drumhead").
    isSynonymWith("head","forefront").
    isSynonymWith("head","mind").
    isSynonymWith("head","oral_sex").
    isSynonymWith("head","heading").
    isSynonymWith("head","principal").
    isSynonymWith("head","promontory").
    isSynonymWith("head","pass").
    isSynonymWith("head","fountainhead").
    isSynonymWith("head","lead").
    isSynonymWith("head","read_write_head").
    isSynonymWith("head","steer").
    isSynonymWith("head","question").
    isSynonymWith("head","point").
    isSynonymWith("head","headway").
    isSynonymWith("head","capitulum").
    isSynonymWith("head","blowjob").
    isSynonymWith("head","summit").
    isSynonymWith("head","origin").
    isSynonymWith("head","beginning").
    isSynonymWith("head","crown").
    isSynonymWith("head","headland").
    isSynonymWith("head","source").
    isSynonymWith("head","understanding").
    isSynonymWith("head","director").
    isSynonymWith("head","topic").
    isSynonymWith("head","thought").
    isSynonymWith("head","chief").
    isSynonymWith("head","headline").
    isSynonymWith("head","brain").
    isSynonymWith("head","top").
    isSynonymWith("head","rise").
    isSynonymWith("head","category").
    isSynonymWith("head","headmaster").
    isSynonymWith("head","commencement").
    isSynonymWith("head","branch").
    isSynonymWith("head","commander").
    isSynonymWith("head","chieftain").
    isSynonymWith("head","composure").
    isSynonymWith("head","boss").
    isSynonymWith("head","class").
    isSynonymWith("head","first").
    isSynonymWith("head","headpiece").
    isSynonymWith("head","subject").
    isSynonymWith("head","acme").
    isSynonymWith("head","section").
    isSynonymWith("head","intellect").
    isSynonymWith("head","caput").
    isSynonymWith("head","superintendent").
    isSynonymWith("spike","ear").
    isSynonymWith("eye","center").
    isSynonymWith("eye","eyelet").
    isSynonymWith("eye","perceptiveness").
    isSynonymWith("engine","motor").
    isSynonymWith("engine","locomotive").
    isSynonymWith("motor","engine").
    isSynonymWith("locomotive","loco").
    isSynonymWith("foot","infantry").
    isSynonymWith("foot","foundation").
    isSynonymWith("foot","animal_foot").
    isSynonymWith("foot","metrical_foot").
    isSynonymWith("horse","sawhorse").
    isSynonymWith("horse","cavalry").
    isSynonymWith("horse","knight").
    isSynonymWith("horse","h").
    isSynonymWith("horse","horsie").
    isSynonymWith("horse","stallion").
    isSynonymWith("horse","gelding").
    isSynonymWith("horse","pommel_horse").
    isSynonymWith("horse","equine").
    isSynonymWith("horse","dobbin").
    isSynonymWith("pointer","hand").
    isSynonymWith("manus","hand").
    isSynonymWith("handlebar","wheel").
    isSynonymWith("handlebar","handlebars").
    isSynonymWith("handlebar","helm").
    isSynonymWith("handlebar","rudder").
    isSynonymWith("handlebars","wheel").
    isSynonymWith("foam","head").
    isSynonymWith("mind","head").
    isSynonymWith("toilet","head").
    isSynonymWith("lead","head").
    isSynonymWith("capitulum","ear").
    isSynonymWith("individual","person").
    isSynonymWith("beginning","head").
    isSynonymWith("chief","head").
    isSynonymWith("caput","head").
    isSynonymWith("headlight","headlamp").
    isSynonymWith("headlamp","headlight").
    isSynonymWith("hoof","foot").
    isSynonymWith("hoof","boot").
    isSynonymWith("boot","hoof").
    isSynonymWith("knight","horse").
    isSynonymWith("cathetus","leg").
    isSynonymWith("license_plate","number_plate").
    isSynonymWith("license_plate","number_plate").
    isSynonymWith("license_plate","number_plate").
    isSynonymWith("number_plate","license_plate").
    isSynonymWith("number_plate","license_plate").
    isSynonymWith("number_plate","license_plate").
    isSynonymWith("number_plate","license_plate").
    isSynonymWith("number_plate","license_plate").
    isSynonymWith("mirror","glass").
    isSynonymWith("bottle","baby_s_bottle").
    isSynonymWith("bottle","baby_s_bottle").
    isSynonymWith("jar","pot").
    isSynonymWith("muzzle","gag").
    isSynonymWith("muzzle","gun_muzzle").
    isSynonymWith("neck","french_kiss").
    isSynonymWith("neck","nape").
    isSynonymWith("nape","neck").
    isSynonymWith("nozzle","beak").
    isSynonymWith("plant","establish").
    isSynonymWith("plant","implant").
    isSynonymWith("plant","grow").
    isSynonymWith("implant","plant").
    isSynonymWith("pot","toilet").
    isSynonymWith("pot","batch").
    isSynonymWith("pot","potentiometer").
    isSynonymWith("pot","middy").
    isSynonymWith("pot","cookpot").
    isSynonymWith("pot","winning_hazard").
    isSynonymWith("pot","kitty").
    isSynonymWith("pot","can").
    isSynonymWith("winning_hazard","pot").
    isSynonymWith("kitty","pot").
    isSynonymWith("saddle","saddleback").
    isSynonymWith("saddle","charge").
    isSynonymWith("saddle","bicycle_seat").
    isSynonymWith("screen","sieve").
    isSynonymWith("screen","riddle").
    isSynonymWith("screen","screenland").
    isSynonymWith("screen","blind").
    isSynonymWith("screen","shield").
    isSynonymWith("sieve","screen").
    isSynonymWith("sofa","couch").
    isSynonymWith("sofa","divan").
    isSynonymWith("sofa","settee").
    isSynonymWith("couch","sofa").
    isSynonymWith("divan","sofa").
    isSynonymWith("stern","buttocks").
    isSynonymWith("stern","austere").
    isSynonymWith("stern","unappeasable").
    isSynonymWith("stern","strict").
    isSynonymWith("stern","isaac_stern").
    isSynonymWith("stern","poop").
    isSynonymWith("austere","stern").
    isSynonymWith("poop","stern").
    isSynonymWith("tail","buttocks").
    isSynonymWith("tail","dock").
    isSynonymWith("tail","chase").
    isSynonymWith("tail","stern").
    isSynonymWith("tail","fag_end").
    isSynonymWith("tail","ass").
    isSynonymWith("trunk","torso").
    isSynonymWith("educate","train").
    isSynonymWith("prepare","train").
    isSynonymWith("tv_monitor","television_monitor").
    isSynonymWith("rack","neck").
    isSynonymWith("wing","fender").
    isSynonymWith("wing","fly").
    isSynonymWith("wing","annex").
    isSynonymWith("wing","flank").
    isSynonymWith("wing","forward").
    isSynonymWith("airplane","aeroplane").
    isSynonymWith("fender","wing").
    isSynonymWith("bird","shuttlecock").
    isSynonymWith("bird","dame").
    isSynonymWith("bird","boo").
    isSynonymWith("bird","girl").
    isSynonymWith("bird","chap").
    isSynonymWith("bird","fowl").
    isSynonymWith("bird","time").
    isSynonymWith("bird","broad").
    isSynonymWith("bird","chick").
    isSynonymWith("bird","guy").
    isSynonymWith("bird","lass").
    isSynonymWith("bird","porridge").
    isSynonymWith("flank","wing").
    isSynonymWith("limb","arm").
    isSynonymWith("sprocket","cog").
    isSynonymWith("sprocket","widget").
    isSynonymWith("wheel","steering_wheel").
    isSynonymWith("wheel","rack").
    isSynonymWith("wheel","roulette_wheel").
    isSynonymWith("wheel","rim").
    isSynonymWith("wheel","breaking_wheel").
    isSynonymWith("bicycle","push_bike").
    isSynonymWith("bicycle","pushbike").
    isSynonymWith("bicycle","bike").
    isSynonymWith("bicycle","two_wheeler").
    isSynonymWith("bicycle","cycle").
    isSynonymWith("bicycle","pedal_cycle").
    isSynonymWith("bicycle","velocipede").
    isSynonymWith("bike","bicycle").
    isSynonymWith("aeroplane","airplane").
    isSynonymWith("arm","branch").
    isSynonymWith("arm","sleeve").
    isSynonymWith("arm","weapon").
    isSynonymWith("arm","beweapon").
    isSynonymWith("body","soundbox").
    isSynonymWith("body","consistency").
    isSynonymWith("body","leotard").
    isSynonymWith("leg","branch").
    isSynonymWith("leg","stage").
    isSynonymWith("leg","peg").
    isSynonymWith("leg","cathetus").
    isSynonymWith("leg","limb").
    isSynonymWith("leg","prop").
    isSynonymWith("branch","arm").
    isSynonymWith("hand","handwriting").
    isSynonymWith("hand","hired_hand").
    isSynonymWith("hand","pass").
    isSynonymWith("hand","bridge_player").
    isSynonymWith("hand","manus").
    isSynonymWith("hand","on_hand").
    isSynonymWith("hand","at_hand").
    isSynonymWith("chair","president").
    isSynonymWith("chair","professorship").
    isSynonymWith("chair","moderate").
    isSynonymWith("chair","electric_chair").
    isSynonymWith("chair","stool").
    isSynonymWith("beak","peck").
    isSynonymWith("beak","honker").
    isSynonymWith("beak","bill").
    isSynonymWith("mouth","sass").
    isSynonymWith("mouth","mouthpiece").
    isSynonymWith("mouth","talk").
    isSynonymWith("nose","nuzzle").
    isSynonymWith("nose","nozzle").
    isSynonymWith("nose","scent").
    isSynonymWith("nose","pry").
    isSynonymWith("honker","beak").
    isSynonymWith("bill","beak").
    isSynonymWith("motorbike","minibike").
    isSynonymWith("motorbike","motorcycle").
    isSynonymWith("cycle","bicycle").
    isSynonymWith("fowl","bird").
    isSynonymWith("boat","gravy_boat").
    isSynonymWith("boat","watercraft").
    isSynonymWith("boat","ship").
    isSynonymWith("boat","craft").
    isSynonymWith("watercraft","boat").
    isSynonymWith("ship","boat").
    isSynonymWith("torso","trunk").
    isSynonymWith("bottle","nursing_bottle").
    isSynonymWith("bottle","feeding_bottle").
    isSynonymWith("bottle","baby_s_bottle").
    isSynonymWith("bottle","flask").
    isSynonymWith("bottle","balls").
    isSynonymWith("feeding_bottle","bottle").
    isSynonymWith("baby_s_bottle","bottle").
    isSynonymWith("flask","bottle").
    isSynonymWith("bus","busbar").
    isSynonymWith("bus","bus_topology").
    isSynonymWith("bus","autobus").
    isSynonymWith("bus","electrical_bus").
    isSynonymWith("car","cable_car").
    isSynonymWith("car","rail_car").
    isSynonymWith("car","carriage").
    isSynonymWith("car","car_nicobarese").
    isSynonymWith("car","railroad_car").
    isSynonymWith("car","automobile").
    isSynonymWith("car","auto").
    isSynonymWith("car","railcar").
    isSynonymWith("car","carload").
    isSynonymWith("car","gondola").
    isSynonymWith("car","motor_car").
    isSynonymWith("autobus","bus").
    isSynonymWith("cap","capital").
    isSynonymWith("cap","hood").
    isSynonymWith("cap","crownwork").
    isSynonymWith("cap","detonator").
    isSynonymWith("cap","ceiling").
    isSynonymWith("cap","lid").
    isSynonymWith("hood","head").
    isSynonymWith("top","head").
    isSynonymWith("top","cap").
    isSynonymWith("lid","cap").
    isSynonymWith("machine","car").
    isSynonymWith("machine","engine").
    isSynonymWith("coach","passenger_car").
    isSynonymWith("coach","carriage").
    isSynonymWith("railroad_car","car").
    isSynonymWith("automobile","car").
    isSynonymWith("auto","car").
    isSynonymWith("railcar","car").
    isSynonymWith("train","aim").
    isSynonymWith("train","gearing").
    isSynonymWith("train","string").
    isSynonymWith("train","caravan").
    isSynonymWith("train","trail").
    isSynonymWith("train","discipline").
    isSynonymWith("train","educate").
    isSynonymWith("train","prepare").
    isSynonymWith("horn","cornet").
    isSynonymWith("horn","french_horn").
    isSynonymWith("horn","automobile_horn").
    isSynonymWith("horn","horn_peninsula").
    isSynonymWith("horn","horn_region").
    isSynonymWith("horn","horn_of_africa").
    isSynonymWith("gondola","car").
    isSynonymWith("motor_car","car").
    isSynonymWith("cat","kat").
    isSynonymWith("cat","vomit").
    isSynonymWith("cat","computerized_tomography").
    isSynonymWith("cat","cat_o__nine_tails").
    isSynonymWith("cat","guy").
    isSynonymWith("cat","big_cat").
    isSynonymWith("cat","saber_toothed_cat").
    isSynonymWith("cat","pantherine_cat").
    isSynonymWith("cat","feliform").
    isSynonymWith("cat","panther").
    isSynonymWith("cat","feline_cat").
    isSynonymWith("cat","ct").
    isSynonymWith("dog","domestic_dog").
    isSynonymWith("dog","andiron").
    isSynonymWith("dog","frump").
    isSynonymWith("dog","frank").
    isSynonymWith("dog","bloke").
    isSynonymWith("dog","chase").
    isSynonymWith("dog","pawl").
    isSynonymWith("dog","cad").
    isSynonymWith("dog","fellow").
    isSynonymWith("dog","canine").
    isSynonymWith("dog","canis_familiaris").
    isSynonymWith("dog","hound").
    isSynonymWith("dog","cur").
    isSynonymWith("dog","guy").
    isSynonymWith("dog","broon").
    isSynonymWith("dog","click").
    isSynonymWith("dog","stud").
    isSynonymWith("dog","soldier").
    isSynonymWith("dog","pooch").
    isSynonymWith("dog","chap").
    isSynonymWith("hair","haircloth").
    isSynonymWith("hair","hair_s_breadth").
    isSynonymWith("ct","cat").
    isSynonymWith("trainer","coach").
    isSynonymWith("wagon","car").
    isSynonymWith("manager","coach").
    isSynonymWith("cow","overawe").
    isSynonymWith("cow","abash").
    isSynonymWith("cow","daunt").
    isSynonymWith("cow","bovine").
    isSynonymWith("cow","frighten").
    isSynonymWith("cow","intimidate").
    isSynonymWith("cow","heifer").
    isSynonymWith("cow","bastard").
    isSynonymWith("cow","discourage").
    isSynonymWith("cow","bitch").
    isSynonymWith("cow","dishearten").
    isSynonymWith("overawe","cow").
    isSynonymWith("canine","dog").
    isSynonymWith("domestic_dog","dog").
    isSynonymWith("canis_familiaris","dog").
    isSynonymWith("broon","dog").
    isSynonymWith("pooch","dog").
    isSynonymWith("door","doorway").
    isSynonymWith("door","car_door").
    isSynonymWith("window","windowpane").
    isSynonymWith("handle","hand").
    isSynonymWith("ear","auricle").
    isSynonymWith("ear","attention").
    isSynonymWith("ear","hearing").
    isSynonymWith("ear","spike").
    isSynonymWith("head","drumhead").
    isSynonymWith("head","forefront").
    isSynonymWith("head","mind").
    isSynonymWith("head","oral_sex").
    isSynonymWith("head","heading").
    isSynonymWith("head","principal").
    isSynonymWith("head","promontory").
    isSynonymWith("head","pass").
    isSynonymWith("head","fountainhead").
    isSynonymWith("head","lead").
    isSynonymWith("head","read_write_head").
    isSynonymWith("head","steer").
    isSynonymWith("head","question").
    isSynonymWith("head","point").
    isSynonymWith("head","headway").
    isSynonymWith("head","capitulum").
    isSynonymWith("head","blowjob").
    isSynonymWith("head","summit").
    isSynonymWith("head","origin").
    isSynonymWith("head","beginning").
    isSynonymWith("head","crown").
    isSynonymWith("head","headland").
    isSynonymWith("head","source").
    isSynonymWith("head","understanding").
    isSynonymWith("head","director").
    isSynonymWith("head","topic").
    isSynonymWith("head","thought").
    isSynonymWith("head","chief").
    isSynonymWith("head","headline").
    isSynonymWith("head","brain").
    isSynonymWith("head","top").
    isSynonymWith("head","rise").
    isSynonymWith("head","category").
    isSynonymWith("head","headmaster").
    isSynonymWith("head","commencement").
    isSynonymWith("head","branch").
    isSynonymWith("head","commander").
    isSynonymWith("head","chieftain").
    isSynonymWith("head","composure").
    isSynonymWith("head","boss").
    isSynonymWith("head","class").
    isSynonymWith("head","first").
    isSynonymWith("head","headpiece").
    isSynonymWith("head","subject").
    isSynonymWith("head","acme").
    isSynonymWith("head","section").
    isSynonymWith("head","intellect").
    isSynonymWith("head","caput").
    isSynonymWith("head","superintendent").
    isSynonymWith("spike","ear").
    isSynonymWith("eye","center").
    isSynonymWith("eye","eyelet").
    isSynonymWith("eye","perceptiveness").
    isSynonymWith("engine","motor").
    isSynonymWith("motor","engine").
    isSynonymWith("locomotive","loco").
    isSynonymWith("foot","infantry").
    isSynonymWith("foot","foundation").
    isSynonymWith("foot","animal_foot").
    isSynonymWith("foot","metrical_foot").
    isSynonymWith("horse","sawhorse").
    isSynonymWith("horse","cavalry").
    isSynonymWith("horse","knight").
    isSynonymWith("horse","h").
    isSynonymWith("horse","horsie").
    isSynonymWith("horse","stallion").
    isSynonymWith("horse","gelding").
    isSynonymWith("horse","pommel_horse").
    isSynonymWith("horse","equine").
    isSynonymWith("horse","dobbin").
    isSynonymWith("pointer","hand").
    isSynonymWith("manus","hand").
    isSynonymWith("handlebar","handlebars").
    isSynonymWith("handlebar","helm").
    isSynonymWith("handlebar","rudder").
    isSynonymWith("handlebars","wheel").
    isSynonymWith("foam","head").
    isSynonymWith("mind","head").
    isSynonymWith("toilet","head").
    isSynonymWith("lead","head").
    isSynonymWith("capitulum","ear").
    isSynonymWith("individual","person").
    isSynonymWith("beginning","head").
    isSynonymWith("chief","head").
    isSynonymWith("caput","head").
    isSynonymWith("headlight","headlamp").
    isSynonymWith("headlamp","headlight").
    isSynonymWith("hoof","boot").
    isSynonymWith("boot","hoof").
    isSynonymWith("knight","horse").
    isSynonymWith("cathetus","leg").
    isSynonymWith("license_plate","number_plate").
    isSynonymWith("license_plate","number_plate").
    isSynonymWith("license_plate","number_plate").
    isSynonymWith("number_plate","license_plate").
    isSynonymWith("number_plate","license_plate").
    isSynonymWith("number_plate","license_plate").
    isSynonymWith("number_plate","license_plate").
    isSynonymWith("number_plate","license_plate").
    isSynonymWith("mirror","glass").
    isSynonymWith("bottle","baby_s_bottle").
    isSynonymWith("bottle","baby_s_bottle").
    isSynonymWith("jar","pot").
    isSynonymWith("muzzle","gag").
    isSynonymWith("muzzle","gun_muzzle").
    isSynonymWith("neck","french_kiss").
    isSynonymWith("neck","nape").
    isSynonymWith("nape","neck").
    isSynonymWith("nozzle","beak").
    isSynonymWith("plant","establish").
    isSynonymWith("plant","implant").
    isSynonymWith("plant","grow").
    isSynonymWith("implant","plant").
    isSynonymWith("pot","toilet").
    isSynonymWith("pot","batch").
    isSynonymWith("pot","potentiometer").
    isSynonymWith("pot","middy").
    isSynonymWith("pot","cookpot").
    isSynonymWith("pot","winning_hazard").
    isSynonymWith("pot","kitty").
    isSynonymWith("pot","can").
    isSynonymWith("winning_hazard","pot").
    isSynonymWith("kitty","pot").
    isSynonymWith("saddle","saddleback").
    isSynonymWith("saddle","charge").
    isSynonymWith("saddle","bicycle_seat").
    isSynonymWith("screen","sieve").
    isSynonymWith("screen","riddle").
    isSynonymWith("screen","screenland").
    isSynonymWith("screen","blind").
    isSynonymWith("screen","shield").
    isSynonymWith("sieve","screen").
    isSynonymWith("sofa","couch").
    isSynonymWith("sofa","divan").
    isSynonymWith("sofa","settee").
    isSynonymWith("couch","sofa").
    isSynonymWith("divan","sofa").
    isSynonymWith("stern","buttocks").
    isSynonymWith("stern","austere").
    isSynonymWith("stern","unappeasable").
    isSynonymWith("stern","strict").
    isSynonymWith("stern","isaac_stern").
    isSynonymWith("stern","poop").
    isSynonymWith("austere","stern").
    isSynonymWith("poop","stern").
    isSynonymWith("tail","buttocks").
    isSynonymWith("tail","dock").
    isSynonymWith("tail","chase").
    isSynonymWith("tail","fag_end").
    isSynonymWith("tail","ass").
    isSynonymWith("trunk","torso").
    isSynonymWith("educate","train").
    isSynonymWith("prepare","train").
    isSynonymWith("tv_monitor","television_monitor").
    isSynonymWith("rack","neck").


    '''

    label_association=r'''

    label(N,I,B,L2):-label(N,I,B,L1), isSynonymWith(L1,L2),pascal_class(L2),not pascal_class(L1).
    label(N,I,B,L2):-label(N,I,B,L1), isSynonymWith(L2,L1),pascal_class(L2),not pascal_class(L1).
    
    '''
    aspProgram+=label_association
    m = Net()
    nnMapping = {'label': m}

    termPath = 'img ./'+img_path
    
    # set the classes that we consider
    # domain =["Aeroplane", "Animal_Wing", "Arm", "Artifact_Wing", "Beak", "Bicycle", "Bird", "Boat", "Body", "Bodywork", "Bottle", "Bus", "Cap", "Car", "Cat", "Chain_wheel", "Chair", "Coach", "Cow", "Dog", "Door", "Ear", "Ebrow", "Engine", "Eye", "Foot", "Hair", "Hand", "Handlebar", "Head", "Headlight", "Hoof", "Horn", "Horse", "Leg", "License_plate", "Locomotive", "Mirror", "Motorbike", "Mouth", "Muzzle", "Neck", "Nose", "Person", "Plant", "Pot", "Pottedplant", "Saddle", "Screen", "Sheep", "Sofa", "Stern", "Tail", "Torso", "Train", "Tvmonitor", "Wheel", "Window", "Diningtable","Other"]
    domain=["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "Other"]
    image_ids,factsList, dataList =termPath2dataList(termPath, img_size, domain,model,"coco") ##coco OR pascal
    
    pascal_labels=["Aeroplane", "Animal_Wing", "Arm", "Artifact_Wing", "Beak", "Bicycle", "Bird", "Boat", "Body", "Bodywork", "Bottle", "Bus", "Cap", "Car", "Cat", "Chain_wheel", "Chair", "Coach", "Cow", "Dog", "Door", "Ear", "Ebrow", "Engine", "Eye", "Foot", "Hair", "Hand", "Handlebar", "Head", "Headlight", "Hoof", "Horn", "Horse", "Leg", "License_plate", "Locomotive", "Mirror", "Motorbike", "Mouth", "Muzzle", "Neck", "Nose", "Person", "Plant", "Pot", "Pottedplant", "Saddle", "Screen", "Sheep", "Sofa", "Stern", "Tail", "Torso", "Train", "Tvmonitor", "Wheel", "Window", "Diningtable","Other"]
    pascal_labels=[x.lower() for x in pascal_labels]

    for p in pascal_labels:
        rule='pascal_class("'+p+'").'
        aspProgram+=rule+"\n"
    
   

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
    opath=output+'/Coco_Detections'
    if path.exists(opath)==False:
        os.mkdir(opath) 
    with open(opath+"/CoCo_inferences_pascalAssociation_JSON.json", "w") as outfile:
        outfile.write(json_object)
    outfile.close()
