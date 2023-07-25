import sys
sys.path.append('../../')
from ConceptNet import partOf_relation as pr
from ConceptNet import isA_relation as isa
from ConceptNet import synonym_rule as sr


print("Creating file containing Rules exported from ConceptNet...")
tags =["wing","sprocket","aeroplane", "animal wing", "arm", "artifact wing", "beak", "bicycle","bird","boat", "body", "bodywork", "bottle", "bus","cap", "car", "cat", "chain wheel","chair", "coach", "cow", "dog", "door", "ear", "eyebrow", "engine", "eye", "foot", "hair", "hand", "handlebar", "head", "headlight", "hoof", "horn", "horse", "leg", "license plate", "locomotive", "mirror", "motorbike", "mouth", "muzzle", "neck", "nose", "person", "plant", "pot", "potted plant", "saddle", "screen", "sheep", "sofa", "stern", "tail", "torso", "train", "tv monitor", "wheel", "window", "dining table"]
superclasses,synonyms=[],[]

with open("CNrules.txt","w") as file:
    synonyms,isSyn=sr.generate_rules_from_graph(tags,True)
    for syn in isSyn:
        file.write(syn+'\n')
    superclasses,isA=isa.generate_rules_from_graph(tags,synonyms,True)
    for sc in isA:
        file.write(sc+'\n')

    partOf,hasA,hasContext=pr.generate_rules_from_graph(tags,superclasses,synonyms,True)
    for p in partOf:
        file.write(p+'\n')

    for h in hasA:
        file.write(h+'\n')
    
    for c in hasContext:
        file.write(c+'\n')
        
print("File is ready!")
