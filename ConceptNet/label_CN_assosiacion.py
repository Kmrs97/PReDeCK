import json
import requests

## The following program is querying ConceptNet, given a label to retrive all the required information
def check_context(label,category):
    
    tok=label.split(",")
    if(len(tok)>1):
        if tok[0]=="n":
            if tok[1].replace(" ","")==category:
                return True
    return False

def getAssociations_Subjects(label,category):
    label=label.replace(" ","_")
    responses={}
    request='http://api.conceptnet.io/query?start=/c/en/'+label+'&limit=1000'
    response = requests.get(request)
    obj = response.json()
    accepted_rels=["IsA","Synonym","HasA","PartOf","HasContext"]
    relations=[]
    subjects=[]
    
    for i in range(len(obj['edges'])):
        relation=obj['edges'][i]['rel']['label']
        if(relation in accepted_rels):      
            if(not obj['edges'][i]['end']['language']=='en'):
                continue
            if(not obj['edges'][i]["weight"]>=1):
                continue  
            
            if(obj['edges'][i]['end']['label'].upper()==category.upper()):
                relations.append(obj['edges'][i]['rel']['label'])
                if(obj['edges'][i]['end']['label'].startswith('a ')):
                    sub=obj['edges'][i]['end']['label'].replace("a ","")
                    subjects.append(sub)
                elif(obj['edges'][i]['end']['label'].startswith('A ')):
                    sub=obj['edges'][i]['end']['label'].replace("A ","")
                    subjects.append(sub)
                else:
                    subjects.append(obj['edges'][i]['end']['label'])
                continue

            if(relation=="HasA"):
                relations.append(obj['edges'][i]['rel']['label'])
                if(obj['edges'][i]['end']['label'].startswith('a ')):
                    sub=obj['edges'][i]['end']['label'].replace("a ","")
                    subjects.append(sub)
                elif(obj['edges'][i]['end']['label'].startswith('A ')):
                    sub=obj['edges'][i]['end']['label'].replace("A ","")
                    subjects.append(sub)    
                else:
                    subjects.append(obj['edges'][i]['end']['label'])
                continue


            if(not"sense_label" in obj['edges'][i]['end']):
                continue
            elif (not check_context(obj['edges'][i]['end']['sense_label'],category)):
                continue
            relations.append(obj['edges'][i]['rel']['label'])
            if(obj['edges'][i]['end']['label'].startswith('a ')):
                sub=obj['edges'][i]['end']['label'].replace("a ","")
                subjects.append(sub)
            elif(obj['edges'][i]['end']['label'].startswith('A ')):
                sub=obj['edges'][i]['end']['label'].replace("A ","")
                subjects.append(sub)
            else:
                subjects.append(obj['edges'][i]['end']['label'])      
    print(relations)
    print(subjects) 
    return relations,subjects  

def getAssociations_Objects(label,category):
    label=label.replace(" ","_")
    responses={}
    request='http://api.conceptnet.io/query?end=/c/en/'+label+'&limit=1000'
    response = requests.get(request)
    obj = response.json()
    accepted_rels=["HasContext","IsA","Synonym","HasA","PartOf"]
    relations=[]
    objects=[]
    
    for i in range(len(obj['edges'])):
        relation=obj['edges'][i]['rel']['label']
        if(relation in accepted_rels):  
            if(not obj['edges'][i]['start']['language']=='en'):
                continue
            if(not obj['edges'][i]["weight"]>=1):
                continue   

            if(obj['edges'][i]['start']['label'].upper()==category.upper()):
                relations.append(obj['edges'][i]['rel']['label'])
                if(obj['edges'][i]['start']['label'].startswith('a ')):
                    sub=obj['edges'][i]['start']['label'].replace("a ","")
                    objects.append(sub)
                elif(obj['edges'][i]['start']['label'].startswith('A ')):
                    sub=obj['edges'][i]['start']['label'].replace("A ","")
                    objects.append(sub)
                else:
                    objects.append(obj['edges'][i]['start']['label'])
                continue

            if(relation=="HasA"):
                relations.append(obj['edges'][i]['rel']['label'])
                if(obj['edges'][i]['start']['label'].startswith('a ')):
                    sub=obj['edges'][i]['start']['label'].replace("a ","")
                    objects.append(sub)
                elif(obj['edges'][i]['start']['label'].startswith('A ')):
                    sub=obj['edges'][i]['start']['label'].replace("A ","")
                    objects.append(sub)
                else:
                    objects.append(obj['edges'][i]['start']['label'])
                continue
               
            if(not"sense_label" in obj['edges'][i]['start']):
                continue
            elif (not check_context(obj['edges'][i]['start']['sense_label'],category)):
                continue
            relations.append(obj['edges'][i]['rel']['label'])
            if(obj['edges'][i]['start']['label'].startswith('a ')):
                sub=obj['edges'][i]['start']['label'].replace("a ","")
                objects.append(sub)
            elif(obj['edges'][i]['start']['label'].startswith('A ')):
                sub=obj['edges'][i]['start']['label'].replace("A ","")
                objects.append(sub)
            else:
                objects.append(obj['edges'][i]['start']['label'])   

    print(relations)
    print(objects) 
    return relations,objects  
         