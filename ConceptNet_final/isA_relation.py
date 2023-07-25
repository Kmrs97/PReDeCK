import requests
import json
from py2neo import  Graph
import copy

##Converts the information about IsA , to ASP atoms

def generate_rules_from_graph(tags,synonyms,ctgr=False):
    #Complete the credentials
    USERNAME = "" 
    PASSWORD = "" 
    sample_tags=list(set(tags+synonyms))
    tags_upper = [tag.upper() for tag in sample_tags]
    subjects=[]
    objects=[]
    categories=[]
    graph = Graph(scheme="bolt", host="localhost", port=YOURPORT, auth=(USERNAME, PASSWORD))

    #IsA relation
    resp=graph.run("MATCH (n)-[r:ISA]->(m) RETURN n,m").data()
    for pair in resp:
     if (str(pair['n']['label']).replace("_"," ") in tags_upper) or (str(pair['m']['label']).replace("_"," ") in tags_upper) or (str(pair['n']['label']) in tags_upper) or (str(pair['m']['label']) in tags_upper):
        subjects.append(pair['n']['label'])
        objects.append(pair['m']['label'])
        if ctgr:
            categories.append(pair['n']['category'])
    
    matched_labels=list(set(subjects+objects))
    hop1_labels = [i for i in matched_labels if i not in tags_upper]

    # IsA relation for super/subclasses
    for pair in resp:
     if (str(pair['n']['label']) in hop1_labels) or (str(pair['m']['label']) in hop1_labels):
        subjects.append(pair['n']['label'])
        objects.append(pair['m']['label'])
        if ctgr:
            categories.append(pair['n']['category'])



    subjects=[x.capitalize() for x in subjects]
    objects=[x.capitalize() for x in objects]
    if ctgr:
        categories=[x.capitalize() for x in categories]

    isA=[]
    for i in range(len(subjects)):
        if subjects[i]==objects[i]:
            continue
        if ctgr:
            rule='isA("'+subjects[i]+'","'+objects[i]+'","'+categories[i]+'").'
        else:
            rule='isA("'+subjects[i]+'","'+objects[i]+'").'
        isA.append(rule)

    set_obj=set(objects)
    objects=list(set_obj)
    return objects,isA

