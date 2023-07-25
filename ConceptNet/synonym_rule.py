import requests
import json
from py2neo import  Graph

##Converts the information about Synonyms , to ASP atoms

def generate_rules_from_graph(sample_tags,ctgr=False):
    #Complete the credentials
    USERNAME = "" 
    PASSWORD = "" 
    # sample_tags=["cat","car"]
    tags_upper = [tag.upper() for tag in sample_tags]
    # print(tags_upper)
    subjects=[]
    objects=[]
    categories=[]
    graph = Graph(scheme="bolt", host="localhost", port=YOURPORT, auth=(USERNAME, PASSWORD))

    resp=graph.run("MATCH (n)-[r:SYNONYM]->(m) RETURN n,m").data()
    for pair in resp:
     if (str(pair['n']['label']).replace("_"," ") in tags_upper) or (str(pair['m']['label']).replace("_"," ") in tags_upper) or (str(pair['n']['label']) in tags_upper) or (str(pair['m']['label']) in tags_upper):        
        subjects.append(pair['n']['label'])
        objects.append(pair['m']['label'])
        if ctgr:
            categories.append(pair['n']['category'])
  
    matched_labels=list(set(subjects+objects))
    hop1_labels = [i for i in matched_labels if i not in tags_upper]

    ##Hop 
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
    synonyms=[]
    for i in range(len(subjects)):
        if subjects[i]==objects[i]:
            continue
        if ctgr:
            rule='isSynonymWith("'+subjects[i]+'","'+objects[i]+'","'+categories[i]+'").'
        else:
           rule='isSynonymWith("'+subjects[i]+'","'+objects[i]+'").'
        synonyms.append(rule)

    set_syn=set(objects)
    objects=list(set_syn)
    return objects,synonyms

