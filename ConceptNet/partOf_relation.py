import requests
import json
from py2neo import  Graph
##Converts the information about PartOf, HasA, HasContext , to ASP atoms

def generate_rules_from_graph(tags,superclasses,synonyms,ctgr=False):
    #Complete the credentials
    USERNAME = "" 
    PASSWORD = "" 
    sample_tags=list(set(tags+superclasses+synonyms))
    tags_upper = [tag.upper() for tag in sample_tags]
    subjects=[]
    objects=[]
    categories=[]
    graph = Graph(scheme="bolt", host="localhost", port=YOURPORT, auth=(USERNAME, PASSWORD))

#PartOf relation
    resp=graph.run("MATCH (n)-[r:PARTOF]->(m) RETURN n,m").data()
    for pair in resp:
     if (str(pair['n']['label']).replace("_"," ") in tags_upper) or (str(pair['m']['label']).replace("_"," ") in tags_upper) or (str(pair['n']['label']) in tags_upper) or (str(pair['m']['label']) in tags_upper):
        subjects.append(pair['n']['label'])
        objects.append(pair['m']['label'])
        if ctgr:
            categories.append(pair['n']['category'])


    matched_labels=list(set(subjects+objects))
    hop1_labels = [i for i in matched_labels if i not in tags_upper]

    ##Hop 2
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
    partOf=[]
    for i in range(len(subjects)):
        #partOf(L1,L2). (partOf/2)
        if subjects[i]==objects[i]:
            continue
        if ctgr:
            rule='partOf("'+subjects[i]+'","'+objects[i]+'","'+categories[i]+'").'
        else:
            rule='partOf("'+subjects[i]+'","'+objects[i]+'").'
        partOf.append(rule)

#hasA relation
    subjects=[]
    objects=[]
    categories=[]
    resp=graph.run("MATCH (n)-[r:HASA]->(m) RETURN n,m").data()
    for pair in resp:
     if (str(pair['n']['label']).replace("_"," ") in tags_upper) or (str(pair['m']['label']).replace("_"," ") in tags_upper):
        subjects.append(pair['n']['label'])
        objects.append(pair['m']['label'])
        if ctgr:
            categories.append(pair['n']['category'])

    matched_labels=list(set(subjects+objects))
    hop1_labels = [i for i in matched_labels if i not in tags_upper]

    ##Hop 2
    for pair in resp:
     if (str(pair['n']['label']).replace("_"," ") in hop1_labels) or (str(pair['m']['label']).replace("_"," ") in hop1_labels):
        subjects.append(pair['n']['label'])
        objects.append(pair['m']['label'])
        if ctgr:
            categories.append(pair['n']['category'])


    subjects=[x.capitalize() for x in subjects]
    objects=[x.capitalize() for x in objects]
    if ctgr:
        categories=[x.capitalize() for x in categories]
    hasA=[]
    for i in range(len(subjects)):
        #hasA(L1,L2).
        if subjects[i]==objects[i]:
            continue
        if ctgr:
            rule='hasA("'+objects[i]+'","'+subjects[i]+'","'+categories[i]+'").'
        else:
            rule='hasA("'+objects[i]+'","'+subjects[i]+'").'
        hasA.append(rule)


#hasContext relation
    subjects=[]
    objects=[]
    categories=[]
    resp=graph.run("MATCH (n)-[r:HASCONTEXT]->(m) RETURN n,m").data()
    for pair in resp:
     if (str(pair['n']['label']).replace("_"," ") in tags_upper) or (str(pair['m']['label']).replace("_"," ") in tags_upper):
        subjects.append(pair['n']['label'])
        objects.append(pair['m']['label'])
        if ctgr:
            categories.append(pair['n']['category'])

    matched_labels=list(set(subjects+objects))
    hop1_labels = [i for i in matched_labels if i not in tags_upper]

    ##Hop 2
    for pair in resp:
     if (str(pair['n']['label']).replace("_"," ") in hop1_labels) or (str(pair['m']['label']).replace("_"," ") in hop1_labels):
        subjects.append(pair['n']['label'])
        objects.append(pair['m']['label'])
        if ctgr:
            categories.append(pair['n']['category'])
    

    subjects=[x.capitalize() for x in subjects]
    objects=[x.capitalize() for x in objects]
    if ctgr:
        categories=[x.capitalize() for x in categories]
    hasContext=[]
    for i in range(len(subjects)):
        #hasA(L1,L2).
        if subjects[i]==objects[i]:
            continue
        if ctgr:
            rule='hasContext("'+subjects[i]+'","'+objects[i]+'","'+categories[i]+'").'
        else:
            rule='hasContext("'+subjects[i]+'","'+objects[i]+'").'
        hasContext.append(rule)
    return partOf,hasA,hasContext

