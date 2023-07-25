import nltk
import json


def separate_rules(model):
    boxes=[]
    labels=[]
    spatial=[]
    semantic=[]
    fired_rule=[]
    for m in model:
        # print(m)
        if m.__contains__("box("):
            boxes.append(m)  
        if m.__contains__("label"):
            labels.append(m)
        if m.__contains__("spatial_partOf"):
            spatial.append(m)
        if m.__contains__("semantic_partOf"):
            semantic.append(m)
        if m.__contains__("fired("):
            fired_rule.append(m)
    return boxes,labels,semantic,spatial,fired_rule

def parse_string(model):
    dict_obj={}
    sp_sub=[]
    sp_obj=[]
    se_sub=[]
    se_obj=[]
    boxes,labels,semantic,spatial,fired_rule=separate_rules(model)
    for b in boxes:
        tokens=b.split(',')
        dict_obj[str(tokens[1])]={}
        dict_obj[str(tokens[1])]['bbox']=list(map(int,tokens[2:6]))
        dict_obj[str(tokens[1])]['conf']=int(tokens[6].replace(")",""))/1000
        dict_obj[str(tokens[1])]['Spatial_Part']=[]
        dict_obj[str(tokens[1])]['Semantic_Part']=[]
    for l in labels:
        tokens = list(nltk.word_tokenize(l))
        dict_obj[str(tokens[6])]['label']=tokens[9]
    for sp in spatial:
        tokens = list(nltk.word_tokenize(sp))
        sp_sub.append((tokens[2],tokens[5]))
        sp_obj.append((tokens[8],tokens[11]))
        dict_obj[str(tokens[8])]['Spatial_Part'].append(str(tokens[2]))
    for se in semantic:
        tokens = list(nltk.word_tokenize(se))
        se_sub.append((tokens[2],tokens[5]))
        se_obj.append((tokens[8],tokens[11]))
        dict_obj[str(tokens[8])]['Semantic_Part'].append(str(tokens[2]))

    # spatial_dict=createDict(sp_obj,sp_sub)
    # for key in spatial_dict:
    #     dict_obj[key]['Spatial_parts']=spatial_dict[key]
    
    # semantic_dict=createDict(se_obj,se_sub)
    # for key in semantic_dict:
    #     dict_obj[key]['Semantic_parts']=semantic_dict[key]
    


    # print("Bounding Boxes:")
    # print(json.dumps(dict_obj,indent=4))
    # print()
    # print("## Spatial PartOf Relationships##")
    # for i in range(len(sp_obj)):
    #     print(sp_sub[i],"---is Spatially PartOf--->",sp_obj[i])
    
    # print()
    # print("## Semantic PartOf Relationships##")
    # for i in range(len(se_obj)):
    #     print(se_sub[i],"---is Semantically PartOf--->",se_obj[i])
    # print()
    # print('Constraint Fired')
    
    for f in fired_rule:
        print(f)
    return dict_obj

   
