import nltk
import clingo
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re
import json


def split_errors(model):
    init_model=model.copy()
    A=[]
    B=[]
    AB=[]
    C=[]
    for m in model:
        if m.startswith('errorCaseA('):
            tok=nltk.word_tokenize(m)
            A.append(tok[2])
            init_model.remove(m)
        if m.startswith('errorCaseB('):
            tok=nltk.word_tokenize(m)
            B.append(tok[2])
            init_model.remove(m)
        if m.startswith('errorCaseAB('):
            tok=nltk.word_tokenize(m)
            AB.append((tok[2],tok[4]))
            init_model.remove(m)
        if m.startswith('errorCaseC('):
            tok=nltk.word_tokenize(m)
            C.append(tok[2])
            init_model.remove(m)

    return A,B,C,AB,init_model

def convertTopClasses(topcl):
    domain =["Aeroplane", "Animal_Wing", "Arm", "Artifact_Wing", "Beak", "Bicycle", "Bird", "Boat", "Body", "Bodywork", "Bottle", "Bus", "Cap", "Car", "Cat", "Chain_wheel", "Chair", "Coach", "Cow", "Dog", "Door", "Ear", "Ebrow", "Engine", "Eye", "Foot", "Hair", "Hand", "Handlebar", "Head", "Headlight", "Hoof", "Horn", "Horse", "Leg", "License_plate", "Locomotive", "Mirror", "Motorbike", "Mouth", "Muzzle", "Neck", "Nose", "Person", "Plant", "Pot", "Pottedplant", "Saddle", "Screen", "Sheep", "Sofa", "Stern", "Tail", "Torso", "Train", "Tvmonitor", "Wheel", "Window", "Diningtable","Other"]
    new_topCl=[]
    for pair in topcl:
        l=list(pair)
        l[0]=domain[l[0]]
        new_topCl.append(l)
    return(new_topCl)

def clean_model(old_model):
    model=[]
    for m in old_model:
        if m.startswith("label(") or m.startswith("box("):
            model.append(m)
    return model


def parse_aspPrgram(program):
    if type(program) is str: #and re.sub(r'\n%[^\n]*', '\n', program).strip().endswith(('.', ']')):
        lines = program.split('\n')
        lines=[re.sub('\s+',' ',l) for l in lines]
        lines=[l for l in lines if l.replace(' ',"") and not l.startswith(' %')]
    aspRules=[]
    rule=''
    for l in lines:
        if l.startswith('%'):
            continue 
        rule+=l
        if rule.endswith("."):
            aspRules.append(rule)
            rule=''
       
    return aspRules

def runClingo(aspRules):
    control = clingo.Control(['--warn=none', '1'])
    # Load rules from the data structure
    for rule in aspRules:
        control.add("base",[],rule.strip())
    # Ground the rules (i.e., generate all possible facts)
    control.ground([("base", [])])
    # Solve the ASP program
    # control.solve()
    model=None
    model=control.solve(yield_=True).model()
    if model:
        model=model.symbols(shown=True)
        model=[str(x) for x in model]
    return model



def changeLabel(old_model,topClasses,b):
    # print(topClasses)
    model=clean_model(old_model)
    # print(model)
    old_m=None
    cmodel=model.copy()
    for m in cmodel:
        if m.startswith('label('):
            tok=nltk.word_tokenize(m)
            # print(tok[6],tok[9])
            if b==tok[6]:
                old_lbl=tok[9]
                old_tok=tok
                model.remove(m)
    # print(model)

    new_label=None
    if len(topClasses[b]) > 0:
        del topClasses[b][0]
        if(len(topClasses[b])>0):
            if float(topClasses[b][0][1]) > 0:
                new_label=topClasses[b][0][0] 
    if new_label==None:
        new_label='Other' 
    print("     ",old_lbl,'->',new_label)

    new_tok=old_tok.copy()       
    new_tok[9]=new_label
    new_m=TreebankWordDetokenizer().detokenize(new_tok)
    new_m=new_m.replace(" ","")
    model.append(new_m)
    # print(model)
    return model,topClasses 

def caseA(A,model,topClasses,aspProgram,error_dict):
    init_aspProgram=aspProgram.copy()   
    print("Handling Case A Errors...")
    model=clean_model(model)
    for rule in aspProgram:
        if rule.replace(" ","").startswith("over90"):
            rep_rule=rule
            break
    aspProgram.remove(rep_rule)
    aspProgram.append('over90(Bmin,Bmax) :- box(_,B1,Xmin1,Ymin1,Xmax1,Ymax1,_), box(_,B2,Xmin2,Ymin2,Xmax2,Ymax2,_), overlap(B1,B2), area(B1, A1), area(B2, A2), Amin=#min{A1;A2}, Amax=#max{A1;A2}, area(Bmin,Amin), area(Bmax,Amax), Ymax=#min{Ymax1;Ymax2}, Ymin=#max{Ymin1;Ymin2}, Xmax=#min{Xmax1;Xmax2}, Xmin=#max{Xmin1;Xmin2}, Aovl=(Ymax - Ymin) * (Xmax- Xmin), 5 <= ((100*Aovl)/ Amin).')
    aspProgram=combine_outputs_asp(model,aspProgram)
    model=runClingo(aspProgram)
    newA,B,C,AB,model=split_errors(model)
    wrong_detection_coords=[x for x in A if x not in newA]
    newA=[x for x in A if x in newA]
    if newA:
        error_dict['A']+=newA
    # print(wrong_detection_coords)
    for b in wrong_detection_coords:
        print('     The model returned WRONG coordinates for box '+b)
    while(len(newA)>0):
        for b in newA:
            print("     Error at ",b)
            model,topClasses=changeLabel(model,topClasses,b)
            print("     Label Changed")
        
        init_aspProgram=clean_aspProgram(init_aspProgram)
        aspProgram=combine_outputs_asp(model,init_aspProgram)
        ##THE PROBLEM IS THAT SOMEHOW THIS CONTAINS THE OUTPUTS
        # print(init_aspProgram)
        ##RUN CLINGO WITH NEW MODEL AND ASP RULES
        model=runClingo(aspProgram)
        # print(model)
        newA,B,C,AB,model=split_errors(model)
        newA=[x for x in newA if x not in wrong_detection_coords]
    return newA,B,C,AB,model,topClasses

def clean_aspProgram(aspProgram):
    init_pr=aspProgram.copy()
    for r in init_pr:
        if r.startswith('label'):
            aspProgram.remove(r)
    return aspProgram

# def caseA(A,model,topClasses,aspProgram):
#     print("Handling Case A Errors...")
#     # model=clean_model(model)
#     # newA,model=errors_with_lowerIR('errorCaseA',model,aspProgram)
#     # wrong_detection_coords=[x for x in A if x not in newA]
#     # # print(wrong_detection_coords)
#     # for b in wrong_detection_coords:
#     #     print("Wrong Coordinates Errors:")
#     #     print('     The model returned WRONG coordinates for box '+b)
#     while(len(A)>0):
#         for b in A:
#             print("     Error at ",b)
#             model,topClasses=changeLabel(model,topClasses,b)
#             print("     Label Changed")
#         # print(model)
#         aspProgram=combine_outputs_asp(model,aspProgram)
#         ##RUN CLINGO WITH NEW MODEL AND ASP RULES
#         model=runClingo(aspProgram)
#         A,B,C,AB,model=split_errors(model)
#     # A,B,C,AB,model=split_errors(model)
#     return A,B,C,AB,model,topClasses

def caseB(B,model,topClasses,aspProgram):
    print("Handling Case B Errors...")
    while(len(B)>0):
        for b in B:
            print("     Error at ",b)
            model,topClasses=changeLabel(model,topClasses,b)
            print("     Label Changed")
        aspProgram=clean_aspProgram(aspProgram)
        aspProgram=combine_outputs_asp(model,aspProgram)
        ##RUN CLINGO WITH NEW MODEL AND ASP RULES
        model=runClingo(aspProgram)
        A,B,C,AB,model=split_errors(model)
    return  A,B,C,AB,model,topClasses

def changeTomostConfPair(pair,model,topClasses):
    # print(topClasses)
    if topClasses[pair[0]] and topClasses[pair[1]]:
        confleft=float(topClasses[pair[0]][1][1])+float(topClasses[pair[1]][0][1])
        confright=float(topClasses[pair[0]][0][1])+float(topClasses[pair[1]][1][1])

    if confleft >= confright:
        model,topClasses=changeLabel(model,topClasses,pair[0])
    else:
        model,topClasses=changeLabel(model,topClasses,pair[1])
    
    return model,topClasses

def caseAB(AB,model,topClasses,aspProgram):
    print("Handling Case AB Errors...")
    while(len(AB)>0):
        for pair in AB:
            print("     Error at ",pair)
            model,topClasses=changeTomostConfPair(pair,model,topClasses)
            print("     Label Changed")
        aspProgram=clean_aspProgram(aspProgram)
        aspProgram=combine_outputs_asp(model,aspProgram)
        ##RUN CLINGO WITH NEW MODEL AND ASP RULES
        model=runClingo(aspProgram)
        A,B,C,AB,model=split_errors(model)
    return A,B,C,AB,model,topClasses



CN_pos_class_detectionRuleset=r'''
isNotUniquePart(X,C):-partOf(X,Y,C),partOf(X,Z,C),Y!=Z.
isUniquePartOf(X,Y,C):-part(X,C),object(Y,C),partOf(X,Y,C),not isNotUniquePart(X,C).
possibleClass(B,Y,C):-single(B),errorCaseC(B),partBox(B,L,_,C),partOf(L,Y,C).
detClass(Y,C):-single(B),partBox(B,L,_,C),errorCaseC(B),isUniquePartOf(L,Y).
'''
CN_numberOfObjectsRuleset=r'''
numMissingobjects(N):- N=#count{C: possibleClass(_,_,C)}.
'''




def find_possible_classes(C,old_aspProgram,model,kdomain):
    pclass_dict={}
    hashtag='#'
    print("     Object bounding box/es are missing")
    if kdomain=='CN':
        pos_class_detectionRuleset=r'''
        isNotUniquePart(X,C):-partOf(X,Y,C),partOf(X,Z,C),Y!=Z.
        isUniquePartOf(X,Y,C):-part(X,C),object(Y,C),partOf(X,Y,C),not isNotUniquePart(X,C).
        possibleClass(B,Y,C):-single(B),errorCaseC(B),partBox(B,L,_,C),object(Y,C),partOf(L,Y,C).
        detClass(Y,C):-single(B),partBox(B,L,_,C),object(Y,C),errorCaseC(B),isUniquePartOf(L,Y,C).
        '''
        numberOfObjectsRuleset=r'''
        in_category(Y,Cat):-possibleClass(_,Y,Cat).
        numMissingobjects(N):- N='''+hashtag+r'''count{Y: in_category(_,Y)}.
        '''
    else:

        pos_class_detectionRuleset=r'''
    isNotUniquePart(X):-partOf(X,Y),partOf(X,Z),Y!=Z.
    isUniquePartOf(X,Y):-part(X),object(Y),partOf(X,Y),not isNotUniquePart(X).
    possibleClass(B,Y):-single(B),errorCaseC(B),label(_,_,B,L),partOf(L,Y).
    detClass(Y):-single(B),label(_,_,B,L),errorCaseC(B),isUniquePartOf(L,Y).
    '''
        numberOfObjectsRuleset=r'''
    numPosClasses(N):-N='''+hashtag+r'''count{X:possibleClass(_,X)}.
    category(1..N):-numPosClasses(N).
    {in_category(S, C) : possibleClass(_,S), category(C)}.
    :-0{in_category(_,_)}N-1,numPosClasses(N).
    sharePart(Y1,Y2):-partOf(X,Y1),partOf(X,Y2),Y1!=Y2.
    :- in_category(S1, C), in_category(S2, C), S1 != S2, not sharePart(S1,S2).
    :- in_category(S1, C1), in_category(S2, C2), C1 != C2, partOf(P,S1), partOf(P,S2).
    numMissingobjects(N):- N='''+hashtag+r'''count{Y: in_category(_,Y)}.'''
        
    single_b=[m for m in model if m.startswith('single')]
    filt_singles=[]
    for m in single_b:
        tok=nltk.word_tokenize(m)
        if tok[2] in C:
             filt_singles.append(m)

    lbl_model=[m for m in model if m.startswith('label')]
    model=filt_singles+lbl_model
    # model=[x for x in model if x.startswith('label') or x.startswith('single')]
    pos_class_detectionRuleset=parse_aspPrgram(pos_class_detectionRuleset)
    aspProgram=old_aspProgram+pos_class_detectionRuleset
    aspProgram=combine_outputs_asp(model,aspProgram)
    model=runClingo(aspProgram)
    pclass=[]
    cclass=[]
    catclass=[]
    for m in model:
        if m.startswith("possibleClass"):
            pclass.append(m)
        if m.startswith("detClass"):
            cclass.append(m)

    numberOfObjectsRuleset=parse_aspPrgram(numberOfObjectsRuleset)
    aspProgram=old_aspProgram+numberOfObjectsRuleset
    aspProgram=combine_outputs_asp(pclass,aspProgram)
    model=runClingo(aspProgram)
    if model:
        for m in model:
            if m.startswith('in_category'):
                catclass.append(m)
            # if m.startswith('numMissing'):
            #     tok=nltk.word_tokenize(m)
            #     print("         At least ",tok[2]," whole object is missing")
        pclass_dict=parse_classes(cclass,catclass,kdomain)
        print("         At least ",len(pclass_dict.keys())," whole object is missing")
    else:
        pclass_dict[0]=[]
        print("         At least 1 whole object is missing")
        if cclass:
            for c in cclass:
                print("         ",c) 
                toks=nltk.word_tokenize(c)
                pclass_dict[0].append(toks[3])
                

        else:
            for p in pclass:
                print("         ",p)
                toks=nltk.word_tokenize(p)
                pclass_dict[0].append(toks[5])
    return pclass_dict

def parse_classes(detclasses,catclasses,kdomain):
    classCat_dict={}
    for c in catclasses:
        # print(c)
        toks=nltk.word_tokenize(c)
        if kdomain=='CN':
            if toks[7] not in classCat_dict:
                classCat_dict[toks[7]]=[]
            classCat_dict[toks[7]].append(toks[3])
        else:
            if toks[5].replace(',','') not in classCat_dict:
                classCat_dict[toks[5].replace(',','')]=[]
            classCat_dict[toks[5].replace(',','')].append(toks[3])

    for det in detclasses:
        toks=nltk.word_tokenize(det)
        uni_class=toks[3]
        for key in classCat_dict:
            if uni_class in classCat_dict[key]:
                classCat_dict[key]=[uni_class] 

    if kdomain=='CN':
        dict_copy = {}
        i=0
        rem=[]
        catgs=list(classCat_dict.keys())
        for i in range(0,len(catgs)):
             for j in range(i+1,len(catgs)):
                val1=classCat_dict[catgs[i]]
                val2=classCat_dict[catgs[j]]
                intersection=[x for x in val1 if x in val2]
                if len(intersection)>0:
                    if len(val1) >= len(val2):
                        rem.append(catgs[j])
                    else:
                        rem.append(catgs[i])

        for r in rem:
            if r in classCat_dict:
                del classCat_dict[r]


    return classCat_dict

    

# def caseC(C,old_model,aspProgram):
#     print("Handling Case C Errors...")
#     model=clean_model(old_model)
#     # newC,model=errors_with_lowerIR('errorCaseC',model,aspProgram)
#     # wrong_detection_coords=[x for x in C if x not in newC]
#     # # print(wrong_detection_coords)
#     # for b in wrong_detection_coords:
#     #     print("Wrong Coordinates Errors:")
#     #     print('     The model returned WRONG coordinates for box '+b)
#     if C:
#         print("Undetected Objects Errors:")
#         find_possible_classes(C,aspProgram,model)
    
def caseC(C,old_model,aspProgram,error_dict,kdomain):
    print("Handling Case C Errors...")
    model=clean_model(old_model)
    for rule in aspProgram:
        if rule.replace(" ","").startswith("over90"):
            rep_rule=rule
            break
    aspProgram.remove(rep_rule)
    aspProgram.append('over90(Bmin,Bmax) :- box(_,B1,Xmin1,Ymin1,Xmax1,Ymax1,_), box(_,B2,Xmin2,Ymin2,Xmax2,Ymax2,_), overlap(B1,B2), area(B1, A1), area(B2, A2), Amin=#min{A1;A2}, Amax=#max{A1;A2}, area(Bmin,Amin), area(Bmax,Amax), Ymax=#min{Ymax1;Ymax2}, Ymin=#max{Ymin1;Ymin2}, Xmax=#min{Xmax1;Xmax2}, Xmin=#max{Xmin1;Xmin2}, Aovl=(Ymax - Ymin) * (Xmax- Xmin), 5 <= ((100*Aovl)/ Amin).')
    aspProgram=combine_outputs_asp(model,aspProgram)
    model=runClingo(aspProgram)
    _,_,newC,_,model=split_errors(model)
    wrong_detection_coords=[x for x in C if x not in newC]
    # print(wrong_detection_coords)
    for b in wrong_detection_coords:
        print('     The model returned WRONG coordinates for box '+b)
    if newC:
        error_dict['C']+=newC
        pclass_dict=find_possible_classes(newC,aspProgram,model,kdomain)
        error_dict['PossibleClasses']=pclass_dict
    

def combine_outputs_asp(model,aspProgram):
    new_program=aspProgram
    for m in model:
        new_program.append(m+".")
    return new_program

def resolve_errors(img_name,error_dict,model,topClasses,aspProgram,kdomain):
    error_dict[img_name]={}
    error_dict[img_name]['A']=[]
    error_dict[img_name]['B']=[]
    error_dict[img_name]['C']=[]
    error_dict[img_name]['AB']=[]

    #split error cases and remove them from model
    A,B,initC,AB,model=split_errors(model)
    error_dict[img_name]['B']+=B
    error_dict[img_name]['AB']+=AB
    print("A->",A)
    print("B->",B)
    print("C->",initC)
    print("AB->",AB)
    #parse ASP ruleset and convert it to clingo format ready
    aspProgram=parse_aspPrgram(aspProgram)
    #covert top classes from ids to labels
    for key in topClasses:
        new_cl=convertTopClasses(topClasses[key])
        topClasses[key]=new_cl

    old_asp=aspProgram.copy()
    old_model=model.copy()
    if initC:
       caseC(initC,old_model,old_asp,error_dict[img_name],kdomain)
    C=None
    while A or B or AB:
        if A:
            A,B,C,AB,model,topClasses=caseA(A,model,topClasses,aspProgram,error_dict[img_name])
            print("A->",A)
            print("B->",B)
            print("C->",C)
            print("AB->",AB)
        if B:
            A,B,C,AB,model,topClasses=caseB(B,model,topClasses,aspProgram)
            print("A->",A)
            print("B->",B)
            print("C->",C)
            print("AB->",AB)
        if AB:
            A,B,C,AB,model,topClasses=caseAB(AB,model,topClasses,aspProgram)
            print("A->",A)
            print("B->",B)
            print("C->",C)
            print("AB->",AB)
    if C !=None:
        C=[x for x in C if x not in initC]
        if C:
            print('New Case C errror occurrences')
            caseC(C,model,aspProgram,error_dict[img_name],kdomain)
       
    # print(error_dict)
    # print(model)
    return model,error_dict

