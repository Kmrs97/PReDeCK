import json

def calc_Area(coords):
    area=(coords[2]-coords[0])*(coords[3]-coords[1])
    return area

def getJSONdata(filename):
    file = open(filename)
    data = json.load(file)
    return data


def iou(box1, box2):
    # box1 and box2 are lists of [xmin, ymin, xmax, ymax]
    # calculate areas of both bounding boxes
    area_box1 = calc_Area(box1)
    area_box2 = calc_Area(box2)
    # find coordinates of the intersection bounding box
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    # calculate area of intersection bounding box
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    # calculate union area
    union_area = (area_box1 + area_box2) - intersection_area
    # calculate IoU
    iou = intersection_area / union_area
    return iou


def find_respect_box(dbox,gt):
    max_index=None
    max_iou=0
    for index in gt:
        iou_score=iou(dbox['bbox'],gt[index]["bbox"])
        if iou_score > max_iou:
            max_iou=iou_score
            max_index=index
    if max_index==None or max_iou < 0.5:
        # print("             Wrong Model detection!")
        return False,max_index
    else:
        if dbox['label']!=gt[max_index]['label']:
            # print("         Error Correctly Detected ")
            return False,max_index
        else:
            # print("         Error Falsely Detected ")
            return True,max_index



def evaluateA_B_errors(elist,init_dets,gt):
    correct=0
    incorrect=0
    for b in elist:
        res,_=find_respect_box(init_dets[b],gt)
        if not res:
            correct+=1
        else:
            incorrect+=1
    total=correct+incorrect
    # print(correct)
    # print(incorrect)
    # print(total)
    precision=(correct*100)/total
    # print("     Precision= ",precision)
    return correct,total

def evaluateAB_errors(pairs,init_dets,gt):
    correct=0
    incorrect=0
    for pair in pairs:
        res1,_=find_respect_box(init_dets[pair[0]],gt)
        res2,_=find_respect_box(init_dets[pair[1]],gt)
        if (not res1) or (not res2):
            correct+=1
        else:
            incorrect+=1
    total=correct+incorrect
    # print(correct)
    # print(incorrect)
    # print(total)
    precision=(correct*100)/total
    # print("     Precision= ",precision)
    return correct,total

def naive_method(missing_objs,elist):
    #naive
    if missing_objs > 0 and len(elist) > 0:
    #    print('True')
       return True
    else: 
        # print('False')
        return False 

def C_method2(missing_objs,elist,poss_classes):
    # print(poss_classes)
    if naive_method(missing_objs,elist):
        numOfCategories_missing=len(poss_classes.keys())
        if numOfCategories_missing==0:
            numOfCategories_missing=1
        if missing_objs == numOfCategories_missing:
            return True
    return False

def C_method3(missing_objs,missing_idxs,elist,gt,poss_classes):
    # print(poss_classes)
    scores=[]
    corrects=0
    incorrects=0
    if naive_method(missing_objs,elist):
        for idx in missing_idxs:
            gt_label=gt[idx]['label']
            # print(gt_label)
            for key in poss_classes:
                if gt_label in poss_classes[key]:
                    # print("Correct Possible Classesfro index ",idx," with score=",1/len(poss_classes[key]))
                    scores.append(1/len(poss_classes[key]))
                    corrects+=1
                else:
                    # print('Incorect Possible Classes for index ',idx)
                    scores.append(0)
                    incorrects+=1
        # print(sum(scores)/len(scores))
        return [sum(scores),len(scores),corrects,corrects+incorrects]
    else:
        return False

def C_method4(missing_objs,missing_idxs,elist,gt,poss_classes):
    # print(poss_classes)
    res=C_method3(missing_objs,missing_idxs,elist,gt,poss_classes)
    res2=C_method2(missing_objs,elist,poss_classes)
    if res and res2 and res[2]==res[3]:
        return True
    else: 
        return False
    

def getMissingObjectsInfo(init_dets,gt):
    objects=["Aeroplane","Bicycle","Bird","Boat","Bottle","Bus","Car", "Cat","Chair","Cow", "Dog","Horse"
             ,"Motorbike","Person","Pottedplant","Sheep", "Sofa","Train", "Tvmonitor", "Diningtable"]
    gt_objs=0
    det_objs=0
    gt_idxs=[]
    for b in gt:
        if gt[b]['label'] in objects:
            gt_idxs.append(b)
            gt_objs+=1

    matched_idxs=[]
    for det in init_dets:
        if init_dets[det]['label'] in objects:
            res,index=find_respect_box(init_dets[det],gt)
            if res and (index not in matched_idxs) :
                det_objs+=1
                matched_idxs.append(index)
    
    missing_objs=gt_objs-det_objs
    missing_idxs=[x for x in gt_idxs if x not in matched_idxs]
    # print(missing_objs)
    return missing_objs,missing_idxs

def evaluateC_errors(elist,init_dets,gt,poss_classes,method):
    missing_objs,missing_idxs=getMissingObjectsInfo(init_dets,gt)
    if method==1:
        return naive_method(missing_objs,elist)
    if method==2:
        return C_method2(missing_objs,elist,poss_classes)
        # print(res)
        return res
    if method==3:
        return C_method3(missing_objs,missing_idxs,elist,gt,poss_classes)
    if method==4:
        return C_method4(missing_objs,missing_idxs,elist,gt,poss_classes)
    

def calculateOverall_CaseC_Recall(opath):
    init_detections=getJSONdata(opath+'groundtruth2/GT2_inferences_JSON.json')
    ground_truth=getJSONdata(opath+'JSON_annotations.json')
    Total_imgs_missing_objects=0
    for img in init_detections:
        missing_objs,_=getMissingObjectsInfo(init_detections[img][0],ground_truth[img])
        if missing_objs>0:
            Total_imgs_missing_objects+=1
    
    return Total_imgs_missing_objects

               
##Change to OUTPUT DIR of error detection experiments
opath=#'ED OUTPUT FILE'




error_json=getJSONdata(opath+'gtKnowledge_ErrorDetection/detected_errors.json')

# error_json=getJSONdata(opath+'CN_ErrorDetection/detected_errors.json')

init_detections=getJSONdata(opath+'groundtruth2/GT2_inferences_JSON.json')
ground_truth=getJSONdata(opath+'JSON_annotations.json')
A_corr=0
total_A=0

B_corr=0
total_B=0

AB_corr=0
total_AB=0

C_corr1=0
C_corr2=0
C_corr4=0
total_C=0

Cmethod3_score=0
Cmethod3_len=0

Cmethod3_corr=0
Cmethod3_total=0


for img in error_json:
    # print(img)
    
    if error_json[img]['A']:
        # print('    Case A')
        corr,total=evaluateA_B_errors(error_json[img]['A'],init_detections[img][0],ground_truth[img])
        A_corr+=corr
        total_A+=total
    if error_json[img]['B']:
        # print('     Case B')
        corr,total=evaluateA_B_errors(error_json[img]['B'],init_detections[img][0],ground_truth[img])
        B_corr+=corr
        total_B+=total
    if error_json[img]['AB']:
        # print('     Case AB')
        corr,total=evaluateAB_errors( error_json[img]['AB'],init_detections[img][0],ground_truth[img])
        AB_corr+=corr
        total_AB+=total
    if error_json[img]['C']:
        total_C+=1
        # print(img)
        # print('     Case C')
        if evaluateC_errors(error_json[img]['C'],init_detections[img][0],ground_truth[img],error_json[img]['PossibleClasses'],method=1):
            C_corr1+=1
        if evaluateC_errors(error_json[img]['C'],init_detections[img][0],ground_truth[img],error_json[img]['PossibleClasses'],method=2):
            C_corr2+=1
        res=evaluateC_errors(error_json[img]['C'],init_detections[img][0],ground_truth[img],error_json[img]['PossibleClasses'],method=3)
        if not res:
            continue
        else:
            Cmethod3_score+=res[0]
            Cmethod3_len+=res[1]
            Cmethod3_corr+=res[2]
            Cmethod3_total+=res[3]
        if evaluateC_errors(error_json[img]['C'],init_detections[img][0],ground_truth[img],error_json[img]['PossibleClasses'],method=4):
            C_corr4+=1
     


print('Precisions:')
try:
    print('     ',A_corr)
    print('     ',total_A)
    print("     Error Case A=",round((A_corr*100)/total_A,2))
except ZeroDivisionError: 
    print("     Error Case A=",0) 

try:
    print('     ',B_corr)
    print('     ',total_B)
    print("     Error Case B=",round((B_corr*100)/total_B,2))
except ZeroDivisionError: 
    print("     Error Case B=",0) 
try:
    print('     ',AB_corr)
    print('     ',total_AB)
    print("     Error Case AB=",round((AB_corr*100)/total_AB,2))
except ZeroDivisionError: 
    print("     Error Case AB=",0) 

print("     Error Case C") 
try:
    print("     Method 1")
    print('     ',C_corr1)
    print('     ',total_C)
    print("             Correctly detected Errors in ",round((C_corr1*100)/total_C,2),"% of Case C error images")
except ZeroDivisionError: 
    print("     Method 1=",0) 
try:
    print("     Method 2")
    print("             Found exactly how many Objects are missing in ",round((C_corr2*100)/total_C,2),"% of Case C error images")
except ZeroDivisionError: 
    print("     Method 2=",0) 

try:
    print("     Method 3")
    print("             Possible Object Class Correctly Detected in ",round((Cmethod3_corr*100)/Cmethod3_total,2),"% of undetected Objects")
    print("             With score=",round((Cmethod3_score*100)/Cmethod3_len,2))
except ZeroDivisionError: 
    print("     Method 3=",0) 

try:
    print("     Method 4")
    print("             Found exactly how many Objects are missing and their Possible Class in ",round((C_corr4*100)/total_C,2),"% of Case C error images")
except ZeroDivisionError: 
    print("     Method 4=",0) 


total_images_withErrorC=calculateOverall_CaseC_Recall(opath)
Rec=round((C_corr1*100)/total_images_withErrorC,2)
print("Recall of Case C errors= ",Rec)
print("Images with objects missing=",total_images_withErrorC)
