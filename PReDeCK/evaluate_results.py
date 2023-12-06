import json
import matplotlib.pyplot as plt
import os.path
from os import path

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

    
def find_respect_box(box,av_indices,inf_boxes,labels,threshold):
    objects=["Aeroplane","Bicycle","Bird","Boat","Bottle","Bus","Car", "Cat","Chair","Cow", "Dog","Horse",
    "Motorbike","Person","Pottedplant","Sheep", "Sofa","Train", "Tvmonitor", "Diningtable"]
    max_index=None
    iou_scores=[]
    if box['label']=="Other":
        return max_index,av_indices
    if box["label"]  in objects:
        threshold=0.5
    for index in av_indices:
        iou_scores.append(iou(box['bbox'],inf_boxes[index]["bbox"]))
    max_iou=max(iou_scores)
    if max_iou>=threshold:
        max_index=av_indices[iou_scores.index(max_iou)]

    if labels==True and max_index!=None:
        if box['label']!=inf_boxes[max_index]['label']:
            max_index=None
    # av_indices.remove(max_index)
    return max_index,av_indices


  

##Format:
##{images[models{classes:[indexes]}]}
def separateDetectionsToClasses(inference_res):
    inferences={}
    for img in inference_res:
        inferences[img]=[]
        if len(inference_res[img]) <1:
            continue
        for model in inference_res[img]:
            classes={}
            for index in model:
                lbl=model[index]['label']
                if lbl not in classes:
                    classes[lbl]=[]
                classes[lbl].append(index)
        inferences[img].append(classes)
    return inferences            

def separateGTToClasses(ground_truth):
    inferences={}
    for img in ground_truth:
        inferences[img]={}
        for index in ground_truth[img]:
            lbl=ground_truth[img][index]['label']
            if lbl not in inferences[img]:
                inferences[img][lbl]=[]
            inferences[img][lbl].append(index)
    return inferences  

def sort_indexes(label,indexes,detections):
    confidences=[]
    for idx in indexes:
        confidences.append(detections[idx]['conf'])
    sorted_by_conf=[conf for _,conf in sorted(zip(confidences,indexes),reverse=True)]
    return sorted_by_conf



def calculatePR(det_index,gt_index,detections,ground_truth):
    # print(detections)
    # print(ground_truth)
    TP,FP,FN=0,0,0
    # if(len(det_index)>len(gt_index)):
    #     FP+=len(det_index)-len(gt_index)
    for gt in gt_index:
        count=0
        for det in det_index:
            iou_score=iou(detections[det]['bbox'],ground_truth[gt]['bbox'])
            if(iou_score>=0.5):
                count+=1
        if count ==0:
            FN+=1
        elif count>1:
            TP+=1
            FP+=(count-1)
        else:
            TP+=1
    matches=TP+FP
    if(len(det_index)!=matches):
        FP+=len(det_index)-matches
    print((TP,FP,FN))
    return TP,FP,FN
                
def plotResults(opath,precision,recall,f1,mod):
    plt.figure()
    plt.plot(recall,label="Recall")
    plt.plot(precision,label="Precision")
    plt.plot(f1,label="F1 score")
    plt.legend()
    if mod=='average':
        plt.title("Average Results")
        plt.savefig(opath+'/AvgResults.png')        
    elif mod=='partsAvg':
        plt.title("isPartOf Average Results")
        plt.savefig(opath+'/PartOfAvgResults.png')

def plotPR_curve_byClass(opath,metrics):
    if path.exists(opath+'/ResultsByClass')==False:
        os.mkdir(opath+'/ResultsByClass')
    for key in metrics:
        plt.figure()
        pr=[]
        rec=[]
        f1=[]
        TP,FP,FN=0,0,0
        for j in range(len(metrics[key]['TP'])):
            TP+=metrics[key]['TP'][j]
            FP+=metrics[key]['FP'][j]
            FN+=metrics[key]['FN'][j]
            if FP+TP==0:
                p=0
            else:
                p=TP/(TP+FP)
            pr.append(p)
            if TP+FN==0:
                r=0
                
            else:   
                r=TP/(TP+FN)
            rec.append(r)
            if p+r==0:
                f1.append(0)
            else:
                f1.append(2*((p*r)/(p+r)))
        # if key=='Eye':
        #     print(pr)
        #     print(rec)
        #     print(f1)
        #     break
        plt.title(key)
        plt.plot(pr,label="Precsion")
        plt.plot(rec,label="Recall")
        plt.plot(f1,label="F1 score")
        plt.legend()
        plt.savefig(opath+'/ResultsByClass/'+key+'.png')
        

def match_bboxes(gt,inf,labels,threshold):
 matchings={}
 for img in gt: 
    matchings[img]={}
    for model in inf[img]:
        avail_boxes=list(model)
        box_pairs=[]
        for box in gt[img]:  
            if(len(avail_boxes)>0):
                index,avail_boxes=find_respect_box(gt[img][box],avail_boxes,model,labels=labels,threshold=threshold)
                if index != None:
                    matchings[img][box]=index
 return matchings




def calcPartmetrics(detections,model,matchings,gt,rel_type):
    TP,FP,FN=0,0,0
    for part in model[rel_type]:
        key=[i for i in matchings if matchings[i]==part]
        if detections[part]["label"]=="Other":		
               continue

        if len(key)>0:
            flag=0
            for k in key:
                if k in gt['parts']:
                    flag+=1
                
            if flag>0:
                TP+=1
            else:
                FP+=1
        else:
            FP+=1
                    
    if(TP<len(gt['parts'])):
        FN+=len(gt['parts'])-TP
   
    return TP,FP,FN


def evaluate_partOf(opath,ofile,gt,inf,labels,rel_type,threshold):
    matchings=match_bboxes(gt,inf,labels,threshold=threshold)
    totalTP,totalFP,totalFN=0,0,0
    totalPr,totalRec,totalF1=[],[],[]
    avgPrecision,avgRecall,avgF1=[],[],[]
    with open(opath+'/'+ofile, "w") as txtfile:
        for img in gt:
            txtfile.write(img+"\n")
            txtfile.write(str(matchings[img])+"\n")

            cumFP,cumTP,cumFN=0,0,0
            for model in inf[img]:
                for box in gt[img]:
                    if 'parts' not in gt[img][box]: 
                        continue 
                
                    if box in matchings[img]:
                        inf_index=matchings[img][box]
                    else:
                        
                        cumFN+=len(gt[img][box]['parts'])
                        continue
                        
                    if len(model[inf_index][rel_type])<1:
                        cumFN+=len(gt[img][box]['parts'])
                        continue
                    
                    TP,FP,FN=calcPartmetrics(inf[img][0],model[inf_index],matchings[img],gt[img][box],rel_type)
                    cumTP+=TP
                    cumFP+=FP
                    cumFN+=FN
                
                    
               
                ignore_labels=["Boat","Other","Chair","Sofa","Diningtable"]
                for ibox in model:
                    key=None
                    if(model[ibox]["label"] in ignore_labels):
                        continue
                    if len(model[ibox][rel_type])>0:
                    
                        key=[i for i in matchings[img] if matchings[img][i]==ibox]
                        if key==None:
                            cumFP+=len(model[ibox][rel_type])
                            continue
                        flag=0
                        for k in key:
                            if 'parts' in gt[img][k]:
                                flag=1

                        if flag!=1:
                            cumFP+=len(model[ibox][rel_type])
                            continue
                break
            
            totalFN+=cumFN
            totalFP+=cumFP
            totalTP+=cumTP
            txtfile.write('True positives='+str(cumTP)+"\n")
            txtfile.write('False positives='+str(cumFP)+"\n")
            txtfile.write('False negatives='+str(cumFN)+"\n")
            if cumTP+cumFP==0:
                precision=0
            else:
                precision=cumTP/(cumTP+cumFP)
            if cumTP+cumFN==0:
                recall=0
            else:
                recall=cumTP/(cumTP+cumFN)
            if precision==recall==0:
                f1=0
            else:
                f1=2*((precision*recall)/(recall+precision))
            

            totalPr.append(precision)
            totalRec.append(recall)
            totalF1.append(f1)
            avgPrecision.append(sum(totalPr)/len(totalPr))
            avgRecall.append(sum(totalRec)/len(totalRec))
            avgF1.append(sum(totalF1)/len(totalF1))
        
    print(totalTP,",",totalFP,",",totalFN)
    with open(opath+'/'+ofile, "a") as txtfile:
        txtfile.write("-------isPartOf Evaluation Scores------- \n ")
        txtfile.write("IoU Threshold="+str(threshold)+'\n')
        txtfile.write("Total TP, FP, FN="+str(totalTP)+" "+str(totalFP)+" "+str(totalFN)+"\n")
        txtfile.write('Average Precision='+str(avgPrecision[len(avgPrecision)-1])+"\n")
        txtfile.write('Average Recall='+str(avgRecall[len(avgRecall)-1])+"\n")
        txtfile.write('Average F1 score='+str(avgF1[len(avgF1)-1])+"\n")
        txtfile.write('\n')
   


def count_partOf(gt):
    sum=0
    for img in gt:
        for key in gt[img]:
            if "parts" in gt[img][key]:
                sum+=len(gt[img][key]['parts'])
    return sum

def count_bboxes(gt):
    sum=0
    for img in gt:
        sum+=len(gt[img])
    return sum
    

def baseline(output):
    if path.exists(output)==False:
        os.mkdir(output)
    opath=output+'baseline'
    if path.exists(opath)==False:
        os.mkdir(opath) 
    ground_truth=getJSONdata(output+"JSON_annotations.json")
    ## {images[models{index{ground_truth}}]}
    inference_res=getJSONdata(opath+"/BL_inferences_JSON.json")
    evaluate_partOf(opath,"BL_PartOfresults.txt",ground_truth,inference_res,labels=True,rel_type='Part',threshold=0.5)
    print("Total partOf relations in Ground Truth=",count_partOf(ground_truth))

##evaluation process with configurable iou value for the parts
def baseline1_iou(output,iou_v):
    if path.exists(output)==False:
        os.mkdir(output)
    opath=output+'baseline'
    if path.exists(opath)==False:
        os.mkdir(opath) 
        
    ground_truth=getJSONdata(output+"JSON_annotations.json")
    ## {images[models{index{ground_truth}}]}
    inference_res=getJSONdata(opath+"/BL_inferences_JSON.json")

    evaluate_partOf(opath,"BL_PartOfresults_PartIoU_"+str(iou_v)+".txt",ground_truth,inference_res,labels=True,rel_type='Part',threshold=iou_v)
    # print("Total partOf relations in Ground Truth=",count_partOf(ground_truth))

def groundtruth1(output):
    if path.exists(output)==False:
        os.mkdir(output)
    opath=output+'groundtruth1'
    if path.exists(opath)==False:
        os.mkdir(opath)   

    ground_truth=getJSONdata(output+"JSON_annotations.json")
    ## {images[models{index{ground_truth}}]}
    inference_res=getJSONdata(opath+"/GT1_inferences_JSON.json")
    evaluate_partOf(opath,"GT1_PartOfresults.txt",ground_truth,inference_res,labels=True,rel_type='Part',threshold=0.5)
    # print("Total partOf relations in Ground Truth=",count_partOf(ground_truth))

##evaluation process with configurable iou value for the parts
def groundtruth1_iou(output,iou_v):
    if path.exists(output)==False:
        os.mkdir(output)
    opath=output+'groundtruth1'
    if path.exists(opath)==False:
        os.mkdir(opath)   

    ground_truth=getJSONdata(output+"JSON_annotations.json")
    ## {images[models{index{ground_truth}}]}
    inference_res=getJSONdata(opath+"/GT1_inferences_JSON.json")
    evaluate_partOf(opath,"GT1_PartOfresults_"+str(iou_v)+".txt",ground_truth,inference_res,labels=True,rel_type='Part',threshold=iou_v)
    # print("Total partOf relations in Ground Truth=",count_partOf(ground_truth))

def groundtruth2(output):
    if path.exists(output)==False:
        os.mkdir(output)
    opath=output+'groundtruth2'
    if path.exists(opath)==False:
        os.mkdir(opath)   

    ground_truth=getJSONdata(output+"JSON_annotations.json")
    ## {images[models{index{ground_truth}}]}
    inference_res=getJSONdata(opath+"/GT2_inferences_JSON.json")
    evaluate_partOf(opath,"GT2_PartOfresults.txt",ground_truth,inference_res,labels=True,rel_type='Part',threshold=0.5)
    print("Total partOf relations in Ground Truth=",count_partOf(ground_truth))

##evaluation process with configurable iou value for the parts
def new_groundtruth2(output):
    if path.exists(output)==False:
        os.mkdir(output)
    opath=output+'groundtruth2'
    if path.exists(opath)==False:
        os.mkdir(opath)   

    ground_truth=getJSONdata(output+"new_groundtruth.json")
    ## {images[models{index{ground_truth}}]}
    inference_res=getJSONdata(opath+"/new_groundtruth2.json")
    evaluate_partOf(opath,"newGT2_PartOfresults.txt",ground_truth,inference_res,labels=True,rel_type='Part',threshold=0.5)
    print("Total partOf relations in Ground Truth=",count_partOf(ground_truth))


def groundtruth2_iou(output,iou_v):
    if path.exists(output)==False:
        os.mkdir(output)
    opath=output+'groundtruth2'
    if path.exists(opath)==False:
        os.mkdir(opath)   

    ground_truth=getJSONdata(output+"JSON_annotations.json")
    ## {images[models{index{ground_truth}}]}
    inference_res=getJSONdata(opath+"/GT2_inferences_JSON.json")
    evaluate_partOf(opath,"GT2_PartOfresults"+str(iou_v)+".txt",ground_truth,inference_res,labels=True,rel_type='Part',threshold=iou_v)
    # print("Total partOf relations in Ground Truth=",count_partOf(ground_truth))


def predeck(output):
    if path.exists(output)==False:
        os.mkdir(output)
    opath=output+'predeck'
    if path.exists(opath)==False:
        os.mkdir(opath)   

    ground_truth=getJSONdata(output+"JSON_annotations.json")
    ## {images[models{index{ground_truth}}]}
    inference_res=getJSONdata(opath+"/predeck_inferences_JSON.json")
    evaluate_partOf(opath,"predeck_PartOfresults.txt",ground_truth,inference_res,labels=True,rel_type='Part',threshold=0.5)
    print("Total partOf relations in Ground Truth=",count_partOf(ground_truth))

def predeck_iou(output,iou_v):
    if path.exists(output)==False:
        os.mkdir(output)
    opath=output+'predeck'
    if path.exists(opath)==False:
        os.mkdir(opath)   

    ground_truth=getJSONdata(output+"JSON_annotations.json")
    ## {images[models{index{ground_truth}}]}
    inference_res=getJSONdata(opath+"/predeck_inferences_JSON.json")
    evaluate_partOf(opath,"predeck_PartOfresults"+str(iou_v)+".txt",ground_truth,inference_res,labels=True,rel_type='Part',threshold=iou_v)
    # print("Total partOf relations in Ground Truth=",count_partOf(ground_truth))


def conceptnet2(output):
    if path.exists(output)==False:
        os.mkdir(output)
    opath=output+'conceptnet2'
    if path.exists(opath)==False:
        os.mkdir(opath)   

    ground_truth=getJSONdata(output+"JSON_annotations.json")
    ## {images[models{index{ground_truth}}]}
    inference_res=getJSONdata(opath+"/CN2_inferences_JSON.json")
    evaluate_partOf(opath,"CN2_PartOfresults.txt",ground_truth,inference_res,labels=True,rel_type='Part',threshold=0.5)
    print("Total partOf relations in Ground Truth=",count_partOf(ground_truth))

def conceptnet2_iou(output,iou_v):
    if path.exists(output)==False:
        os.mkdir(output)
    opath=output+'conceptnet2'
    if path.exists(opath)==False:
        os.mkdir(opath)   

    ground_truth=getJSONdata(output+"JSON_annotations.json")
    ## {images[models{index{ground_truth}}]}
    inference_res=getJSONdata(opath+"/CN2_inferences_JSON.json")
    evaluate_partOf(opath,"CN2_PartOfresults"+str(iou_v)+".txt",ground_truth,inference_res,labels=True,rel_type='Part',threshold=iou_v)
    # print("Total partOf relations in Ground Truth=",count_partOf(ground_truth))

def conceptnet1(output):
    if path.exists(output)==False:
        os.mkdir(output)
    opath=output+'conceptnet1'
    if path.exists(opath)==False:
        os.mkdir(opath)   
    ground_truth=getJSONdata(output+"JSON_annotations.json")
    ## {images[models{index{ground_truth}}]}
    inference_res=getJSONdata(opath+"/CN1_inferences_JSON.json")
    evaluate_partOf(opath,"CN1_PartOfresults.txt",ground_truth,inference_res,labels=True,rel_type='Part',threshold=0.5)
    # print("Total partOf relations in Ground Truth=",count_partOf(ground_truth))


def conceptnet_iou(output,iou_v):
    if path.exists(output)==False:
        os.mkdir(output)
    opath=output+'conceptnet1'
    if path.exists(opath)==False:
        os.mkdir(opath)   
    ground_truth=getJSONdata(output+"JSON_annotations.json")
    ## {images[models{index{ground_truth}}]}
    inference_res=getJSONdata(opath+"/CN1_inferences_JSON.json")
    evaluate_partOf(opath,"CN1_PartOfresults"+str(iou_v)+".txt",ground_truth,inference_res,labels=True,rel_type='Part',threshold=iou_v)
    # print("Total partOf relations in Ground Truth=",count_partOf(ground_truth))


def noisy_conceptnet_iou(output,iou_v):
    if path.exists(output)==False:
        os.mkdir(output)
    opath=output+'noisy_conceptnet'
    if path.exists(opath)==False:
        os.mkdir(opath)   

    ground_truth=getJSONdata(output+"JSON_annotations.json")
    ## {images[models{index{ground_truth}}]}
    inference_res=getJSONdata(opath+"/NCN_inferences_JSON.json")
    evaluate_partOf(opath,"NCN_PartOfresults"+str(iou_v)+".txt",ground_truth,inference_res,labels=True,rel_type='Spatial_Part',threshold=iou_v)
    # print("Total partOf relations in Ground Truth=",count_partOf(ground_truth))

def noisy_conceptnet(output):
    if path.exists(output)==False:
        os.mkdir(output)
    opath=output+'noisy_conceptnet'
    if path.exists(opath)==False:
        os.mkdir(opath)   

    ground_truth=getJSONdata(output+"JSON_annotations.json")
    ## {images[models{index{ground_truth}}]}
    inference_res=getJSONdata(opath+"/NCN_inferences_JSON.json")
    evaluate_partOf(opath,"NCN_PartOfresults"+str(0.5)+".txt",ground_truth,inference_res,labels=True,rel_type='Spatial_Part',threshold=0.5)
    # print("Total partOf relations in Ground Truth=",count_partOf(ground_truth))