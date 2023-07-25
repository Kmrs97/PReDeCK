import torch
from torch.autograd import Variable
from examples.yolov5_connection.yolo.utils.dataloaders import LoadImages
from examples.yolov5_connection.yolo.utils.datasets import ImageFolder
from examples.yolov5_connection.yolo.utils.general import non_max_suppression
# from yolo.models import Darknet

def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names



def termPath2dataList(termPath, img_size, domain,model,dataset_name):
    """
    @param termPath: a string of the form 'term path' denoting the path to the files represented by term
    """
    factsList = []
    dataList = []
    # # Load Yolo network, which is used to generate the facts for bounding boxes and a tensor for each bounding box
    # config_path = './yolo/yolov3.cfg'
    # weights_path = './yolo/yolov3.weights'
    # yolo = Darknet(config_path, img_size)
    # yolo.load_weights(weights_path)
    # yolo.eval()

    # # feed each image into yolo
    
    term, path = termPath.split(' ')
    dataset= ImageFolder(path, img_size=img_size)
    # print(dataset)
    dataloader = torch.utils.data.DataLoader(dataset, 
        batch_size=1, 
        shuffle=False
    )
    # print(dataloader)
    images, labels = next(iter(dataloader))
    # print(images)
    model = torch.hub.load('ultralytics/yolov5', 'custom',path=model)   # or yolov5n - yolov5x6, custom
    image_ids=[]
    for ipath, img in dataloader:
        filename=ipath[0].replace(path,"")
        image_ids.append(filename[:6])
    # Images
    # img = '/home/kmrs97/Downloads/Documents/100760_000001.jpg'  # or file, Path, PIL, OpenCV, numpy, list
        ##NOTE
        #They have it in their code for some reason 
        #Doesnt affect anything when I commend it
        img = Variable(img.type(torch.FloatTensor))
        with torch.no_grad():
            output = model(img)     
            facts, dataDic = postProcessing(output, term, domain,dataset_name=dataset_name)
            factsList.append(facts)
            dataList.append(dataDic)        
    return image_ids,factsList, dataList



def BboxesAndConfidences(det,cls_name,printAll=False):
    
    if printAll:
        count=1
        for d in det:
            print("Bounding Box",count,":")
            count+=1
            print("Coordinates=",end=" ")
            for i in range(4):
                print(int(d[i]),end =" ")
            print()
            print("Confidences=",end=" ")
            conf=[]
            for i in range(4,len(d)):
                conf.append(float(d[i]))
                print(float(d[i]),end =" ")
            print()
            maxval=max(conf)
            print("Max Confidence: ",cls_name[conf.index(maxval)],maxval)
            print()

    confidences=[]
    coordinates=[]
    max_confs=[]
    for d in det:
        cords=[]
        for i in range(4):
            cords.append(d[i])
        coordinates.append(cords)
        conf=[]
        for i in range(4,len(d)):
            conf.append(float(d[i]))
        confidences.append(conf)
        maxval=max(conf)
        max_confs.append(maxval)
    
    return coordinates,confidences,max_confs
    



def postProcessing(output, term, domain, num_classes=60, conf_thres=0.25, iou_thres=0.45,dataset_name=None):
    facts = ''
    dataDic = {}
    if dataset_name=="pascal":
        cls_name = load_classes('./yolo/pascal_part.names')
    else:
        cls_name = load_classes('./yolo/coco.names')
    detections,detcomplete = non_max_suppression(output, conf_thres, iou_thres,multi_label=True)
    # print("DET: ",detections)
    # print("DET_complete: ",detcomplete)
    if detcomplete:
        for det in detcomplete:
            coordinates,confidences,max_confs=BboxesAndConfidences(det,cls_name,printAll=False)
            for idx, (x1, y1, x2, y2) in enumerate(coordinates):
                terms = '{},b{}'.format(term, idx)
                facts += 'box({}, {}, {}, {}, {},{}).\n'.format(terms, int(x1), int(y1), int(x2), int(y2),int(max_confs[idx]*1000))
                # print(idx,(int(x1), int(y1), int(x2), int(y2)))
                X = torch.zeros([1, len(domain)], dtype=torch.float64)
                for i in range(len(confidences[idx])):
                    # print(cls_name[i],"with ",confidences[idx][i])
                    className = '{}'.format(cls_name[i])
                    # print("class: ",className)
                    if className in domain:
                        X[0, domain.index(className)] += round(float(confidences[idx][i]), 3)
                    else:
                        X[0, -1] += round(float(confidences[idx][i]), 3)
                # print(X)
                dataDic[terms] = X
                # print(dataDic[terms])

    # if detections:
    #     for detection in detections:
    #         for idx, (x1, y1, x2, y2,cls_conf,cls_pred) in enumerate(detection):
    #             #print((int(x1), int(y1), int(x2), int(y2), cls_conf,cls_name[int(cls_pred)]))
    #             # print()
    #             terms = '{},b{}'.format(term, idx)
    #             facts += 'box({}, {}, {}, {}, {}).\n'.format(terms, int(x1), int(y1), int(x2), int(y2))
               
    #             className = '{}'.format(cls_name[int(cls_pred)])
    #             # print("class: ",className)
    #             X = torch.zeros([1, len(domain)], dtype=torch.float64)
    #             if className in domain:
    #                 X[0, domain.index(className)] += round(float(cls_conf), 3)
    #             else:
    #                 X[0, -1] += round(float(cls_conf), 3)
    #             dataDic[terms] = X
    #             print(dataDic[terms])
    return facts, dataDic


