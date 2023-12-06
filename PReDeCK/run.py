##run pipeline
import yaml
import parse_dataset as prsds
import evaluate_results as evr


import baseline as bl
import predeck_gt1 as pgt1
import predeck_gt2 as pgt2
import predeck as prd
import predeck_cn1 as pcn1
import predeck_cn2 as pcn2
import predeck_cnnoisy as pcn_noisy

##Error Detection
import predeck_ed_GT as ped_gt
import predeck_ed_CN as ped_cn


import coco_pre_detections as coco



class System:
    img_path=None
    out_path=None
    model=None
    create_gtJSON=False
    inference=None
    dup_drop=None
    evaluation=None
    img_size=None
    experiments=[]
    exp_list=[0,evr.baseline,evr.groundtruth1,evr.groundtruth2,evr.predeck,evr.conceptnet1,evr.conceptnet2,evr.noisy_conceptnet] 
    inf_list=[0,bl.infer,pgt1.infer,pgt2.infer,prd.infer,pcn1.infer,pcn2.infer,pcn_noisy.infer,ped_gt.infer,ped_cn.infer,coco.infer]

    def getConfigurations(self): 
        with open('config.yaml', 'r') as file:
            confs = yaml.safe_load(file)
        self.img_path=confs['img_path']
        self.img_size=confs['img_size']
        self.model=confs['model']
        self.out_path=confs['output_path']
        self.create_gtJSON=confs['gt_file']
        self.experiments=confs['evaluation_experiments']
        self.inference=confs['inference']
        self.evaluation=confs['evaluation']
        self.dup_drop=confs['duplicates_drop']
    
         
    def createGTfile(self):
        if self.create_gtJSON:
            prsds.prepareTestFiles(self.img_size,self.img_path,self.out_path)

    def infer(self):
        if self.inference==False:
            return 
        for i in self.experiments:
            if i < len(self.inf_list):
                self.inf_list[i](self.img_path,self.img_size,self.model,self.out_path)
                # if self.dup_drop==True:
                #     drop_dups(self.out_path+"/experiment"+i,i)
                    
    
    def eval(self):
        if self.evaluation==False:
            return
        for i in self.experiments:
            if i < len(self.exp_list):
                self.exp_list[i](self.out_path)
        return
    
s1=System()
s1.getConfigurations()
s1.createGTfile()
s1.infer()
s1.drop_duplicates()
s1.eval()
