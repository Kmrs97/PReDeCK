import requests
import json
from py2neo import  Graph
import label_CN_assosiacion as laas

## The following script is used to construct the local subgraph using the infomation retrieved from ConceptNet


def pruneRelationNodes(graph,relation): 
    resp=graph.run("MATCH (n)-[r:"+relation+"]->(m) RETURN n,m").data()
    count=0
    for pair in resp:
        res=graph.run("MATCH"+str(pair['n'])+"-[r]->"+str(pair['m'])+"RETURN COUNT(r) AS count").data()
        if(res[0]['count']>1):
            count+=1
            graph.run("MATCH"+str(pair['n'])+"-[r:"+relation+"]->"+str(pair['m'])+" DELETE r")
    print("Deleted ",count," edges!")
     
def retNodeExist(graph,label,category):
	type=["Object","Subject"]
	for t in type:
		query="MATCH (u:"+t+"{label:'"+label.upper().replace("'","_").replace(" ","_")+"', category:'"+category.upper()+"'}) \n  RETURN u.label \n"
		val=graph.run(query).data()
		if(len(val)>0):
			return [t,val[0]['u.label']]	
	
	return 0

def updateGraph_Subjects(graph,relations,subjects,obj,category):
	for i in range(len(relations)):
		rel=relations[i]
		sub=subjects[i]
		right_b="("
		left_b=")"
		replace_empty=['"',"*",".",",","%",';','\\','!','@','#','$','%','^','&','*','[',']','{','}',':','<','>','?','|','`','~']
		replace_underscore=[right_b,left_b," ","+","-",'/',"'","’",'=']
		for x in replace_empty:
			obj=obj.replace(x,"")
			sub=sub.replace(x,"")
		for x in replace_underscore:
			obj=obj.replace(x,"_")
			sub=sub.replace(x,"_")
		if(sub[0]=="_"):
			sub=sub[1:]
		print(sub)
		node_obj=retNodeExist(graph,obj,category)
		try:
			if(node_obj==0):
				graph.run("CREATE (tag_"+obj+":Object{label:'"+obj.upper()+"', category:'"+category.upper()+"'}) \n")
				node_sub=retNodeExist(graph,sub,category)
				if(node_sub==0):
					graph.run("CREATE (sub_"+sub+":Subject{label:'"+sub.upper()+"', category:'"+category.upper()+"'}) \n")
					graph.run("MATCH (o:Object) MATCH(s:Subject) WHERE o.label='"+obj.upper()+"'AND s.label='"+sub.upper()+"'AND o.category=s.category MERGE (o)-[:"+rel.upper()+"]->(s) RETURN * \n")
				else:
					graph.run("MATCH (o:Object) MATCH(s:"+node_sub[0]+") WHERE o.label='"+obj.upper()+"'AND s.label='"+node_sub[1].replace("'","_")+"' AND o.category=s.category MERGE (o)-[:"+rel.upper()+"]->(s) RETURN * \n")
			else:
				node_sub=retNodeExist(graph,sub,category)
				if(node_sub==0):
					graph.run("CREATE (sub_"+sub+":Subject{label:'"+sub.upper()+"', category:'"+category.upper()+"'}) \n")
					graph.run("MATCH (o:"+node_obj[0]+") MATCH(s:Subject) WHERE o.label='"+node_obj[1].replace("'","_")+"'AND s.label='"+sub.upper()+"' AND o.category=s.category MERGE (o)-[:"+rel.upper()+"]->(s) RETURN * \n")
				else:
					graph.run("MATCH (o:"+node_obj[0]+") MATCH(s:"+node_sub[0]+") WHERE o.label='"+node_obj[1].replace("'","_")+"'AND s.label='"+node_sub[1].replace("'","_")+"' AND o.category=s.category MERGE (o)-[:"+rel.upper()+"]->(s) RETURN * \n")	
		except:
			continue	

def updateGraph_Objects(graph,relations,objects,sub,category):
	for i in range(len(relations)):
		rel=relations[i]
		obj=objects[i]
		right_b="("
		left_b=")"
		replace_empty=['"',"*",".",",","%",';','\\','!','@','#','$','%','^','&','*','[',']','{','}',':','<','>','?','|','`','~']
		replace_underscore=[right_b,left_b," ","+","-",'/',"'","’",'=']
		for x in replace_empty:
			obj=obj.replace(x,"")
			sub=sub.replace(x,"")
		for x in replace_underscore:
			obj=obj.replace(x,"_")
			sub=sub.replace(x,"_")
		if(sub[0]=="_"):
			sub=sub[1:]
		print(obj)
		node_obj=retNodeExist(graph,obj,category)
		try:
			if(node_obj==0):
				graph.run("CREATE (tag_"+obj+":Object{label:'"+obj.upper()+"', category:'"+category.upper()+"'}) \n")
				node_sub=retNodeExist(graph,sub,category)
				if(node_sub==0):
					graph.run("CREATE (sub_"+sub+":Subject{label:'"+sub.upper()+"', category:'"+category.upper()+"'}) \n")
					graph.run("MATCH (o:Object) MATCH(s:Subject) WHERE o.label='"+obj.upper()+"'AND s.label='"+sub.upper()+"'AND o.category=s.category MERGE (o)-[:"+rel.upper()+"]->(s) RETURN * \n")
				else:
					graph.run("MATCH (o:Object) MATCH(s:"+node_sub[0]+") WHERE o.label='"+obj.upper()+"'AND s.label='"+node_sub[1].replace("'","_")+"' AND o.category=s.category MERGE (o)-[:"+rel.upper()+"]->(s) RETURN * \n")
			else:
				node_sub=retNodeExist(graph,sub,category)
				if(node_sub==0):
					graph.run("CREATE (sub_"+sub+":Subject{label:'"+sub.upper()+"', category:'"+category.upper()+"'}) \n")
					graph.run("MATCH (o:"+node_obj[0]+") MATCH(s:Subject) WHERE o.label='"+node_obj[1].replace("'","_")+"'AND s.label='"+sub.upper()+"' AND o.category=s.category MERGE (o)-[:"+rel.upper()+"]->(s) RETURN * \n")
				else:
					graph.run("MATCH (o:"+node_obj[0]+") MATCH(s:"+node_sub[0]+") WHERE o.label='"+node_obj[1].replace("'","_")+"'AND s.label='"+node_sub[1].replace("'","_")+"' AND o.category=s.category MERGE (o)-[:"+rel.upper()+"]->(s) RETURN * \n")	
		except:
			continue



def get_infoByConceptNet(graph,sample_tags,category):
	for tag in sample_tags:
		if(retNodeExist(graph,tag,category)==0):
			graph.run("CREATE (tag_"+tag.replace(" ","_")+":Object{label:'"+tag.upper()+"' , category:'"+category.upper()+"'}) \n")
	for tag in sample_tags:
		print("----------------------OBJECT:--------------",tag)
		relations,subjects=laas.getAssociations_Subjects(tag,category)
		updateGraph_Subjects(graph,relations,subjects,tag,category)
		relations,objects=laas.getAssociations_Objects(tag,category)
		updateGraph_Objects(graph,relations,objects,tag,category)
		
		#Get hop1 nodes information
		for sub in subjects:
			hop1_relations,hop1_subjects=laas.getAssociations_Subjects(sub,category)
			updateGraph_Subjects(graph,hop1_relations,hop1_subjects,sub,category)
			hop1_relations,hop1_objects=laas.getAssociations_Objects(sub,category)
			updateGraph_Objects(graph,hop1_relations,hop1_objects,sub,category)

		for obj in objects:
			hop1_relations,hop1_subjects=laas.getAssociations_Subjects(obj,category)
			updateGraph_Subjects(graph,hop1_relations,hop1_subjects,obj,category)
			hop1_relations,hop1_objects=laas.getAssociations_Objects(obj,category)
			updateGraph_Objects(graph,hop1_relations,hop1_objects,obj,category)
		
def main():
	#Complete the credentials
	USERNAME = "" 
	PASSWORD = "" 
	graph = Graph(scheme="bolt", host="localhost", port=YOURPORT, auth=(USERNAME, PASSWORD)) 

	animals=["wing","beak","bird","cat","cow","dog","foot","hoof","horn","horse","muzzle",
	"neck","leg","saddle","sheep","tail","torso","head","eye","nose","ear"]
	body=["body","ear","eyebrow","eye","foot","hair","hand","head","leg","mouth","neck",
	"nose","torso","arm","person"]
	artifacts=["wing","sprocket","aeroplane","bicycle","boat","bodywork","bottle","bus","cap",
	"car","chair","coach","door","engine","handlebar","headlight","locomotive","mirror",
	"motorbike","pot","saddle","screen","sofa","stern","tail","train","tv monitor","wheel"
	,"window","dining table","body","tail","license plate"]
	person=["arm","person"]
	plant=["plant","potted plant"]


	
	get_infoByConceptNet(graph,animals,"animal")
	get_infoByConceptNet(graph,body,"body")
	get_infoByConceptNet(graph,artifacts,"artifact")
	get_infoByConceptNet(graph,person,"person")
	get_infoByConceptNet(graph,plant,"plant")
	
main()


