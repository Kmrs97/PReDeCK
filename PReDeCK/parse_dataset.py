from rdflib import Graph, Literal, RDF, URIRef
from rdflib.namespace import FOAF , XSD, RDF, OWL
import pdb
import csv
import os
from os import path
import xmltodict
import xml.etree.ElementTree as ET
from rdflib import Namespace
import numpy as np
import json
import os
from tkinter.filedialog import askdirectory

def extract_bb_coordinates(id, ann_dict):
    if id > -1:
        x_1 = int(ann_dict[id]['polygon']['pt'][0]['x'])
        y_1 = int(ann_dict[id]['polygon']['pt'][0]['y'])
        x_2 = int(ann_dict[id]['polygon']['pt'][1]['x'])
        y_2 = int(ann_dict[id]['polygon']['pt'][1]['y'])
        x_3 = int(ann_dict[id]['polygon']['pt'][2]['x'])
        y_3 = int(ann_dict[id]['polygon']['pt'][2]['y'])
        x_4 = int(ann_dict[id]['polygon']['pt'][3]['x'])
        y_4 = int(ann_dict[id]['polygon']['pt'][3]['y'])
    else:
        x_1 = int(ann_dict['polygon']['pt'][0]['x'])
        y_1 = int(ann_dict['polygon']['pt'][0]['y'])
        x_2 = int(ann_dict['polygon']['pt'][1]['x'])
        y_2 = int(ann_dict['polygon']['pt'][1]['y'])
        x_3 = int(ann_dict['polygon']['pt'][2]['x'])
        y_3 = int(ann_dict['polygon']['pt'][2]['y'])
        x_4 = int(ann_dict['polygon']['pt'][3]['x'])
        y_4 = int(ann_dict['polygon']['pt'][3]['y'])
    xmin = np.min([x_1, x_2, x_3, x_4])
    ymin = np.min([y_1, y_2, y_3, y_4])
    xmax = np.max([x_1, x_2, x_3, x_4])
    ymax = np.max([y_1, y_2, y_3, y_4])
    return xmin, ymin, xmax, ymax



def create_txtFile(split,img_path):
  # dir_path =askdirectory()
  # print(dir_path)
  path_pascal = 'semanticPascalPart'
  # print(os.path.join(path_pascal,"test.txt"))
  print("Creating text file containing all the Test images names...")
  with open(os.path.join(path_pascal,split+".txt"),"w") as file:
    for filename in os.listdir(img_path):
      if filename.__contains__(".jpg"):
        # print(filename.replace(".jpg",".xml"))
        file.write(filename.replace(".jpg",".xml"))
        file.write("\n")
  #  with open(os.path.join(os.getcwd(), filename), 'r') as f:

  # for filename in os.listdir(os.getcwd()):
  #   with open(os.path.join(os.getcwd(), filename), 'r') as f:

class PASCALPArt_annotations:
  def __init__(self):
    self.split = ""

    # this is just an annotation example useful for your own "fun"
    self.annotations = {
      "00001": {
        "1": {
          "class": "Person",
          "x_1": 1,
          "y_1": 1,
          "x_2": 123,
          "y_2": 124,
          "isPartOf": ""
        },
        "2": {
          "class": "Leg",
          "x_1": 23,
          "y_1": 23,
          "x_2": 44,
          "y_2": 44,
          "isPartOf": "1"
        },
        "3": {
          "class": "Body",
          "x_1": 28,
          "y_1": 321,
          "x_2": 312,
          "y_2": 932,
          "isPartOf": "1"
        }
      },
      "00002": {
        "1": {
          "class": "Horse",
          "x_1": 1,
          "y_1": 1,
          "x_2": 123,
          "y_2": 124,
          "isPartOf": ""
        },
        "2": {
          "class": "Muzzle",
          "x_1": 23,
          "y_1": 23,
          "x_2": 44,
          "y_2": 44,
          "isPartOf": "1"
        },
        "3": {
          "class": "Tail",
          "x_1": 28,
          "y_1": 321,
          "x_2": 312,
          "y_2": 932,
          "isPartOf": "1"
        }
      }
    }
    self.annotations = {}


  def get_parts_ids(self, filename, obj_id):
    try:
      part_ids = self.annotations[filename][obj_id]["hasParts"]
      #pdb.set_trace()
      if part_ids == "":
        #print(f"Object {obj_id} in image {filename} does not have parts.")
        return None
      else:
        return part_ids.split(",")
    except KeyError:
      print(f"Annotation file {filename} or object id {obj_id} do not exist.")


  def get_whole_ids(self, filename, obj_id):
    try:
      part_id = self.annotations[filename][obj_id]["isPartOf"]
      if part_id == "":
        #print(f"Object {obj_id} in image {filename} is a whole object.")
        return None
      else:
        return part_id
    except KeyError:
      print(f"Annotation file {filename} or object id {obj_id} do not exist.")


  def get_objects(self, filename):
    try:
      return self.annotations[filename]
    except KeyError:
      print(f"Annotation file {filename} does not exist.")


  def get_obj_class(self, filename, obj_id):
    try:
      return self.annotations[filename][obj_id]["class"]
    except KeyError:
      print(f"Annotation file {filename} or object id {obj_id} do not exist.")


  def get_BB(self, filename, obj_id):
    try:
      x_1 = self.annotations[filename][obj_id]["x_1"]
      y_1 = self.annotations[filename][obj_id]["y_1"]
      x_2 = self.annotations[filename][obj_id]["x_2"]
      y_2 = self.annotations[filename][obj_id]["y_2"]
      return [x_1, y_1, x_2, y_2]
    except KeyError:
      print(f"Annotation file {filename} or object id {obj_id} do not exist.")




  def load_data(self, split,img_size):
    assert split == "test" or split == "trainval" or split=="all", "split should be 'test' or 'trainval'"
    self.annotations = {}
    self.split = split
    print(f"Parsing {split} set ...")
    path_pascal = 'semanticPascalPart' # here put the path where you have the dataset

    with open(os.path.join(path_pascal, split + ".txt")) as csv_file:
        csv_reader = csv.reader(csv_file)
        for image_name in csv_reader:
            if(image_name[0].__contains__(".rf.")):
              image_name=str(image_name[0][:6])+".jpg"
            #print(f"Processing annotation file {image_name[0]}")
              tree = ET.parse(os.path.join(path_pascal, f"Annotations_{split}", image_name.split(".")[0] + ".xml"))
            else:
              tree = ET.parse(os.path.join(path_pascal, f"Annotations_{split}", image_name[0].split(".")[0] + ".xml"))

            xml_data = tree.getroot()
            xmlstr = ET.tostring(xml_data, encoding='utf8', method='xml')
            data_dict = dict(xmltodict.parse(xmlstr))
            filename = data_dict['annotation']['filename'].split(".")[0]
            self.annotations[filename] = {}

            #get image size and use the resized width and height to do the scaling on the bboxes
            width=int(data_dict['annotation']['imagesize']['ncols'])
            height=int(data_dict['annotation']['imagesize']['nrows'])
            width_scale=img_size/width
            height_scale=img_size/height
            # if image with many objects
            if isinstance(data_dict['annotation']['object'], list):
                # processing each object
                for i in range(len(data_dict['annotation']['object'])):
                    self.annotations[filename][str(i)] = {"class": data_dict['annotation']['object'][i]["name"].lower().capitalize()}
                    xmin, ymin, xmax, ymax = extract_bb_coordinates(i, data_dict['annotation']['object'])
                    self.annotations[filename][str(i)]['x_1'] = int(xmin*width_scale)
                    self.annotations[filename][str(i)]['y_1'] = int(ymin*height_scale)
                    self.annotations[filename][str(i)]['x_2'] = int(xmax*width_scale)
                    self.annotations[filename][str(i)]['y_2'] = int(ymax*height_scale)
                    whole_obj_id = ""
                    part_obj_ids = ""

                    if "ispartof" in data_dict['annotation']['object'][i]['parts']:
                        if data_dict['annotation']['object'][i]['parts']["ispartof"] is not None:
                            whole_obj_id = data_dict['annotation']['object'][i]['parts']["ispartof"]
                    if "hasparts" in data_dict['annotation']['object'][i]['parts']:
                        if data_dict['annotation']['object'][i]['parts']["hasparts"] is not None:
                            part_obj_ids  = data_dict['annotation']['object'][i]['parts']["hasparts"]
                    self.annotations[filename][str(i)]['isPartOf'] = whole_obj_id
                    self.annotations[filename][str(i)]['hasParts'] = part_obj_ids
            else:
                self.annotations[filename][str(0)] = {"class": data_dict['annotation']['object']["name"].lower().capitalize()}
                self.annotations[filename][str(0)]['isPartOf'] = ""
                self.annotations[filename][str(0)]['hasParts'] = ""
                xmin, ymin, xmax, ymax = extract_bb_coordinates(-1, data_dict['annotation']['object'])
                self.annotations[filename][str(0)]['x_1'] = int(xmin*width_scale)
                self.annotations[filename][str(0)]['y_1'] = int(ymin*height_scale)
                self.annotations[filename][str(0)]['x_2'] = int(xmax*width_scale)
                self.annotations[filename][str(0)]['y_2'] = int(ymax*height_scale)


  def toRDF(self, name=""):
    print("RDF conversion ...")
    pasPart_namespace = Namespace("http://example.org/pasPart/")
    wordnet_yago_alignment = {}
    with open('WordNet_Yago_alignment.tsv') as f:
      reader = csv.DictReader(f, delimiter='\t')
      for row in reader:
        wordnet_yago_alignment[row['PASCAL-Part_class'].lower().capitalize()] = [row['WDsynset'], row['YagoConcept']]

    g = Graph()
    pas_part_IRI = "https://dkm.fbk.eu/ontologies/semanticPASCALPart/"
    g.bind("https://dkm.fbk.eu/ontologies/semanticPASCALPart", pas_part_IRI)
    pof_uri_ref = URIRef(pasPart_namespace.isPartOf)
    hasParts_uri_ref = URIRef(pasPart_namespace.hasParts)
    g.add((pasPart_namespace.x_1, RDF.type, OWL.DatatypeProperty))
    g.add((pasPart_namespace.y_1, RDF.type, OWL.DatatypeProperty))
    g.add((pasPart_namespace.x_2, RDF.type, OWL.DatatypeProperty))
    g.add((pasPart_namespace.y_2, RDF.type, OWL.DatatypeProperty))
    g.add((pof_uri_ref, RDF.type, OWL.ObjectProperty))
    g.add((hasParts_uri_ref, RDF.type, OWL.ObjectProperty))
    g.add((hasParts_uri_ref, OWL.inverseOf, pof_uri_ref))
    for filename in self.annotations.keys():

      for obj_id in self.annotations[filename].keys():
        obj_class = self.get_obj_class(filename, obj_id)
        bb = self.get_BB(filename, obj_id)
        obj_URI = URIRef(f"{pas_part_IRI}{filename}_{obj_class}_{obj_id}")
        class_URI = URIRef(f"{pas_part_IRI}{obj_class}")

        whole_id = self.get_whole_ids(filename, obj_id)
        if whole_id is not None:
          part_class = self.get_obj_class(filename, whole_id)
          whole_URI = URIRef(f"{pas_part_IRI}{filename}_{part_class}_{whole_id}")
          g.add((obj_URI, pof_uri_ref, whole_URI))

        part_id_list = self.get_parts_ids(filename, obj_id)
        if part_id_list is not None:
            for part_id in part_id_list:
                part_class = self.get_obj_class(filename, part_id)
                part_URI = URIRef(f"{pas_part_IRI}{filename}_{part_class}_{part_id}")
                g.add((obj_URI, hasParts_uri_ref, part_URI))

        g.add((obj_URI, RDF.type, class_URI))
        g.add((obj_URI, pasPart_namespace.x_1, Literal(bb[0])))
        g.add((obj_URI, pasPart_namespace.y_1, Literal(bb[1])))
        g.add((obj_URI, pasPart_namespace.x_2, Literal(bb[2])))
        g.add((obj_URI, pasPart_namespace.y_2, Literal(bb[3])))
        g.add((obj_URI, pasPart_namespace.hasWordnetId, Literal(wordnet_yago_alignment[obj_class][0])))
        g.add((obj_URI, pasPart_namespace.hasImageName, Literal(filename)))
        g.add((obj_URI, pasPart_namespace.hasYagoConcept, URIRef(wordnet_yago_alignment[obj_class][1])))
    print("Saving RDF file ...")
    g.serialize(destination=f"semantic-PASCAL-Part_{name}.RDF")

def create_annotationFile(ann,output):
  
  # ann.load_data(split="test")
  ann_dict={}
  for filename in ann.annotations.keys():
    # print("File: ",filename)
    ann_dict[filename]={}
    # count=0
    for obj_id in ann.annotations[filename].keys():
      ann_dict[filename][obj_id]={}
      obj_class = ann.get_obj_class(filename, obj_id)
      # print("        ",obj_class)
      ann_dict[filename][obj_id]["label"]=obj_class
      bb = ann.get_BB(filename, obj_id)
      # print("            ",bb)
      ann_dict[filename][obj_id]["bbox"]=bb

      whole_id = ann.get_whole_ids(filename, obj_id)
      # print("             ",whole_id)
      if whole_id is not None:
        part_class = ann.get_obj_class(filename, whole_id)
        # print("             obj",part_class)

      part_id_list = ann.get_parts_ids(filename, obj_id)
      if part_id_list is not None:
          ann_dict[filename][obj_id]["parts"]=part_id_list
          for part_id in part_id_list:
              part_class = ann.get_obj_class(filename, part_id)
              # print("             p",part_id,part_class)
      # count+=1
      # print("     ",obj_class)
      # print("         ",bb)
  json_object=json.dumps(ann_dict,indent=4)
  # print(json_object)
  print("Creating JSON file with the annottations of the test images...")
  if path.exists(output)==False:
        os.mkdir(output)
  with open(output+"JSON_annotations.json", "w") as outfile:
    outfile.write(json_object)

    



def prepareTestFiles(img_size,img_path,output):
    ann = PASCALPArt_annotations()
    create_txtFile('all',img_path)
    ann.load_data(split="all",img_size=img_size)
    create_annotationFile(ann,output)
