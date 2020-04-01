import json
import spacy
from pprint import pprint
with open('MLMap1.json') as f:
  cm1 = json.load(f)

with open('MLMap2.json') as f:
  cm2 = json.load(f)

with open('MLMap3.json') as f:
  cm3 = json.load(f)

nlp = spacy.load("en_core_web_sm")

def is_valid_token(token):
        ret = True
        if token.is_stop:
            ret = False
        if token.pos_ == 'SYM':
            ret = False
        return ret
       
def preprocess_sentence(sentence):
    return " ".join([token.lemma_ for token in nlp(sentence) if is_valid_token(token)]).lower()
        
def is_similar(sentence1,sentence2,threshold):
    if sentence1.similarity(sentence2) >= threshold:
        return True
    return False

def process_map_for_group_map(group_map,cm):
    concepts = dict()
    for con in cm['concepts']:
        #concepts[con['id']] = preprocess_sentence(con['text'])
        concepts[con['id']] = con['text']
    for kc in group_map.keys():
        text = list()
        prop = list()
        key_concept = nlp(kc)
        for c in concepts.values():
            c = nlp(preprocess_sentence(c))
            if is_similar(key_concept,c,0.89):
                for pro in cm['propositions']:
                    pro_txt = nlp(preprocess_sentence(concepts[pro['from']]))
                    if is_similar(pro_txt,c,0.89):
                        prop.append(pro['text'])
                        text.append(concepts[pro['to']])
        group_map[kc]["propositions"] = group_map[kc]["propositions"] + prop
        group_map[kc]["text"] = group_map[kc]["text"] + text
    return group_map

key_concepts = ["machine learning", "unsupervised learning", "supervised learning",
                "applications", "deep learning" ,"reinforcement learning"]
for i,c in enumerate(key_concepts):
    key_concepts[i] = preprocess_sentence(c)

group_map = dict()
for kc in key_concepts:
    group_map[kc] = {"propositions": list(), "text": list()}
    
group_map = process_map_for_group_map(group_map,cm1)
group_map = process_map_for_group_map(group_map,cm2)
group_map = process_map_for_group_map(group_map,cm3)

def clean_group_map(group_map):
    for k,v in group_map.items():
        i = 0
        while i<len(v['text']):
            j = i + 1
            while j<len(v['text']):
                if is_similar(nlp(v['text'][i]),nlp(v['text'][j]),0.7):
                    del v['text'][j]
                    del v['propositions'][j]
                j+=1
            i+=1
        group_map[k] = v
    return group_map

group_map = clean_group_map(group_map)

for k,v in group_map.items():
    print("Key Concept:",k)
    print("What students say")
    for p,t in enumerate(v['text']):
            print(v['propositions'][p],t)
    




