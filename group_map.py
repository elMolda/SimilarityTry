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

kc_pair = dict()
key_concepts = ["machine learning", "unsupervised learning", "supervised learning",
                "applications", "deep learning" ,"reinforcement learning"]
for i,c in enumerate(key_concepts):
    key_concepts[i] = preprocess_sentence(c)
    

group_map = dict()
for kc in key_concepts:
    group_map[kc] = {"propositions": list(), "text": list()}
cms = [cm1,cm2,cm3]    
for cm in cms:
    group_map = process_map_for_group_map(group_map,cm)    
group_map = clean_group_map(group_map)         
    
json_grp_map = dict()
json_grp_map['concepts'] = []
json_grp_map['propositions'] = []

id_gen = 0
concepts = dict()
for k,v in group_map.items():
    concepts[k] = id_gen
    id_gen += 1
    for con in v['text']:
        concepts[con] = id_gen
        id_gen += 1

for text, id in concepts.items():
    con_entry = dict()
    con_entry['text'] = text
    con_entry['id'] = id
    json_grp_map['concepts'].append(con_entry)
    for v in group_map.values():
        for pro,txt in enumerate(v['text']):
            pro_entry = dict()
            pro_entry['text'] = v['propositions'][pro]
            pro_entry['from'] = concepts[text]
            pro_entry['to'] = concepts[txt]
            json_grp_map['propositions'].append(pro_entry)

with open('result.json', 'w') as fp:
    json.dump(json_grp_map, fp)





