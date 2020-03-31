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
        key_concepts_in_map = list()
        key_concept = nlp(kc)
        for c in concepts.values():
            c = nlp(preprocess_sentence(c))
            if is_similar(key_concept,c,0.89):
                for pro in cm['propositions']:
                    pro_txt = nlp(preprocess_sentence(concepts[pro['from']]))
                    if is_similar(pro_txt,c,0.89):
                        sentence = " ".join((pro['text'],concepts[pro['to']]))
                        key_concepts_in_map.append(sentence)
        group_map[kc] = group_map[kc] + key_concepts_in_map
    return group_map

key_concepts = ["machine learning", "unsupervised learning", "supervised learning",
                "applications", "deep learning" ,"reinforcement learning"]
for i,c in enumerate(key_concepts):
    key_concepts[i] = preprocess_sentence(c)

group_map = dict()
for kc in key_concepts:
    group_map[kc] = list()
    
group_map = process_map_for_group_map(group_map,cm1)
group_map = process_map_for_group_map(group_map,cm2)
group_map = process_map_for_group_map(group_map,cm3)

for k,v in group_map.items():
    print(k)
    print(v)





