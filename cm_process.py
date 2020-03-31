import json
import numpy as np
from pprint import pprint





with open('cm0.json') as f:
  cm0 = json.load(f)

with open('cm3.json') as f:
  cm3 = json.load(f)

with open('cm2.json') as f:
  cm2 = json.load(f)

with open('cm4.json') as f:
  cm4 = json.load(f)

def cm_to_adj(cm):
    nodes = dict()
    graph = dict()
    num_concepts = len(cm['concepts'])
    adj_mat = np.zeros((num_concepts,num_concepts), dtype=np.int)

    for idx,c in enumerate(cm['concepts']):
        nodes[c['id']] = (c['text'], idx)
        graph[c['id']] = list()

    for p in cm['propositions']:
        graph[p['from']].append(p['to'])
        adj_mat[nodes[p['from']][1]][nodes[p['to']][1]] = 1

    for i in range(num_concepts):
        for k,v in graph.items():
            for n in v:
                graph[k] = v+graph[n]


    for k,v in graph.items():
        adj_mat[nodes[k][1]][[nodes[i][1] for i in v]] = 1
        
    for idx,row in enumerate(adj_mat):
        adj_mat[idx] = np.cumsum(row)

    return adj_mat, list(nodes.values())

def get_cn(nds1,nds2):
    nodes1 = set([node[0] for node in nds1])
    nodes2 = set([node[0] for node in nds2])
    return nodes1 & nodes2

def compute_a(cn,cm1,cm2):
    a = len(cn) / min(len(cm1),len(cm2))
    return a

def get_comon_indxs(com_nodes,nodes1,nodes2):
    idx1 = list()
    idx2 = list()
    for pair in nodes1:
        if pair[0] in com_nodes:
            idx1.append(pair[1])
    for pair in nodes2:
        if pair[0] in com_nodes:
            idx2.append(pair[1])
    return idx1, idx2

def create_adj_vector(graph,idx):
    vector = []
    for i in idx:
        vector.append(i)
        vector.append(graph[i][i])
    return vector

def compute_adj_mats(grph1,grph2,com_nodes,nodes1,nodes2):
    idx1, idx2 = get_comon_indxs(com_nodes,nodes1,nodes2)
    vector1 = create_adj_vector(grph1,idx1)
    vector2 = create_adj_vector(grph2,idx2)
    return vector1,vector2

def compute_b(adj_mat1,adj_mat2):
    dot = np.dot(adj_mat1,adj_mat2)
    norm1 = np.linalg.norm(adj_mat1)
    norm2 = np.linalg.norm(adj_mat2)

    cosine_sim = dot / (norm1*norm2)
    return cosine_sim

def compute_alpha(nds1,nds2):
    nodes1 = set([node[0] for node in nds1])
    nodes2 = set([node[0] for node in nds2])
    alpha = (2 * len(nodes1 & nodes2)) / (len(nodes1) + len(nodes2))
    print("a",alpha)
    return alpha

def compute_overlapping_degree(cm1,cm2):
    ad1, n1 = cm_to_adj(cm1)   
    ad2, n2 = cm_to_adj(cm2)

    cn = get_cn(n1,n2)
    a = compute_a(cn,n1,n2)
    adj1, adj2 = compute_adj_mats(ad1,ad2,cn,n1,n2)
    b = compute_b(adj1,adj2)
    alpha = compute_alpha(n1,n2)
    od = ((a+b)/2) * alpha
    return od




print("0,2",compute_overlapping_degree(cm0,cm2))
print("0,3",compute_overlapping_degree(cm0,cm3))
print("0,4",compute_overlapping_degree(cm0,cm4))


