import numpy as np

vector1 = np.array([
            [0, 4],
            [0, 0]
          ])

vector2 = np.array([
            [0, 1],
            [0, 0],
          ])
def compute_overlapping_degree(cm1,cm2):
  grp1,nds1 = cm_to_adj(cm1)
  grp2,nds2 = cm_to_adj(cm2)
  cn = get_cn(nds1,nds2)
  a = compute_a(cn,cm1,cm2)
  adj_mat1, adj_mat2 = compute_adj_mats(cm1,cm2)
  b = compute_b(adj_mat1,adj_mat2)
  alpha = compute_alpha(cm1,cm2)
  od = ((a+b)/2) * alpha


