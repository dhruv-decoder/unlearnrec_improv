import os
import sys
import pickle
import torch as t
import torch_sparse as ts 
import scipy.sparse as sp
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
from Utils.safe_io import safe_pickle_load


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
file1 = os.path.join(_SCRIPT_DIR, 'trn_mat.pkl')
file2 = os.path.join(_SCRIPT_DIR, 'adv_mat.pkl')
file3 = os.path.join(_SCRIPT_DIR, 'adv_lightgcn_mat.pkl')


def load_one_file( filename, adversarial_attack=False,test_file=False,non_binary=False):
    tem = safe_pickle_load(filename)
    if adversarial_attack and (not test_file):
        adv_edges = tem[1] 
        tem = tem[0]                           
    ret = tem if non_binary else (tem != 0).astype(np.float32)
    if type(ret) != coo_matrix:
        ret = sp.coo_matrix(ret)
    ret = ts.SparseTensor.from_scipy(ret)
    return ret



trnMat = load_one_file(file1)
advMat = load_one_file(file2, adversarial_attack=True)
adv_lightgcn_mat = load_one_file(file3, adversarial_attack=True)

print("#################trnMat####################")
print(trnMat)

print("#################advMat####################")
print(advMat)

print("#################adv_lightgcn_mat####################")
print(adv_lightgcn_mat)


