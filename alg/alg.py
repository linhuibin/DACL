# coding=utf-8
from alg.algs.ERM import ERM
from alg.algs.MMD import MMD
from alg.algs.CORAL import CORAL
from alg.algs.DANN import DANN
from alg.algs.RSC import RSC
from alg.algs.Mixup import Mixup
from alg.algs.Mixup1 import Mixup1

from alg.algs.MLDG import MLDG
from alg.algs.GroupDRO import GroupDRO
from alg.algs.ANDMask import ANDMask
from alg.algs.VREx import VREx
from alg.algs.IRM import IRM
from alg.algs.MTL import MTL
from alg.algs.DIFEX import DIFEX
from alg.algs.FACT import FACT
from alg.algs.DNA import DNA
from alg.algs.DACL import DACL
from alg.algs.PCL import PCL
from alg.algs.SAGM_DG import SAGM_DG

ALGORITHMS = [
    'ERM',
    'Mixup',
    'CORAL',
    'IRM',
    'MMD',
    'DANN',
    'MLDG',
    'GroupDRO',
    'RSC',
    'ANDMask',
    'VREx',
    'MTL',
    'DIFEX',
    'FACT',
    'DAN',
    'DACL',
    'DAPC_EMA',
    'PCL',
    'SAGM_DG',
    'CLIP_Linear'
    'CLIP_ZS'
]


def get_algorithm_class(algorithm_name):
    if algorithm_name not in globals():
        raise NotImplementedError(
            "Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
