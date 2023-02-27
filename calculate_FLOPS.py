
import torch
import utils
import torch.nn as nn
import genotypes
from torch.autograd import Variable
from model import NetworkImageNet as Network
from ptflops import get_model_complexity_info

genotype = eval("genotypes.%s" % "DARTS")
model = Network(48, 1000, 14, False, genotype)

flops, params = get_model_complexity_info(model, (224, 224), as_strings=True, print_per_layer_stat=True)
print('Flops:  {}'.format(flops))
print('Params: ' + params)