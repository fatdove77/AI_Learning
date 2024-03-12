import torch
import torch.nn as nn

# Import pprint, module we use for making our print statements prettier
import pprint

x = torch.arange(1,7)
a = x.view(2,3)*1.0
a_cat0 = torch.cat([a,a,a],0)
a_cat1 = torch.cat([a,a,a],1)
pprint.pprint(a)
pprint.pprint(a_cat0)
pprint.pprint(a_cat1)