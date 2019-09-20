import openslide, glob
from openslide import OpenSlide
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from pycodelib.dataset import SlideSet
from torchvision.models import DenseNet

model = DenseNet(block_config=(4,4,4,4)).to(torch.device('cuda:0'))
model.eval()

wsi_list = glob.glob('C:\\wsi_sample\\*.ndpi')
patch_size = 256
test_set = SlideSet(wsi_list[0], patch_size=patch_size, level=1)
dl = DataLoader(test_set, batch_size=64, num_workers=16, timeout=0, pin_memory=False, drop_last=True)

for x in tqdm(dl):
    x = x.permute([0, 3, 1, 2])[:, 0:-1, :].type('torch.FloatTensor')
    x = x.to(torch.device('cuda:0'))
    model(x)
