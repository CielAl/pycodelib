from pycodelib.dlengine import SkinEngine
from torchvision.models.densenet import DenseNet
import torch
import torch.nn as nn
from pycodelib.patients.patient_gt import *
from torch.optim import Adam
import glob
from torchvision import transforms
from pycodelib.dlengine.iterator_getter import H5DataGetter
# patient col
file_list = glob.glob("E:\\melanoma\\melanoma_data\\ROI_new_extractor_20x_v1\\*.png")
patient_table = SheetCollection(file_list=file_list, sheet_name="E:\\melanoma\\melanoma_data\\Patient_June24.xlsx")

# getter
pydir_t = 'C:\\pytable\\ROI_inspect\\melanoma_20x_inspect_train.pytable'
pydir_v = 'C:\\pytable\\ROI_inspect\\melanoma_20x_inspect_val.pytable'
img_transform =  \
 transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=.5),
        transforms.RandomGrayscale(),
        transforms.ToTensor()
    ])
getter = H5DataGetter.build({True: pydir_t, False: pydir_v}, {True: img_transform, False: img_transform})

# engine
model = DenseNet(growth_rate=4, block_config=(1, 1), num_init_features=16, bn_size=4, num_classes=2)
device = torch.device('cuda:0')
loss = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())
engine = SkinEngine(device=device, model=model, loss=loss, iterator_getter=getter, val_phases=['train', 'val'],
                    patient_col=patient_table, sub_class_list=['No Path', 'SCC'],
                    class_partition=[[0], [2, 3]])
print('start engine')
if __name__ == '__main__':
    engine.process(maxepoch=1, optimizer=optimizer, num_workers=12)


