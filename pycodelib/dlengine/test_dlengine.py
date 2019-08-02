from pycodelib.dlengine.iterator_getter import H5DataGetter
pydir = 'C:\\pytable\\ROI_new_extractor_20x_512\\melanoma_20x_new_train.pytable'
from torchvision import transforms
import matplotlib.pyplot as plt
import PIL
import time
img_transform = \
 transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=.5),
        transforms.RandomGrayscale(),
        transforms.ToTensor()
    ])

getter = H5DataGetter.build({True: pydir, False: pydir}, {True: img_transform, False: img_transform})


dl = getter.get_iterator(True, True, num_workers=6, pin_memory=True)

start = time.time()
for x in dl:
    continue
end = time.time()
print(end-start)
#None: 268.679571390152
#aug: 976
#250.12752485275269 --> 6 workers