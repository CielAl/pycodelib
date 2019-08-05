from pycodelib.dlengine.iterator_getter import H5DataGetter
pydir = 'C:\\pytable\\ROI_new_extractor_20x_512\\melanoma_20x_new_train.pytable'
from torchvision import transforms
import matplotlib.pyplot as plt
import PIL
import time
img_transform =  \
 transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(180),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=.5),
        transforms.RandomGrayscale(),
        transforms.ToTensor()
    ])

getter = H5DataGetter.build({True: pydir, False: pydir}, {True: img_transform, False: img_transform})


dl = getter.get_iterator(True, True, num_workers=12, pin_memory=True, batch_size=32)

start = time.time()
for x in dl:
    break
end = time.time()
print(end-start)
# test = img_transform(x[0][1,].numpy()).permute([1,2,0])
#None: 268.679571390152
#aug: 976
# No Aug = 98.83628535270691 -> 6workers/Batch 32 ||
# Aug = 250.12752485275269 --> 6 workers/Batch 32 ||237.85382866859436 64batch
