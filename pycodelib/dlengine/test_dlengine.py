from pycodelib.dlengine.iterator_getter import H5DataGetter
pydir = 'C:\\pytable\\ROI_new_extractor_20x_512\\melanoma_20x_new_train.pytable'
getter = H5DataGetter.build({True: pydir, False: pydir}, {True: None, False: None})
