from pycodelib.patients.patient import *
import glob
file_list = glob.glob("E:\\melanoma\\melanoma_data\\ROI_new_extractor_20x_v1\\*.png")
table = SkinCollection(file_list=file_list, sheet_name="E:\\melanoma\\melanoma_data\\Patient_June24.xlsx")
