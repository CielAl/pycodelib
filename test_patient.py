from pycodelib.patients.patient import *
import glob
file_list = glob.glob("E:\\melanoma\\melanoma_data\\ROI_new_extractor_20x_v1\\*.png")
table = SheetCollection(file_list=file_list, sheet_name="E:\\melanoma\\melanoma_data\\Patient_June24.xlsx")
table.load_prediction(['BCC'],['17S049641_2A_L123 - 2018-03-20 07.41.52_BCC-LR-Superficial_(1, 89770, 15260, 758, 720).png'])