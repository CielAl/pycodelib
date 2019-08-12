from pycodelib.patients.patient_gt import *
from pycodelib.patients.patient_pred import *
import glob
file_list = glob.glob("E:\\melanoma\\melanoma_data\\test_patient\\img\\*.png")
# glob.glob("E:\\melanoma\\melanoma_data\\ROI_new_extractor_20x_v1\\*.png")
sheet_loc = "E:\\melanoma\\melanoma_data\\test_patient\\test_patient.xlsx"
# "E:\\melanoma\\melanoma_data\\Patient_June24.xlsx"
table = SheetCollection(file_list=file_list, sheet_name=sheet_loc)

pred = CascadedPred(table, ['No Path', 'Cancer'], [[0], [1, 2, 3]])
pred.get_ground_truth(['15205311501.29', '125550.0'])
# table.load_prediction(['BCC'],['17S049641_2A_L123 - 2018-03-20 07.41.52_BCC-LR-Superficial_(1, 89770, 15260, 758, 720).png'])
pred.load_score([[0.5, 0.5]],
                ['17S049630_1A_L12 - 2018-03-20 07.25.16_SCC-Invasive-Well_(1, 12342, 7556, 887, 2148).png'],
                flush=True)