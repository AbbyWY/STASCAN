import sys
import os

#STASCAN_PATH = sys.argv[1]
#vignettes_path = sys.argv[2]
#output_path = sys.argv[3]
#epochs = int(sys.argv[4])

STASCAN_PATH = "/xtdisk/yangyg_group/wuying/Analysis/STASCAN/package/" 
vignettes_path = "/xtdisk/yangyg_group/wuying/Analysis/STASCAN/package/Vignettes/Fawkner-Corbett_Intestinal_Slide6/" 
output_path = "/xtdisk/yangyg_group/wuying/Analysis/STASCAN/package/Demo1/"
epochs = 2
#epochs = 50


sys.path.append(STASCAN_PATH)
import STASCAN


######################################################################

label_list = ['Epithelium', 'Fibroblasts', 'Muscularis', 'Neural']
color_list = ["purple", "orange", "red", "gold"]
dict_label = dict(zip(label_list, color_list))

crop_size = 40
raw_image = vignettes_path + "/A6.jpg"
adjacent_image = vignettes_path+"/Simulated_A7.jpg"


######################################################################


## Module 1: Cell annotation for unseen spots 

if not os.path.exists(output_path + "/Module1/"):
	os.makedirs(output_path + "/Module1/")
	
Module1_output = output_path + "/Module1/"

run = STASCAN.run_STASCAN.Module()
run.UnseenSpot(Module1_output, vignettes_path + "/tissue_positions_list.csv", raw_image, crop_size, vignettes_path, epochs=epochs)


# Metrics
fig = STASCAN.StatPlot.Metric()
fig.ROC_Curve(Module1_output + "/PriorSpot/test/", Module1_output + "/Predict/Rawpredict_detail.txt", label_list, color_list, Module1_output + "/Models/ROC.pdf")
fig.Loss_Accuracy_Curve(Module1_output + "/Models/Log_BaseModel.txt", Module1_output + "/Models/")


fig = STASCAN.StatPlot.Check()
fig.Check_PriorSpot(Module1_output + "/ImputedSpot/adjust_raw_spot.txt", Module1_output +  "/PriorSpot/prior_spot.txt", raw_image, dict_label, Module1_output)
fig.Check_ImputedSpot(Module1_output + "/ImputedSpot/adjust_raw_spot.txt", Module1_output + "ImputedSpot/imputed_spot.txt", raw_image, Module1_output)


# Visualization
fig = STASCAN.run_STASCAN.Visualization()
fig.EnhancedPlot(Module1_output + "/Predict/Raw_predict.txt", Module1_output + "/PriorSpot/prior_spot.txt", Module1_output + "/Predict/Imputed_predict.txt", raw_image, dict_label, Module1_output)



## Module 2: Cell annotation for subdivided spots

if not os.path.exists(output_path + "/Module2/"):
	os.makedirs(output_path + "/Module2/")
	
Module2_output = output_path + "/Module2/"

run.SubdividedSpot(Module2_output, vignettes_path + "/tissue_positions_list.csv", raw_image, crop_size, vignettes_path, threshold_proportion=0.5, epochs=epochs)


# Metrics
fig = STASCAN.StatPlot.Metric()
fig.ROC_Curve(Module2_output + "/SubSpot/prior_divided/test/", Module2_output + "/Predict/Subdividedpredict_detail.txt", label_list, color_list, Module2_output + "/Models/ROC.pdf")
fig.Loss_Accuracy_Curve(Module2_output + "/Models/Log_BaseModel.txt", Module2_output + "/Models/")


# Visualization
fig = STASCAN.run_STASCAN.Visualization()
fig.SubResolutionPlot(Module1_output+"/Predict/Raw_predict.txt", Module2_output+"/Predict/Subdivided_predict.txt", raw_image, dict_label, Module2_output)



## Module 3: Cell annotation for unseen sections

if not os.path.exists(output_path + "/Module3/"):
	os.makedirs(output_path + "/Module3/")
	
Module3_output = output_path + "/Module3/"

spot_label = [[x] for x in label_list]

run.UnseenSection(Module3_output, vignettes_path+"/Simulated_tissue_positions_list.csv", adjacent_image, crop_size, Module1_output+"/Models/base_model.h5", spot_label)

fig = STASCAN.run_STASCAN.Visualization()
fig.AdjacentSectionPlot(Module3_output+"/Predict/Adjacent_raw_predict.txt", Module3_output+"/Predict/Adjacent_imputed_predict.txt", adjacent_image, dict_label, Module3_output)


