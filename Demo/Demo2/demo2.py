import sys
import os
import shutil
import glob

#STASCAN_PATH = sys.argv[1]
#vignettes_path = sys.argv[2]
#output_path = sys.argv[3]

#STASCAN_PATH = "../../"
#sys.path.append(STASCAN_PATH)
import STASCAN


vignettes_path = "../../Vignettes/Cui_Planarian_Multisections/"
output_path = "./"
adjacent_path = "../../Vignettes/Cui_Planarian_Adjacentimages/"


######################################################################

label_list = ['Epidermal', 'Gut', 'Muscle', 'Neoblast', 'Neuronal', 'Parenchymal', 'Secretory']
color_list = ["blue", "green", "red", "grey", "orange", "pink", "purple"]
dict_label = dict(zip(label_list, color_list))
spot_label = [[x] for x in label_list]

crop_size = 30
imagetype = "png"
grey_level, white_threshold = 200, 0.7


######################################################################

## Pre-training

run = STASCAN.run_STASCAN.Optional()
#run.SectionSpecificTraining(output_path, vignettes_path, imagetype, crop_size, epoch1=2, epoch2=2)
run.SectionSpecificTraining(output_path, vignettes_path, imagetype, crop_size, epoch1=50, epoch2=50)


######################################################################

## Module 1: Cell annotation for unseen spots (Section-25, Fig.3b)

# Preparation
step1 = STASCAN.preparation.Generator(output_path + "/section8/", vignettes_path + "/section8/tissue_positions_list.csv", vignettes_path + "/section8/Section-25.png", crop_size)
step1.RawSpot_generator()
step1.ImputedSpot_generator()


# Prediction
step2 = STASCAN.model.BuildPrediction(output_path + "/section8/Models/finetuned_model.h5", output_path + "/section8/")
step2.prediction(spot_label, output_path + "/section8/ImputedSpot", output_path + "/section8/ImputedSpot/imputed_spot.txt", "Imputed")
step2.prediction(spot_label, output_path + "/section8/RawSpot", output_path + "/section8/ImputedSpot/adjust_raw_spot.txt", "Raw")


# Metrics
fig = STASCAN.StatPlot.Metric()
fig.ROC_Curve(output_path + "/section8/PriorSpot/test/", output_path + "/section8/Predict/Rawpredict_detail.txt", label_list, color_list, output_path + "/section8/Models/ROC.pdf")
fig.Loss_Accuracy_Curve(output_path + "/section8/Models/Log_FinetunedModel.txt", output_path + "/section8/Models/")
fig.Confusion_Matrix(output_path + "/section8/PriorSpot/test/", output_path + "/section8/Predict/Raw_predict.txt", label_list, output_path + "/section8/Models/")

fig = STASCAN.StatPlot.Check()
fig.Check_PriorSpot(output_path + "/section8/ImputedSpot/adjust_raw_spot.txt", output_path + "/section8/PriorSpot/prior_spot.txt", vignettes_path + "/section8/Section-25.png", dict_label, output_path + "/section8/")
fig.Check_ImputedSpot(output_path + "/section8/ImputedSpot/adjust_raw_spot.txt", output_path + "/section8/ImputedSpot/imputed_spot.txt", vignettes_path + "/section8/Section-25.png", output_path + "/section8/")


# Visualization
fig = STASCAN.run_STASCAN.Visualization()
fig.EnhancedPlot(output_path + "/section8/Predict/Raw_predict.txt", output_path + "/section8/PriorSpot/prior_spot.txt", output_path + "/section8/Predict/Imputed_predict.txt", vignettes_path + "/section8/Section-25.png", dict_label, output_path + "/section8/", pointsize=18)


######################################################################

## Module 2: Cell annotation for subdivided spots (Section-25, Fig.3c)

if not os.path.exists(output_path + "/section8/SubModule/"):
	os.makedirs(output_path + "/section8/SubModule/")

# Optional
pseudo_spot = run.PseudoLabelling(output_path + "/section8/SubModule/", output_path + "/section8/RawSpot", output_path + "/section8/Predict/Rawpredict_detail.txt", label_list)

for each in label_list:
	if each not in list(set([x[-1] for x in pseudo_spot])):
		imgsLib = []
		imgsLib.extend(glob.glob(os.path.join(output_path + "/section8/PriorSpot/test/" + each + "/", "*.png")))
		for n in imgsLib:
			shutil.copy(n, output_path + "/section8/SubModule/PseudoSpot/")

shutil.copytree(output_path + "/section8/SubModule/PseudoSpot/", output_path + "/section8/SubModule/PriorSpot/")
shutil.copytree(output_path + "/section8/RawSpot/", output_path + "/section8/SubModule/RawSpot/")
shutil.move(output_path + "/section8/SubModule/PriorSpot/pseudo_spot.txt",output_path + "/section8/SubModule/PriorSpot/prior_spot.txt")


# Preparation
step3 = STASCAN.preparation.Generator(output_path + "/section8/SubModule/", vignettes_path + "/section8/tissue_positions_list.csv", vignettes_path + "/section8/Section-25.png", crop_size)
step3.SubSpot_generator()

step4 = STASCAN.preparation.Shuffler(output_path + "/section8/SubModule/SubSpot/prior_divided/")
step4.Dataset_divider(spot_label)

for each in label_list:
	if not os.listdir(output_path + "/section8/SubModule/SubSpot/prior_divided/test/" + each):
		imgsLib = []
		imgsLib.extend(glob.glob(os.path.join(output_path + "/section8/SubModule/SubSpot/prior_divided/train/" + each + "/", "*.png")))
		shutil.move(imgsLib[0], output_path + "/section8/SubModule/SubSpot/prior_divided/test/" + each + "/")
		

# Training
step5 = STASCAN.model.BuildModel()
traindata_path, testdata_path = output_path + "/section8/SubModule/SubSpot/prior_divided/train/", output_path + "/section8/SubModule/SubSpot/prior_divided/test/"
step5.finetuned_model(output_path + "/section8/Models/finetuned_model.h5", traindata_path, testdata_path, output_path + "/section8/SubModule/SubSpot/")


# Prediction
step6 = STASCAN.model.BuildPrediction(output_path + "/section8/SubModule/SubSpot/Models/finetuned_model.h5", output_path + "/section8/SubModule/SubSpot/")
step6.prediction(spot_label, output_path + "/section8/SubModule/SubSpot/raw_divided/", output_path + "/section8/SubModule/SubSpot/raw_divided/subloc.txt", "Subdivided")


# Metrics
fig = STASCAN.StatPlot.Metric()
fig.ROC_Curve(output_path + "/section8/SubModule/SubSpot/prior_divided/test/", output_path + "/section8/SubModule/SubSpot/Predict/Subdividedpredict_detail.txt", label_list, color_list, output_path + "/section8/SubModule/SubSpot/Models/ROC.pdf")
fig.Loss_Accuracy_Curve(output_path + "/section8/SubModule/SubSpot/Models/Log_FinetunedModel.txt", output_path + "/section8/SubModule/SubSpot/Models/")
fig.Confusion_Matrix(output_path + "/section8/SubModule/SubSpot/prior_divided/test/", output_path + "/section8/SubModule/SubSpot/Predict/Subdivided_predict.txt", label_list, output_path + "/section8/SubModule/SubSpot/Models/")


# Visualization
fig = STASCAN.run_STASCAN.Visualization()
fig.SubResolutionPlot(output_path + "/section8/Predict/Raw_predict.txt", output_path + "/section8/SubModule/SubSpot/Predict/Subdivided_predict.txt", vignettes_path+ "/section8/Section-25.png", dict_label, output_path + "/section8/SubModule/SubSpot/", pointsize=18)


######################################################################

## Module 3: Cell annotation for unseen sections (Adjacent-22/Section-21, Fig.3e)

if not os.path.exists(output_path + "/section6/AdjacentModule/"):
	os.makedirs(output_path + "/section6/AdjacentModule/")

# Prepatation
step7 = STASCAN.preparation.Generator(output_path + "/section6/AdjacentModule/", vignettes_path + "/section6/tissue_positions_list.csv", adjacent_path+"/image22/img22.png", crop_size)
step7.RawSpot_generator()
step7.ImputedSpot_generator()

step8 = STASCAN.preparation.Detection()
step8.DetectWhiteRegion(output_path + "/section6/AdjacentModule/RawSpot/", grey_level, white_threshold)
step8.DetectWhiteRegion(output_path + "/section6/AdjacentModule/ImputedSpot/", grey_level, white_threshold)


# Prediction
step9 = STASCAN.model.BuildPrediction(output_path + "/section6/Models/finetuned_model.h5", output_path + "/section6/AdjacentModule/")
step9.prediction(spot_label, output_path + "/section6/AdjacentModule/ImputedSpot/", output_path + "/section6/AdjacentModule/ImputedSpot/imputed_spot.txt", "Adjacent_imputed")
step9.prediction(spot_label, output_path + "/section6/AdjacentModule/RawSpot", output_path + "/section6/AdjacentModule/ImputedSpot/adjust_raw_spot.txt", "Adjacent_raw")
       
fig = STASCAN.run_STASCAN.Visualization()
fig.AdjacentSectionPlot(output_path + "/section6/AdjacentModule/Predict/Adjacent_raw_predict.txt", output_path + "/section6/AdjacentModule/Predict/Adjacent_imputed_predict.txt", adjacent_path+"/image22/img22.png", dict_label, output_path + "/section6/AdjacentModule/", pointsize=18)

