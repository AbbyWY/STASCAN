import sys
import os
import shutil
import glob



#STASCAN_PATH = "../../"
#sys.path.append(STASCAN_PATH)
import STASCAN


vignettes_path = "../../Vignettes/Jiang_mousebrain_E15_5/"
output_path = "./"


######################################################################

image = vignettes_path + "./E15_5-S1-HE.jpg"
crop_size = 8

origin_set = [10, 10]
gap_length = 29.5

Dir = ["Corticalorhippocampalglutamatergic", "ForebrainGABAergic", "Forebrainglutamatergic",
       "HindbrainGABAergic", "Hindbrainglutamatergic", "Hindbrain",
       "Hypothalamus", "MidbrainGABAergic", "Midbrainglutamatergic",
       "Midbrain", "Mixedregionandneurotransmitter", "MixedregionGABAergic",
       "Mixedregionglutamatergic", "Mixedregion", "Undefined"]

color_map = {'CerebellumGABAergic': ['#FFC0CB', 'pink'],
             'Corticalorhippocampalglutamatergic': ['#FF1493', 'deeppink'],
             'Forebrain': ['#FF00FF', 'magenta'],
             'ForebrainGABAergic': ['#800080', 'purple'],
             'Forebrainglutamatergic': ['#8A2BE2', 'blueviolet'],
             'Hindbrain': ['#0000FF', 'blue'],
             'HindbrainGABAergic': ['#00BFFF', 'deepskyblue'],
             'Hindbrainglutamatergic': ['#6495ED', 'cornflowerblue'],
             'Hindbrainglycinergic': ['#00FFFF', 'cyan'],
             'Hypothalamus': ['#00CED1', 'darkturquoise'],
             'Midbrain':['#2F4F4F', 'darkslategray'],
             'MidbrainGABAergic': ['#808000', 'olive'],
             'Midbrainglutamatergic': ['#F5FFFA', 'mediumspringgreen'],
             'Mixedregion': ['#2E8B57', 'seagreen'],
             'MixedregionGABAergic': ['#90EE90', 'lightgreen'],
             'Mixedregionandneurotransmitter': ['#FFFF00', 'yellow'],
             'Mixedregionglutamatergic': ['#FFA500', 'orange'],
             'Pia3': ['#FF0000', 'red'],
             'Undefined': ['#A0522D', 'sienna']}

######################################################################

#epochs = 2
epochs = 50
position = vignettes_path + "/xy.txt"
prelabel_path = vignettes_path + "/RCTD_prior.txt"
allprelabel_file = vignettes_path + "/RCTD_prelabelling_all.txt"
downsample_number = 300
min_num = 12

run = STASCAN.run_STASCAN_dbit.Module()
run.UnseenSpot(output_path, position, image, crop_size, origin_set, gap_length, prelabel_path, allprelabel_file, Dir, downsample_number=downsample_number, min_num=min_num, epochs=epochs)


fig = STASCAN.run_STASCAN_dbit.Visualization()
fig.EnhancedPlot(origin_set, gap_length, position, output_path + "/Predict/Raw_predict.txt", prelabel_path, output_path + "/Predict/Imputed_predict.txt", output_path + "/fill_full_list.txt", color_map, output_path)


import STASCAN.downstream as downstream
n_kmeans = 10
cmap = "tab10"
raw_pixel_array = vignettes_path + "/MISAR-seq_barcode.csv"
colormap = {"0": "#1F77B4", "1": "#FF7F0E", "2": "#2CA02C", "3": "#D62728",
            "4": "#9467BD", "5": "#8C564B", "6": "#E377C2", "7": "#7F7F7F",
            "8": "#BCBD22", "9": "#17BECF"}

niche = downstream.FindNiche(output_path)
niche.Kmeans(output_path + "/Predict/Rawpredict_detail.txt", output_path + "/Predict/Imputedpredict_detail.txt", n_kmeans)
niche.TSNE_Plot(output_path + "/Predict/", output_path + "/Downstream/kmeans_label.txt", n_kmeans, cmap)
niche.process_xy_dbit(origin_set, gap_length, raw_pixel_array, output_path + "/fill_full_list.txt")
niche.SpatialNiche_dbit(output_path + "/Downstream/kmeans_label.txt", colormap)


