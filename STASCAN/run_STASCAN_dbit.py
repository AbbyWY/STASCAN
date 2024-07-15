import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import SubplotZero
import matplotlib.image as imgplt
#%matplotlib inline



import STASCAN.preparation_dbit as preparation_dbit
import STASCAN.model as model
import STASCAN.preparation as preparation






class Module():

    def __init__(self):
        self.name = "Module"



    def UnseenSpot(self, result_dir, position, image, crop_size,  origin_set, gap_length, prelabel_path, allprelabel_file, dir_list, downsample_number, min_num, shuffle_scale=0.2, lr=1e-3, epochs=50, model_name="base_model.h5", grey_level=200, threshold=0.7):

        """
        Cell annotation for unseen spots.

        Parameters
        ----------
        result_dir
            Path of the output folder.
        position
            The position of raw spots. Standard output file for Space Ranger.
        image
            The original H&E staining images, used for Space Ranger previously.
        crop_size
            Pixel radius of spot images.
        origin_set
            Setted location for the original spot.
        gap_length
            Setted width of the gaps between channels.
        prelabel_path
            Path of prelabel files.
        shuffle_scale
            Shuffle scale between training and testing sets.
        lr
            Learning rate for SGDOptimizer.
        epochs
            Number of total epochs in training.
        model_name
            Name of trained model
        """

        # Preparation
        step1 = preparation_dbit.Generator(result_dir, position, image, crop_size, origin_set, gap_length)
        prior_spot = step1.PriorSpot_generator(prelabel_path)
        step1.RawSpot_generator(allprelabel_file)
        step1.ImputedSpot_generator()
        
        step2 = preparation.Detection()
        step2.DetectWhiteRegion(result_dir + "/PriorSpot/", grey_level, threshold)

        step3 = preparation_dbit.Shuffler(result_dir + "/PriorSpot/")
        class_dir = step3.Dataset_divider_downsampling(dir_list, downsample_number, min_num)

        # Training
        step4 = model.BuildModel()
        traindata_path, testdata_path = result_dir + "/PriorSpot/train/", result_dir + "/PriorSpot/test/"
        step4.base_model(traindata_path, testdata_path, result_dir, len(class_dir), lr, epochs, model_name)

        # Prediction
        step5 = model.BuildPrediction(result_dir + "/Models/" + model_name, result_dir)
        pre_class = [[x] for x in class_dir]
        step5.prediction_dbit(pre_class, result_dir + "/ImputedSpot", "Imputed")       
        step5.prediction_dbit(pre_class, result_dir + "/RawSpot", "Raw")





class Visualization():

    def __init__(self):

        self.name = "Visualization"



    def EnhancedPlot(self, origin_set, gap_length, position, raw_spot, prior_spot, imputed_spot, fill_full_list, label_color, output_path, figsize=(10, 10), xylim=1500, pointsize=10):

        """
        Visualization of enhanced cell annotations.

        Parameters
        ----------
        origin_set
            Setted location for the original spot.
        gap_length
            Setted width of the gaps between channels.
        position
            The position of raw spots.
        raw_spot
            Path of the predicted raw spot file.
        prior_spot
            Path of the predicted prior spot file.
        imputed_spot
            Path of the predicted imputed spot file.
        fill_full_list
            Path of the fill_full_list file.
        label_color
            Dictionary of label colors.
        output_path
            Path of the output files.
        pointsize
            Size of scatter points.
        """

        self.origin_set = origin_set

        self.gap_length = gap_length

        self.xy_array = []
        for i in range(50):
            for j in range(50):
                self.xy_array.append([str(i + 1) + 'x' + str(j + 1), self.origin_set[0] + j * self.gap_length,
                                      self.origin_set[1] + i * self.gap_length])

        self.xy_array_dict = {}
        for each in self.xy_array:
            self.xy_array_dict[each[0]] = each[1:]

        self.pixel_xy = {}
        with open(position) as f:
            next(f)
            for line in f.readlines():
                temp = line.split()
                self.pixel_xy[temp[2]] = temp[1]

        ###
        celltype = []
        with open(raw_spot) as f:
            for line in f.readlines():
                temp = line.split()
                celltype.append(temp)

        xy_celltype = []
        for each in celltype:
            temp = each
            temp.extend(self.xy_array_dict[self.pixel_xy[each[0].split("-")[1]+"-1"]])
            xy_celltype.append(temp)

        x = [int(k[-1]) for k in xy_celltype]
        y = [int(k[-2]) for k in xy_celltype]
        z = [k[2] for k in xy_celltype]
        b = [k[0] for k in xy_celltype]

        ###
        celltype = []
        with open(imputed_spot) as f:
            for line in f.readlines():
                temp = line.split()
                celltype.append(temp)

        fill_dict = {}
        with open(fill_full_list) as f:
            for line in f.readlines():
                temp = line.split()
                fill_dict[temp[0]] = temp[1:]

        xy_celltype = []
        for each in celltype:
            temp = each
            temp.extend(fill_dict[self.pixel_xy[each[0].split('_')[0]] + each[0].split('-1')[-1]])
            xy_celltype.append(temp)

        fill_x = [float(k[-1]) for k in xy_celltype]
        fill_y = [float(k[-2]) for k in xy_celltype]
        fill_z = [k[2] for k in xy_celltype]

        ###
        celltype = []
        with open(prior_spot) as f:
            for line in f.readlines():
                temp = line.split()
                celltype.append(temp)

        xy_celltype = []
        for each in celltype:
            temp = each
            temp.extend(self.xy_array_dict[self.pixel_xy[each[0]]])
            xy_celltype.append(temp)

        select_x = [int(k[-1]) for k in xy_celltype]
        select_y = [int(k[-2]) for k in xy_celltype]
        select_z = [k[1] for k in xy_celltype]
        select_b = [k[0] for k in xy_celltype]

        ###
        fig = plt.figure(figsize=figsize)
        ax = SubplotZero(fig, 1, 1, 1)
        fig.add_subplot(ax)
        ax.set_xlim(left=0, right=xylim)
        ax.set_ylim(bottom=xylim, top=0)

        for i in range(len(select_x)):
            plt.scatter(select_x[i], select_y[i], c=label_color[select_z[i]][1], marker='s', s=pointsize)
        for i in range(len(x)):
            if b[i] not in select_b:
                plt.scatter(x[i], y[i], c=label_color[z[i]][1], marker='s', s=pointsize)
        for i in range(len(fill_x)):
            plt.scatter(fill_x[i], fill_y[i], c=label_color[fill_z[i]][1], marker='s', s=pointsize)
        plt.savefig(output_path + "/EnhancedPlot.pdf")

        ###
        fig = plt.figure(figsize=figsize)
        ax = SubplotZero(fig, 1, 1, 1)
        fig.add_subplot(ax)
        ax.set_xlim(left=0, right=xylim)
        ax.set_ylim(bottom=xylim, top=0)

        for i in range(len(select_x)):
            plt.scatter(select_x[i], select_y[i], c=label_color[select_z[i]][1], marker='s', s=pointsize)

        for i in range(len(x)):
            if b[i] not in select_b:
                plt.scatter(x[i], y[i], facecolors='none', marker='s', edgecolors='black', linewidths=0.1, s=10)

        plt.savefig(output_path + "/PriorPlot.pdf")




