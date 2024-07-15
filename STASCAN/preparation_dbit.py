from pylab import *
import numpy as np
import cv2
import os
import random
import glob
import shutil
from PIL import Image
from mpl_toolkits.axisartist.axislines import SubplotZero
import matplotlib.image as imgplt





class Generator():

    def __init__(self, record_dir, position, image, crop_size, origin_set, gap_length):

        """
        Preparation of spot images.

        Parameters
        ----------
        record_dir
            Path of the output folder.
        position
            The position of raw spots.
        image
            The original H&E staining images, used for Space Ranger previously.
        crop_size
            Pixel radius of spot images.
        origin_set
            Setted location for the original spot.
        gap_length
            Setted width of the gaps between channels.
        """

        self.name = "Generator"

        self.record_dir = record_dir

        self.origin_set = origin_set

        self.gap_length = gap_length

        self.xy_array = []
        for i in range(50):
            for j in range(50):
                self.xy_array.append([str(i + 1) + 'x' + str(j + 1), self.origin_set[0] + j * self.gap_length, self.origin_set[1] + i * self.gap_length])

        self.xy_array_dict = {}
        for each in self.xy_array:
            self.xy_array_dict[each[0]] = each[1:]

        self.pixel_xy = {}
        with open(position) as f:
            next(f)
            for line in f.readlines():
                temp = line.split()
                self.pixel_xy[temp[2]] = temp[1]

        self.rawimage = image
        self.crop_size = crop_size



    def SpotImage_generator(self, type):

        """
        Generator of spot images.

        Parameters
        ----------
        type: {'RawSpot', 'PriorSpot', 'ImputedSpot'}
            Type of spots.
        """

        img = cv2.imread(self.rawimage)
        crop_size = self.crop_size

        if type == 'RawSpot':
            position = self.raw_position
            for each in position:
                cropped = img[int(each[-2]) - crop_size:int(each[-2]) + crop_size,
                          int(each[-1]) - crop_size:int(each[-1]) + crop_size]
                cv2.imwrite(self.record_dir + "/RawSpot/" + each[1] + '-' + each[0] + ".png", cropped)
        elif type == 'PriorSpot':
            position = self.prior_spot
            for each in position:
                cropped = img[int(each[-2]) - crop_size:int(each[-2]) + crop_size,
                          int(each[-1]) - crop_size:int(each[-1]) + crop_size]
                cv2.imwrite(self.record_dir + "/PriorSpot/" + each[1] + '-' + each[0] + ".png", cropped)
        elif type == 'ImputedSpot':
            position = self.imputed_spot
            for i in range(len(position)):
                if i % 4 == 0:
                    try:
                        crop_name = position[i] + '_1'
                        cropped = img[int(position[i + 1][0]) - crop_size: int(position[i + 1][0]) + crop_size,
                                  int(position[i + 1][1]) - crop_size: int(position[i + 1][1]) + crop_size]
                        cv2.imwrite(self.record_dir + "/ImputedSpot/" + crop_name + ".png", cropped)
                        crop_name = position[i] + '_2'
                        cropped = img[int(position[i + 2][0]) - crop_size: int(position[i + 2][0]) + crop_size,
                                  int(position[i + 2][1]) - crop_size: int(position[i + 2][1]) + crop_size]
                        cv2.imwrite(self.record_dir + "/ImputedSpot/" + crop_name + ".png", cropped)
                        crop_name = position[i] + '_3'
                        cropped = img[int(position[i + 3][0]) - crop_size: int(position[i + 3][0]) + crop_size,
                                  int(position[i + 3][1]) - crop_size: int(position[i + 3][1]) + crop_size]
                        cv2.imwrite(self.record_dir + "/ImputedSpot/" + crop_name + ".png", cropped)
                    except Exception as e:
                        pass
        else:
            print("ERROR :The type of SpotImage doesn't exist!")



    def PriorSpot_generator(self, prelabel_path):

        """
        Generator of prior spot.

        Parameters
        ----------
        prelabel_path
            Path of prelabel files.
        """

        if not os.path.exists(self.record_dir+"/PriorSpot/"):
            os.makedirs(self.record_dir+"/PriorSpot/")

        celltype = []
        with open(prelabel_path) as f:
            for line in f.readlines():
                temp = line.split()
                celltype.append(temp)

        xy_celltype = []
        for each in celltype:
            temp = each
            temp.extend(self.xy_array_dict[self.pixel_xy[each[0]]])
            xy_celltype.append(temp)

        self.prior_spot = xy_celltype
        print("The number of prior spots : ", len(xy_celltype))
        np.savetxt(self.record_dir+"/PriorSpot/prior_spot.txt", xy_celltype, fmt='%s', delimiter='\t')
        self.SpotImage_generator("PriorSpot")

        return xy_celltype



    def RawSpot_generator(self, allprelabel_file):

        """
        Generator of raw spot.

        Parameters
        ----------
        allprelabel_file
            Path of all_pre_label files.
        """

        if not os.path.exists(self.record_dir + "/RawSpot/"):
            os.makedirs(self.record_dir + "/RawSpot/")

        celltype = []
        with open(allprelabel_file) as f:
            for line in f.readlines():
                temp = line.split()
                celltype.append(temp)

        xy_celltype = []
        for each in celltype:
            temp = each
            temp.extend(self.xy_array_dict[self.pixel_xy[each[0]]])
            xy_celltype.append(temp)

        self.raw_position = xy_celltype

        self.SpotImage_generator("RawSpot")

        #return xy_celltype



    def ImputedSpot_generator(self, figsize=(6, 6), xylim=1500, linewidths=0.1, s=8):

        if not os.path.exists(self.record_dir + "/ImputedSpot/"):
            os.makedirs(self.record_dir + "/ImputedSpot/")

        xy_celltype = self.raw_position
        x = [int(k[-1]) for k in xy_celltype]
        y = [int(k[-2]) for k in xy_celltype]

        fill_array = {}
        for each in self.xy_array:
            temp1 = [each[1] + self.gap_length/2, each[2]]
            temp2 = [each[1], each[2] + self.gap_length/2]
            temp3 = [each[1] + self.gap_length/2, each[2] + self.gap_length/2]
            fill_array[each[0]] = [temp1, temp2, temp3]

        ###
        fig = plt.figure(figsize=figsize)
        ax = SubplotZero(fig, 1, 1, 1)
        fig.add_subplot(ax)
        ax.set_xlim(left=0, right=xylim)
        ax.set_ylim(bottom=xylim, top=0)

        for each in self.xy_array_dict.values():
            plt.scatter(each[1], each[0], facecolors='none', marker='s', edgecolors='black', linewidths=linewidths, s=s)

        for each in fill_array.values():
            plt.scatter(each[0][1], each[0][0], facecolors='none', marker='s', edgecolors='r', linewidths=linewidths, s=s)
            plt.scatter(each[1][1], each[1][0], facecolors='none', marker='s', edgecolors='r', linewidths=linewidths, s=s)
            plt.scatter(each[2][1], each[2][0], facecolors='none', marker='s', edgecolors='r', linewidths=linewidths, s=s)

        plt.savefig(self.record_dir + "/ImputedSpot/fill_all_check.pdf")

        ###
        fig = plt.figure(figsize=figsize)
        ax = SubplotZero(fig, 1, 1, 1)
        fig.add_subplot(ax)
        ax.set_xlim(left=0, right=xylim)
        ax.set_ylim(bottom=xylim, top=0)

        Image = imgplt.imread(self.rawimage)
        plt.imshow(Image)

        for i in range(len(x)):
            plt.scatter(x[i], y[i], facecolors='none', marker='s', edgecolors='black', linewidths=linewidths, s=s)

        for each in xy_celltype:
            array_index = self.pixel_xy[each[0]]
            plt.scatter(fill_array[array_index][0][1], fill_array[array_index][0][0], facecolors='none', marker='s', edgecolors='r', linewidths=linewidths, s=s)
            plt.scatter(fill_array[array_index][1][1], fill_array[array_index][1][0], facecolors='none', marker='s', edgecolors='r', linewidths=linewidths, s=s)
            plt.scatter(fill_array[array_index][2][1], fill_array[array_index][2][0], facecolors='none', marker='s', edgecolors='r', linewidths=linewidths, s=s)

        plt.savefig(self.record_dir + "/imputed_check.pdf")

        fill_list = []
        for each in xy_celltype:
            array_index = self.pixel_xy[each[0]]
            fill_list.append(each[0])
            fill_list.append(fill_array[array_index][0])
            fill_list.append(fill_array[array_index][1])
            fill_list.append(fill_array[array_index][2])

        self.imputed_spot = fill_list
        print("The number of imputed spots : ", len(fill_list)*(3/4))

        fill_full_list = []
        for each in fill_array.keys():
            temp1 = [each + '_1']
            temp1.extend(fill_array[each][0])
            temp2 = [each + '_2']
            temp2.extend(fill_array[each][1])
            temp3 = [each + '_3']
            temp3.extend(fill_array[each][2])
            fill_full_list.append(temp1)
            fill_full_list.append(temp2)
            fill_full_list.append(temp3)

        np.savetxt(self.record_dir + "/fill_full_list.txt", fill_full_list, fmt='%s', delimiter='\t')

        self.SpotImage_generator("ImputedSpot")





class Shuffler():

    def __init__(self, record_dir):

        """
        Shuffler of spot images.

        Parameters
        ----------
        record_dir
            Path of the input folder.
        """

        self.name = "Shuffler"
        self.record_dir = record_dir



    def Dataset_divider_downsampling(self, dir_list, downsample_number=500, min_num=25, shuffle_scale=0.2):

        """
        Divider of spot image dataset.

        Parameters
        ----------
        dir_list
            list of the cell classes.
        shuffle_scale
            Shuffle scale between training and testing sets.
        """

        Dir = dir_list

        os.makedirs(self.record_dir + "/train/")
        os.makedirs(self.record_dir + "/test/")
        for each in Dir:
            os.makedirs(self.record_dir + "/train/" + each)
        os.makedirs(self.record_dir + "/more/")
        os.makedirs(self.record_dir + "/del/")

        imgsLib = []
        number = []
        for n in Dir:
            imgsLib.extend(glob.glob(os.path.join(self.record_dir, n + "*.png")))
            number.append(len(imgsLib))
            for each in imgsLib:
                dirpath = self.record_dir + "/train/" + n + "/"
                shutil.move(each, dirpath)
            imgsLib = []

        downsample_number = downsample_number
        min_num = min_num
        dowsample_dir = []
        min_dif = []
        for i in range(len(number)):
            if number[i] > downsample_number:
                celldir = Dir[i]
                imgsLib = []
                imgsLib.extend(glob.glob(os.path.join(self.record_dir + "/train/" + celldir + "/", "*.png")))
                np.random.shuffle(imgsLib)
                for j in range(len(imgsLib) - downsample_number):
                    scrpath = imgsLib[j]
                    dispath = self.record_dir + "/more/"
                    shutil.move(scrpath, dispath)
                dowsample_dir.append(celldir)
            if number[i] < min_num:
                celldir = Dir[i]
                shutil.move(self.record_dir + "/train/" + celldir, self.record_dir + "/del/")
                min_dif.append(celldir)
        print("The downsampling dirs are : ")
        print(dowsample_dir)
        print("The rare cell dirs are : ")
        print(min_dif)

        Dir = list(set(Dir)-set(min_dif))
        
        for each in Dir:
            os.makedirs(self.record_dir + "/test/" + each)

        imgsLib = []
        for n in Dir:
            imgsLib.extend(glob.glob(os.path.join(self.record_dir + "/train/" + n + "/", "*.png")))
            np.random.shuffle(imgsLib)
            num = int(len(imgsLib) * 0.2)
            for i in range(len(imgsLib[:num])):
                scrpath = imgsLib[i]
                dispath = self.record_dir + "/test/" + n + "/"
                shutil.move(scrpath, dispath)
            imgsLib = []

        return Dir

