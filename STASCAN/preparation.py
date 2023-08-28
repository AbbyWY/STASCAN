from pylab import *
import numpy as np
import cv2
import os
import random
import glob
import shutil
from PIL import Image





class Generator():

    def __init__(self, record_dir, position, image, crop_size):

        """
        Preparation of spot images.

        Parameters
        ----------
        record_dir
            Path of the output folder.
        position
            The position of raw spots. Standard output file for Space Ranger.
        image
            The original H&E staining images, used for Space Ranger previously.
        crop_size
            Pixel radius of spot images.
        """

        self.name = "Generator"

        self.record_dir = record_dir

        self.position = []
        with open(position) as f:
            for line in f.readlines():
                temp = line.split(',')
                if int(temp[1]) == 1:
                    self.position.append([int(temp[-1].split('\n')[0]), int(temp[-2]), temp[0]])
        self.position.sort(key=lambda x: (x[1], x[0]))
        print("The number of raw spots : ", len(self.position))
        
        self.Data = {}
        for each in self.position:
            self.Data[each[2]] = [each[0], each[1]]

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
            position = self.position
            for each in position:
                k = each[2].replace("\"", "")
                cropped = img[int(each[1]) - crop_size:int(each[1]) + crop_size,
                          int(each[0]) - crop_size:int(each[0]) + crop_size]
                cv2.imwrite(self.record_dir + "/RawSpot/" + k + ".png", cropped)
        elif type == 'PriorSpot':
            position = self.prior_spot
            for each in position:
                k = each[0].replace("\"", "")
                cropped = img[int(each[2]) - crop_size:int(each[2]) + crop_size,
                          int(each[1]) - crop_size:int(each[1]) + crop_size]
                celltype = each[-1]
                cv2.imwrite(self.record_dir + "/PriorSpot/" + celltype + "-" + k + ".png", cropped)
        elif type == 'ImputedSpot':
            position = self.imputed_spot
            for each in position:
                k = each[2].replace("\"", "")
                each[0], each[1] = float(each[0]), float(each[1])
                cropped = img[int(each[1]) - crop_size:int(each[1]) + crop_size,
                          int(each[0]) - crop_size:int(each[0]) + crop_size]
                cv2.imwrite(self.record_dir + "/ImputedSpot/" + k + ".png", cropped)
        else:
            print("ERROR :The type of SpotImage doesn't exist!")



    def SubSpotImage_generator(self, type):

        """
        Generator of subdivided spot images.

        Parameters
        ----------
        type: {'raw_divided', 'prior_divided'}
            Type of spots.
        """

        imgsLib = []
        subloc = []
        position = self.Data

        if type == 'raw_divided':
            img_path = self.record_dir + "/RawSpot/"
            save_path = self.record_dir + "/SubSpot/raw_divided/"
            #position = self.Data
        if type == 'prior_divided':
            img_path = self.record_dir + "/PriorSpot/"
            save_path = self.record_dir + "/SubSpot/prior_divided/"
            #position = {}
            #if self.prior_spot == True:
            #    for each in self.prior_spot:
            #        position[each[0]] = [int(each[1]), int(each[2])]
            #else:
            #with open(self.record_dir + "/PriorSpot/prior_spot.txt") as f:
            #    for line in f.readlines():
            #        temp = line.split()
            #        position[temp[0]] = [float(temp[1]), float(temp[2])]          
        
        

        imgsLib.extend(glob.glob(os.path.join(img_path, "*.png")))

        for each in imgsLib:
            # img
            image = Image.open(each)
            img = image.convert('RGB')
            w = img.size[0]
            h = img.size[1]
            rawname = each.split("/")[-1].split(".")[0]
            img_1 = img.crop([0, 0, w / 2, h / 2])
            img_1.save(save_path + rawname + "_1.png")
            img_2 = img.crop([w / 2, 0, w, h / 2])
            img_2.save(save_path + rawname + "_2.png")
            img_3 = img.crop([0, h / 2, w / 2, h])
            img_3.save(save_path + rawname + "_3.png")
            img_4 = img.crop([w / 2, h / 2, w, h])
            img_4.save(save_path + rawname + "_4.png")
            # loc
            rawname = each.split("/")[-1].split(".")[0].split("-")[-2] + "-" + each.split("/")[-1].split(".")[0].split("-")[-1]
            rawxy = position[rawname]
            subloc_1 = [rawxy[0] - self.crop_size / 2, rawxy[1] - self.crop_size / 2, rawname + '_1']
            subloc.append(subloc_1)
            subloc_2 = [rawxy[0] + self.crop_size / 2, rawxy[1] - self.crop_size / 2, rawname + '_2']
            subloc.append(subloc_2)
            subloc_3 = [rawxy[0] - self.crop_size / 2, rawxy[1] + self.crop_size / 2, rawname + '_3']
            subloc.append(subloc_3)
            subloc_4 = [rawxy[0] + self.crop_size / 2, rawxy[1] + self.crop_size / 2, rawname + '_4']
            subloc.append(subloc_4)

        newsubloc = np.array(subloc)
        np.savetxt(save_path + "subloc.txt", newsubloc, fmt='%s', delimiter='\t')



    def PriorSpot_generator(self, prelabel_path, strategies='joint', threshold_proportion = 0.6, threshold_ration = 1.5):

        """
        Generator of prior spot.

        Parameters
        ----------
        prelabel_path
            Path of prelabel files.
        strategies: {'joint', 'single_proportion', 'single_ration'}
            Type of strategies to label the cell types for each spot.
        threshold_proportion
            The threshold of prior spot selection. Spots which the proportion of dominant cell types reached the threshold were selected as prior spots.
        threshold_ration
            The threshold of prior spot selection. Spots which the proportion of dominant cell types exceeded the given multiple of the proportion of secondary cell types were selected as prior spots.
        """

        self.prelabel_strategies = strategies
        self.threshold_proportion = threshold_proportion
        self.threshold_ration = threshold_ration
        print("The strategy of pre-labelling is: " + str(self.prelabel_strategies))

        if not os.path.exists(self.record_dir+"/PriorSpot/"):
            os.makedirs(self.record_dir+"/PriorSpot/")

        #self.Data = {}
        #for each in self.position:
        #    self.Data[each[2]] = [each[0], each[1]]

        spot_C = {}
        with open(prelabel_path + "Cell2location_prelabelling.txt") as f:
            for line in f.readlines():
                temp = line.split()
                spot_C[temp[0]] = temp[1:]
        with open(prelabel_path + "Cell2location_prelabelling.txt") as f:
            Label_C = f.readlines()[0].split()

        prior_spot = []

        if self.prelabel_strategies == 'joint':
            spot_S = {}
            with open(prelabel_path + "Seurat_prelabelling.txt") as f:
                for line in f.readlines():
                    temp = line.split()
                    spot_S[temp[0]] = temp[1:]
            with open(prelabel_path + "Seurat_prelabelling.txt") as f:
                Label_S = f.readlines()[0].split()

            for each in self.Data.keys():
                if each in spot_C.keys() and each in spot_S.keys():
                    value_C = [float(k) for k in spot_C[each]]
                    maxtype_C = Label_C[value_C.index(max(value_C))]
                    value_S = [float(k) for k in spot_S[each]]
                    maxtype_S = Label_S[value_S.index(max(value_S))]
                    if maxtype_C == maxtype_S:
                        temp = [each]
                        temp.extend(self.Data[each])
                        temp.extend([maxtype_C, maxtype_S])
                        prior_spot.append(temp)

        if self.prelabel_strategies == 'single_proportion':
            print("The threshold of cell proportion is: " + str(threshold_proportion))
            for each in self.Data.keys():
                if each in spot_C.keys():
                    value_C = [float(k) for k in spot_C[each]]
                    maxtype_C = Label_C[value_C.index(max(value_C))]
                    if max(value_C) / sum(value_C) > threshold_proportion:
                        temp = [each]
                        temp.extend(self.Data[each])
                        temp.extend([maxtype_C, maxtype_C])
                        prior_spot.append(temp)

        if self.prelabel_strategies == 'single_ration':
            print("The threshold of cell ration is: " + str(threshold_ration))
            for each in self.Data.keys():
                if each in spot_C.keys():
                    value_C = [float(k) for k in spot_C[each]]
                    ctype = Label_C[value_C.index(max(value_C))]
                    value_C.sort()
                    if value_C[-1] > value_C[-2] * threshold_ration:
                        temp = [each]
                        temp.extend(self.Data[each])
                        temp.extend([ctype, ctype])
                        prior_spot.append(temp)

        self.prior_spot = prior_spot
        print("The number of prior spots : ", len(prior_spot))
        np.savetxt(self.record_dir+"/PriorSpot/prior_spot.txt", prior_spot, fmt='%s', delimiter='\t')
        self.SpotImage_generator("PriorSpot")

        return prior_spot



    def RawSpot_generator(self):

        if not os.path.exists(self.record_dir + "/RawSpot/"):
            os.makedirs(self.record_dir + "/RawSpot/")

        self.SpotImage_generator("RawSpot")



    def SubSpot_generator(self):

        if not os.path.exists(self.record_dir + "/SubSpot/"):
            os.makedirs(self.record_dir + "/SubSpot/")
        if not os.path.exists(self.record_dir + "/SubSpot/raw_divided/"):
            os.makedirs(self.record_dir + "/SubSpot/raw_divided/")
        if not os.path.exists(self.record_dir + "/SubSpot/prior_divided/"):
            os.makedirs(self.record_dir + "/SubSpot/prior_divided")

        self.SubSpotImage_generator("raw_divided")
        self.SubSpotImage_generator("prior_divided")



    def ImputedSpot_generator(self):

        if not os.path.exists(self.record_dir + "/ImputedSpot/"):
            os.makedirs(self.record_dir + "/ImputedSpot/")

        all_s = []
        sess = []
        s = int(self.position[0][1])

        for i in range(len(self.position)):
            if self.position[i][1] == s:
                sess.append(self.position[i])
            else:
                if int(self.position[i][1]) < s + self.crop_size:
                    sess.append(self.position[i])
                else:
                    maxv = int(self.position[i - 1][1])
                    averv = (s + maxv) / 2
                    for j in range(len(sess)):
                        sess[j][1] = averv
                    all_s.append(sess)
                    sess = []
                    s = self.position[i][1]
                    sess.append(self.position[i])

        maxv = int(self.position[-1][1])
        averv = (s + maxv) / 2
        for j in range(len(sess)):
            sess[j][1] = averv
        all_s.append(sess)

        for each in all_s:
            each.sort(key=lambda x: (x[0]))

        alls = []
        for i in range(len(all_s)):
            for j in range(len(all_s[i])):
                alls.append(all_s[i][j])

        np.savetxt(self.record_dir + "/ImputedSpot/adjust_raw_spot.txt", alls, fmt='%s', delimiter='\t')

        fill = []

        for i in range(len(all_s)):
            for j in range(len(all_s[i]) - 1):
                a = int(all_s[i][j + 1][0]) - int(all_s[i][j][0])
                nx = int(all_s[i][j][0]) + a / 2
                ny = int(all_s[i][j][1]) + int(a / ((3 ** 0.5) * 2))
                nl = 'new-down' + all_s[i][j][2]
                fill.append(nx)
                fill.append(ny)
                fill.append(nl)
                ny = int(all_s[i][j][1]) - int(a / ((3 ** 0.5) * 2))
                nl = 'new-up' + all_s[i][j][2]
                fill.append(nx)
                fill.append(ny)
                fill.append(nl)

        newfill = np.array(fill)
        newfill = newfill.reshape(int(len(newfill) / 3), 3)

        np.savetxt(self.record_dir + "/ImputedSpot/imputed_spot.txt", newfill, fmt='%s', delimiter='\t')
        print("The number of imputed spots : ", len(newfill))

        self.imputed_spot = newfill.tolist()
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



    def Dataset_divider(self, input_spot, shuffle_scale=0.2):

        """
        Divider of spot image dataset.

        Parameters
        ----------
        input_spot
            Spots in the dataset, which used to determine cell classes.
        shuffle_scale
            Shuffle scale between training and testing sets.
        """

        Dir = list(set([x[-1] for x in input_spot]))

        os.makedirs(self.record_dir + "/train/")
        os.makedirs(self.record_dir + "/test/")
        for each in Dir:
            os.makedirs(self.record_dir + "/train/" + each)
            os.makedirs(self.record_dir + "/test/" + each)

        imgsLib = []
        for n in Dir:
            imgsLib.extend(glob.glob(os.path.join(self.record_dir, n + "*.png")))
            for each in imgsLib:
                dirpath = self.record_dir + "/train/" + n + "/"
                shutil.move(each, dirpath)
            imgsLib = []

        imgsLib = []
        for n in Dir:
            imgsLib.extend(glob.glob(os.path.join(self.record_dir + "/train/" + n + "/", "*.png")))
            np.random.shuffle(imgsLib)
            num = int(len(imgsLib) * shuffle_scale)
            for i in range(len(imgsLib[:num])):
                scrpath = imgsLib[i]
                dispath = self.record_dir + "/test/" + n + "/"
                shutil.move(scrpath, dispath)
            imgsLib = []



    def KCross_divider(self, K_group, output_path):

        """
        Divider of K-fold cross image dataset.

        Parameters
        ----------
        K_group
            Number of fold cross.
        output_path
            Path of output folders.
        """

        rawimg = []
        rawimg.extend(glob.glob(os.path.join(self.record_dir, "*.png")))

        allcelltype = []
        for each in rawimg:
            allcelltype.append(each.split("/")[-1].split("-")[0])

        Dir = list(set(allcelltype))

        for i in range(K_group):
            os.makedirs(output_path + "/group" + str(i+1) + "/")
            os.makedirs(output_path + "/test" + str(i+1) + "/")
            for each in rawimg:
                shutil.copy(each, output_path + "/group" + str(i+1) + "/")
            imgsLib = []
            for n in Dir:
                os.makedirs(output_path + "/group" + str(i+1) + "/" + n + "/")
                os.makedirs(output_path + "/test" + str(i+1) + "/" + n + "/")
                imgsLib.extend(glob.glob(os.path.join(output_path + "/group" + str(i+1) + "/", n + "*.png")))
                for m in imgsLib:
                    shutil.move(m, output_path + "/group" + str(i+1) + "/" + n + "/")
                imgsLib = []
                imgsLib.extend(glob.glob(os.path.join(output_path + "/group" + str(i+1) + "/" + n + "/", "*.png")))
                np.random.shuffle(imgsLib)
                num = int(len(imgsLib)/K_group)
                for j in range(len(imgsLib[:num])):
                    shutil.move(imgsLib[j], output_path + "/test" + str(i+1) + "/" + n + "/")
                imgsLib = []





class Detection():

    def __init__(self):

        """
        Detection of spot images.

        """

        self.name = "Detection"



    def DetectWhiteRegion(self, input_path, grey_level, threshold):

        """
        Detection of spot images with lagre white regions.

        Parameters
        ----------
        input_path
            Path of input images.
        grey_level
            Grey-level threshold of the image.
        threshold
            Threshold of white regions.
        """
        
        imgsLib = []
        imgsLib.extend(glob.glob(os.path.join(input_path, "*.png")))
        
        if not os.path.exists(input_path + "/whiteimg/"):
            os.makedirs(input_path + "/whiteimg/")


        for each in imgsLib:
            img = Image.open(each)
            imgG = img.convert('L')

            rows = imgG.size[0]
            cols = imgG.size[1]

            q = 0
            for i in range(rows):
                for j in range(cols):
                    if imgG.getpixel((i, j)) > grey_level:
                        q += 1
                        
            if q/(rows*cols) > threshold:
                shutil.move(each, input_path + "/whiteimg/")