import os
import glob
import shutil
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import SubplotZero
import matplotlib.image as imgplt
#%matplotlib inline



import STASCAN.preparation as preparation
import STASCAN.model as model






class Module():

    def __init__(self):
        self.name = "Module"



    def UnseenSpot(self, result_dir, position, image, crop_size, prelabel_path,
                   strategies='joint', threshold_proportion=0.6, threshold_ration=1.5,
                   shuffle_scale=0.2, lr=1e-3, epochs=50, model_name="base_model.h5"):

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
        prelabel_path
            Path of prelabel files.
        strategies: {'joint', 'single_proportion', 'single_ration'}
            Type of strategies to label the cell types for each spot.
        threshold_proportion
            The threshold of prior spot selection. Spots which the proportion of dominant cell types reached the threshold were selected as prior spots.
        threshold_ration
            The threshold of prior spot selection. Spots which the proportion of dominant cell types exceeded the given multiple of the proportion of secondary cell types were selected as prior spots.
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
        step1 = preparation.Generator(result_dir, position, image, crop_size)
        prior_spot = step1.PriorSpot_generator(prelabel_path, strategies, threshold_proportion, threshold_ration)
        step1.RawSpot_generator()
        step1.ImputedSpot_generator()

        step2 = preparation.Shuffler(result_dir + "/PriorSpot/")
        step2.Dataset_divider(prior_spot, shuffle_scale)

        # Training
        step3 = model.BuildModel()
        traindata_path, testdata_path = result_dir + "/PriorSpot/train/", result_dir + "/PriorSpot/test/"
        n_class = len(set([x[-1] for x in prior_spot]))
        step3.base_model(traindata_path, testdata_path, result_dir, n_class, lr, epochs, model_name)

        # Prediction
        step4 = model.BuildPrediction(result_dir + "/Models/" + model_name, result_dir)
        step4.prediction(prior_spot, result_dir + "/ImputedSpot", result_dir + "/ImputedSpot/imputed_spot.txt", "Imputed")
        step4.prediction(prior_spot, result_dir + "/RawSpot", result_dir + "/ImputedSpot/adjust_raw_spot.txt", "Raw")



    def SubdividedSpot(self, result_dir, position, image, crop_size, prelabel_path,
                       strategies='single_proportion', threshold_proportion=0.2, threshold_ration=1.5,
                       shuffle_scale=0.2, lr=1e-3, epochs=50, model_name="base_model.h5"):

        """
        Cell annotation for subdivided spots.

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
        prelabel_path
            Path of prelabel files.
        strategies: {'joint', 'single_proportion', 'single_ration'}
            Type of strategies to label the cell types for each spot.
        threshold_proportion
            The threshold of prior spot selection. Spots which the proportion of dominant cell types reached the threshold were selected as prior spots.
        threshold_ration
            The threshold of prior spot selection. Spots which the proportion of dominant cell types exceeded the given multiple of the proportion of secondary cell types were selected as prior spots.
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
        step1 = preparation.Generator(result_dir, position, image, crop_size)
        prior_spot = step1.PriorSpot_generator(prelabel_path, strategies, threshold_proportion, threshold_ration)
        step1.RawSpot_generator()
        step1.SubSpot_generator()

        step2 = preparation.Shuffler(result_dir + "/SubSpot/prior_divided/")
        step2.Dataset_divider(prior_spot, shuffle_scale)

        # Training
        step3 = model.BuildModel()
        traindata_path, testdata_path = result_dir + "/SubSpot/prior_divided/train/", result_dir + "/SubSpot/prior_divided/test/"
        n_class = len(set([x[-1] for x in prior_spot]))
        step3.base_model(traindata_path, testdata_path, result_dir, n_class, lr, epochs, model_name)

        # Prediction
        step4 = model.BuildPrediction(result_dir + "/Models/" + model_name, result_dir)
        step4.prediction(prior_spot, result_dir + "/SubSpot/raw_divided/", result_dir + "/SubSpot/raw_divided/subloc.txt", "Subdivided")



    def UnseenSection(self, result_dir, position, image, crop_size, model_path, input_spot):

        """
        Cell annotation for unseen sections.

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
        model_path
            Path of the trained model used for prediction.
        input_spot
            Spots in the dataset, which used to determine labels.
        """

        # Prepatation
        step1 = preparation.Generator(result_dir, position, image, crop_size)
        step1.RawSpot_generator()
        step1.ImputedSpot_generator()

        # Prediction
        step2 = model.BuildPrediction(model_path, result_dir)
        step2.prediction(input_spot, result_dir + "/ImputedSpot", result_dir + "/ImputedSpot/imputed_spot.txt", "Adjacent_imputed")
        step2.prediction(input_spot, result_dir + "/RawSpot", result_dir + "/ImputedSpot/adjust_raw_spot.txt", "Adjacent_raw")





class Optional():

    def __init__(self):

        self.name = "Optional"



    def SectionSpecificTraining(self, output_path, sections_path, imagetype, crop_size,
                                strategies='joint', threshold_proportion=0.2, threshold_ration=1.5, shuffle_scale=0.2,
                                lr1=1e-3, epoch1=50, lr2=1e-4, epoch2=50):

        """
        Section-specific training.

        Parameters
        ----------
        sections_path
            Path of the section folders.
        imagetype
            The type of original H&E staining images.
        crop_size
            Pixel radius of spot images.
        strategies: {'joint', 'single_proportion', 'single_ration'}
            Type of strategies to label the cell types for each spot.
        threshold_proportion
            The threshold of prior spot selection. Spots which the proportion of dominant cell types reached the threshold were selected as prior spots.
        threshold_ration
            The threshold of prior spot selection. Spots which the proportion of dominant cell types exceeded the given multiple of the proportion of secondary cell types were selected as prior spots.
        shuffle_scale
            Shuffle scale between training and testing sets.
        """
      
        allsection = []
        for each in os.listdir(sections_path):
            allsection.append(each)
            if not os.path.exists(output_path + each):
                os.makedirs(output_path + each)
                #os.makedirs(output_path + each + "/PriorSpot/")
        
        if not os.path.exists(output_path + "/PriorSpot/"):
            os.makedirs(output_path + "/PriorSpot/")
                
        allpriorspot = []
        for each in allsection:
            section_number = each.split('section')[-1]
            specific_output = output_path + each + "/"
            section_rawdata = sections_path + "/" + each + "/"
            Section_Generator = preparation.Generator(specific_output, section_rawdata + 'tissue_positions_list.csv', glob.glob(os.path.join(section_rawdata, "*."+imagetype))[0], crop_size)
            specific_prior_spot = Section_Generator.PriorSpot_generator(section_rawdata, strategies, threshold_proportion, threshold_ration)
            allpriorspot.extend(specific_prior_spot)
            imgsLib = []
            imgsLib.extend(glob.glob(os.path.join(specific_output + "/PriorSpot", "*.png")))
            for n in imgsLib:
                shutil.copyfile(n, output_path + "/PriorSpot/" + n.split("/")[-1].split(".")[0] + '_' + str(section_number) + ".png")
            specific_shuffler = preparation.Shuffler(specific_output + "/PriorSpot/")
            specific_shuffler.Dataset_divider(specific_prior_spot, shuffle_scale)

        section_shuffler = preparation.Shuffler(output_path + "/PriorSpot/")
        section_shuffler.Dataset_divider(allpriorspot, shuffle_scale)
        
        all_type = os.listdir(output_path + "/PriorSpot/train/")
        
        for each in allsection:
            for n in all_type:
                if not os.path.exists(output_path + each + "/" + "/PriorSpot/train/" + n):
                    shutil.copytree(output_path + "/PriorSpot/train/" + n + '/', output_path + each + "/" + "/PriorSpot/train/" + n)
                    shutil.copytree(output_path + "/PriorSpot/test/" + n + '/', output_path + each + "/" + "/PriorSpot/test/" + n)
                if not os.listdir(output_path + each + "/" + "/PriorSpot/test/" + n + "/"):
                    os.rmdir(output_path + each + "/" + "/PriorSpot/test/" + n)
                    shutil.copytree(output_path + each + "/" + "/PriorSpot/train/" + n, output_path + each + "/" + "/PriorSpot/test/" + n)
                    
        

        Base_model = model.BuildModel()
        n_class = len(set([x[-1] for x in allpriorspot]))
        Base_model.base_model(output_path + "/PriorSpot/train/", output_path + "/PriorSpot/test/", output_path, n_class, lr1, epoch1, model_name="base_model.h5")

        for each in allsection:
            specific_output = output_path + each + "/"
            specific_model = model.BuildModel()
            specific_model.finetuned_model(output_path + "/Models/base_model.h5", specific_output + "/PriorSpot/train/", specific_output + "/PriorSpot/test/", specific_output, lr2, epoch2, model_name="finetuned_model.h5")



    def PseudoLabelling(self, output_path, input_img, prediction_file, prediction_label, probability=0.90):

        """
        Pseudo labelling.

        Parameters
        ----------
        output_path
            Path of the output folder.
        input_img
            Path of predicted image folders.
        prediction_file
            The prediction files with detailed probabilities of cell types.
        prediction_label
            The label of predicted classes.
        probability
            The threshold of probability to select pseudo labels.
        """

        if not os.path.exists(output_path + "/PseudoSpot/"):
            os.makedirs(output_path + "/PseudoSpot/")

        prediction = []
        with open(prediction_file) as f:
            for line in f.readlines():
                temp = line.split()
                prediction.append(temp)

        pseudo_spot = []
        for each in prediction:
            value = [float(x) for x in each[1:-2]]
            celltype = prediction_label[value.index(max(value))]
            if max(value) >= probability:
                pseudo_spot.append([each[0], each[-2], each[-1], celltype, celltype])
        print("The number of pseudo spots : ", len(pseudo_spot))
        np.savetxt(output_path + "/PseudoSpot/pseudo_spot.txt", pseudo_spot, fmt='%s', delimiter='\t')

        for each in pseudo_spot:
            shutil.copy(os.path.join(input_img, each[0] + ".png"), os.path.join(output_path + "/PseudoSpot/", each[-1] + "-" + each[0] + ".png"))

        return pseudo_spot





class Visualization():

    def __init__(self):

        self.name = "Visualization"



    def EnhancedPlot(self, raw_spot, prior_spot, imputed_spot, image_path, label_color, output_path, pointsize=5, back_image=True):

        """
        Visualization of enhanced cell annotations.

        Parameters
        ----------
        raw_spot
            Path of the predicted raw spot file.
        prior_spot
            Path of the predicted prior spot file.
        imputed_spot
            Path of the predicted imputed spot file.
        image_path
            Path of the original H&E staining image.
        label_color
            Dictionary of label colors.
        output_path
            Path of the output files.
        pointsize
            Size of scatter points.
        back_image
            Shown of the original H&E staining image.
        """

        Image = imgplt.imread(image_path)
        h, w, _ = Image.shape
        colors = label_color

        rawspot = {}
        with open(raw_spot) as f:
            for line in f.readlines():
                temp = line.split()
                rawspot[temp[0]] = temp[2:]

        priorspot = {}
        with open(prior_spot) as f:
            for line in f.readlines():
                temp = line.split()
                priorspot[temp[0]] = [temp[-1], temp[1], temp[2]]

        imputedspot = {}
        with open(imputed_spot) as f:
            for line in f.readlines():
                temp = line.split()
                imputedspot[temp[0]] = temp[2:]

        fig = plt.figure(figsize=(10, 10))
        ax = SubplotZero(fig, 1, 1, 1)
        fig.add_subplot(ax)

        ax.set_xlim(left=0, right=int(max(h, w)))
        ax.set_ylim(bottom=int(max(h, w)), top=0)
        ax.set_xticks([])
        ax.set_yticks([])

        if back_image == True:
            plt.imshow(Image)

        for each in rawspot.keys():
            plt.scatter(float(rawspot[each][1]), float(rawspot[each][2]), c='white', marker='o', s=pointsize+2, edgecolors='black', linewidths=0.1, alpha=0.3)
        for each in priorspot.keys():
            plt.scatter(float(rawspot[each][1]), float(rawspot[each][2]), c=colors[priorspot[each][0]], marker='o', s=pointsize)
        plt.savefig(output_path + "/PriorPlot.pdf")

        for each in rawspot.keys():
            if each not in priorspot.keys():
                plt.scatter(float(rawspot[each][1]), float(rawspot[each][2]), c=colors[rawspot[each][0]], marker='o', s=pointsize)
        for each in imputedspot.keys():
            plt.scatter(float(imputedspot[each][1]), float(imputedspot[each][2]), c=colors[imputedspot[each][0]], marker='o', s=pointsize)
        plt.savefig(output_path + "/EnhancedPlot.pdf")



    def SubResolutionPlot(self, raw_spot, sub_spot, image_path, label_color, output_path, pointsize=5, back_image=True):

        """
        Visualization of subdivided cell annotations.

        Parameters
        ----------
        raw_spot
            Path of the predicted raw spot file.
        sub_spot
            Path of the predicted subdivided spot file.
        image_path
            Path of the original H&E staining image.
        label_color
            Dictionary of label colors.
        output_path
            Path of the output files.
        pointsize
            Size of scatter points.
        back_image
            Shown of the original H&E staining image.
        """

        Image = imgplt.imread(image_path)
        h, w, _ = Image.shape
        colors = label_color

        rawspot = {}
        with open(raw_spot) as f:
            for line in f.readlines():
                temp = line.split()
                rawspot[temp[0]] = temp[2:]

        subspot = {}
        with open(sub_spot) as f:
            for line in f.readlines():
                temp = line.split()
                subspot[temp[0]] = temp[2:]
                
        #%matplotlib inline

        fig = plt.figure(figsize=(10, 10))
        ax = SubplotZero(fig, 1, 1, 1)
        fig.add_subplot(ax)

        ax.set_xlim(left=0, right=int(max(h, w)))
        ax.set_ylim(bottom=int(max(h, w)), top=0)
        ax.set_xticks([])
        ax.set_yticks([])

        if back_image == True:
            plt.imshow(Image)

        for each in rawspot.keys():
            plt.scatter(float(rawspot[each][1]), float(rawspot[each][2]), c=colors[rawspot[each][0]], marker='o', s=pointsize)
        plt.savefig(output_path + "/RawResolution.pdf")
        #plt.show()
        #plt.clf()

        fig = plt.figure(figsize=(10, 10))
        ax = SubplotZero(fig, 1, 1, 1)
        fig.add_subplot(ax)

        ax.set_xlim(left=0, right=int(max(h, w)))
        ax.set_ylim(bottom=int(max(h, w)), top=0)
        ax.set_xticks([])
        ax.set_yticks([])

        if back_image == True:
            plt.imshow(Image)

        for each in subspot.keys():
            plt.scatter(float(subspot[each][1]), float(subspot[each][2]), c=colors[subspot[each][0]], marker='o', s=int(pointsize/2))
        plt.savefig(output_path + "/SubResolutionPlot.pdf")
        #plt.clf()



    def AdjacentSectionPlot(self, raw_spot, imputed_spot, image_path, label_color, output_path, pointsize=5, back_image=True):

        """
        Visualization of adjacent section cell annotations.

        Parameters
        ----------
        raw_spot
            Path of the predicted raw spot file.
        imputed_spot
            Path of the predicted imputed spot file.
        image_path
            Path of the original H&E staining image.
        label_color
            Colors of class labels.
        output_path
            Path of the output files.
        pointsize
            Size of scatter points.
        back_image
            Shown of the original H&E staining image.
        """

        Image = imgplt.imread(image_path)
        h, w, _ = Image.shape
        colors = label_color

        rawspot = {}
        with open(raw_spot) as f:
            for line in f.readlines():
                temp = line.split()
                rawspot[temp[0]] = temp[2:]

        imputedspot = {}
        with open(imputed_spot) as f:
            for line in f.readlines():
                temp = line.split()
                imputedspot[temp[0]] = temp[2:]

        fig = plt.figure(figsize=(10, 10))
        ax = SubplotZero(fig, 1, 1, 1)
        fig.add_subplot(ax)

        ax.set_xlim(left=0, right=int(max(h, w)))
        ax.set_ylim(bottom=int(max(h, w)), top=0)
        ax.set_xticks([])
        ax.set_yticks([])

        if back_image == True:
            plt.imshow(Image)

        for each in rawspot.keys():
            plt.scatter(float(rawspot[each][1]), float(rawspot[each][2]), c=colors[rawspot[each][0]], marker='o', s=pointsize)
        for each in imputedspot.keys():
            plt.scatter(float(imputedspot[each][1]), float(imputedspot[each][2]), c=colors[imputedspot[each][0]], marker='o', s=pointsize)
        plt.savefig(output_path + "/AdjacentSectionPlot.pdf")






