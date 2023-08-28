import matplotlib.pyplot as plt
from mpl_toolkits.axisartist.axislines import SubplotZero
from sklearn.metrics import confusion_matrix
from skimage.metrics import structural_similarity as ssim
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import cv2
import numpy as np
import itertools
import json
import os
import glob
import matplotlib.image as imgplt





class Metric():

    def __init__(self):

        """
        Metrics.

        """

        self.name = "Metric"



    def Loss_Accuracy_Curve(self, input_file, output_path):

        """
        Visualization of loss curve and accuracy curve.

        Parameters
        ----------
        input_file
            Path of input files.
        output_path
            Path of output files.
        """

        with open(input_file) as f:
            loss = list(map(float, f.readlines()[0].split()))
        with open(input_file) as f:
            val_loss = list(map(float, f.readlines()[1].split()))
        with open(input_file) as f:
            accuracy = list(map(float, f.readlines()[2].split()))
        with open(input_file) as f:
            val_accuracy = list(map(float, f.readlines()[3].split()))

        x = range(1, len(loss)+1)

        plt.figure(figsize=(12, 5))

        # Loss Curve
        plt.subplot(1, 2, 1)
        plt.xlim(0, len(loss)+1)
        plt.ylim(0, max(max(loss), max(val_loss))+1)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xticks(np.arange(0, len(loss)+1, 10))

        plt.plot(x, loss, c='#228B22', linestyle=':', linewidth=1, label='Train')
        plt.scatter(x, loss, c='#228B22', s=1, marker='o')
        plt.text(x[-1]+1, loss[-1], '%.2f' % loss[-1], ha='center', va='bottom', fontsize=5)

        plt.plot(x, val_loss, c='#1874CD', linestyle=':', linewidth=1, label='Test')
        plt.scatter(x, val_loss, c='#1874CD', s=1, marker='o')
        plt.text(x[-1], val_loss[-1], '%.2f' % val_loss[-1], ha='center', va='bottom', fontsize=5)

        plt.legend(loc='upper right', fontsize=5)

        # Accuracy Curve
        plt.subplot(1, 2, 2)
        plt.xlim(0, len(accuracy) + 1)
        plt.ylim(0, 1)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.xticks(np.arange(0, len(accuracy) + 1, 10))

        plt.plot(x, accuracy, c='#EE7600', linestyle=':', linewidth=1, label='Train')
        plt.scatter(x, accuracy, c='#EE7600', s=1, marker='o')
        plt.text(x[-1] + 1, accuracy[-1] + 0.02, '%.2f' % (accuracy[-1] * 100) + "%", ha='center', va='bottom', fontsize=5)

        plt.plot(x, val_accuracy, c='#CD2626', linestyle=':', linewidth=1, label='Test')
        plt.scatter(x, val_accuracy, c='#CD2626', s=1, marker='o')
        plt.text(x[-1], val_accuracy[-1] - 0.06, '%.2f' % (val_accuracy[-1] * 100) + "%", ha='center', va='bottom', fontsize=5)

        plt.legend(loc='lower left', fontsize=5)

        plt.savefig(output_path + "/Accr-Loss.pdf")



    def plot_confusion_matrix(self, cm, classes, output_fig):

        """
        Plotting confusion matrix.

        Parameters
        ----------
        cm
            Inputted confusion matrix.
        classes
            List of predicted classes.
        output_fig
            Path of output files.
        """
        
        plt.figure(figsize=(5, 5))

        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
        plt.title('The Recall Value of Classes')
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")
            #plt.text(j, i + 0.3, '(' + '{:,}'.format(cm[i, j]) + ')', horizontalalignment="center", color="black", fontsize=6)
        plt.tight_layout()
        plt.ylabel('Truth')
        plt.xlabel('Prediction')
        plt.savefig(output_fig)



    def Confusion_Matrix(self, image_path, predicted_path, label, output_fig):

        """
        Visualization of confusion matrix.

        Parameters
        ----------
        image_path
            Path of image folder.
        predicted_path
            Path of predicted files.
        label
            List of predicted labels.
        output_fig
            Path of output files.
        """

        imgsLib = []
        for n in label:
            imgsLib.extend(glob.glob(os.path.join(image_path + "/" + n + "/", "*.png")))

        Labels = {}
        for each in imgsLib:
            Labels[each.split("/")[-1].split(".")[0].split("-")[1]+'-'+each.split("/")[-1].split(".")[0].split("-")[2]] = [each.split("/")[-1].split("-")[0]]

        with open(predicted_path) as f:
            for line in f.readlines():
                temp = line.split()
                if temp[0] in Labels.keys():
                    Labels[temp[0]].append(temp[2])

        for each in Labels:
            if len(Labels[each]) != 2:
                print("Error!")
                print(str(each))


        label_true, label_pred = [], []
        for each in Labels.keys():
            label_true.append(Labels[each][0])
            label_pred.append(Labels[each][1])

        cm = confusion_matrix(label_true, label_pred, labels=label)
        self.plot_confusion_matrix(cm, label, output_fig+"/Confusion_Matrix.pdf")



    def ROC_Curve(self, image_path, predicted_path, labels, color, output_fig):

        """
        Visualization of ROC curve.

        Parameters
        ----------
        image_path
            Path of image folder.
        predicted_path
            Path of predicted files with detailed probabilities.
        labels
            List of predicted labels.
        color
            List of colors for labels.
        output_fig
            Path of output files.
        """

        digtal_label, label_digtal = {}, {}
        labels.sort()
        for i in range(len(labels)):
            digtal_label[i] = labels[i]
            label_digtal[labels[i]] = i

        imgsLib = []
        for n in labels:
            imgsLib.extend(glob.glob(os.path.join(image_path + "/" + n + "/", "*.png")))

        Label_true = {}
        for each in imgsLib:
            Label_true[each.split("/")[-1].split(".")[0].split("-")[1] + '-' + each.split("/")[-1].split(".")[0].split("-")[2]] = [each.split("/")[-1].split("-")[0]]

        label_true_digtal = []
        for each in Label_true:
            label_true_digtal.append(label_digtal[Label_true[each][0]])

        y_test = label_binarize(np.array(label_true_digtal), classes=list(range(0, len(labels))))
        n_classes = y_test.shape[1]

        with open(predicted_path) as f:
            for line in f.readlines():
                temp = line.split()
                if temp[0] in Label_true.keys():
                    temp[1:-2] = [float(k) for k in temp[1:-2]]
                    Label_true[temp[0]].extend(temp[1:-2])

        p_type_detail = []
        for each in Label_true:
            p_type_detail.append(Label_true[each][1:])

        y_score = np.array(p_type_detail)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        lw = 2
        plt.figure(figsize=(5, 5))

        colors = cycle(color)
        for i, color, label in zip(range(n_classes), colors, cycle(list(label_digtal.keys()))):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=label + '(AUC = {1:0.3f})' ''.format(i, roc_auc[i]))

        plt.plot([0, 1], [0, 1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right", fontsize=7)
        plt.savefig(output_fig)



    def SSIM(self, imfil1, imfil2):

        """
        Calculation of SSIM.

        Parameters
        ----------
        imfil1
            Path of image 1.
        imfil2
            Path of image 2.
        """

        img1 = cv2.imread(imfil1)
        (h, w) = img1.shape[:2]
        img2 = cv2.imread(imfil2)
        resized = cv2.resize(img2, (w, h))
        (h1, w1) = resized.shape[:2]
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        return ssim(img1, img2, multichannel=True)





class Check():

    def __init__(self):

        """
        Visualization of pre-labelling or imputing.

        """

        self.name = "Check"



    def Check_PriorSpot(self, adjust_raw_spot, prior_spot, image_path, label_color, output_path, pointsize=5, back_image=False):

        """
        Visualization of pre-labelling.

        Parameters
        ----------
        adjust_raw_spot
            Path of the adjusted raw spot file.
        prior_spot
            Path of the prior spot file.
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
        with open(adjust_raw_spot) as f:
            for line in f.readlines():
                temp = line.split()
                rawspot[temp[-1]] = temp[0:2]

        priorspot = {}
        select = []
        with open(prior_spot) as f:
            for line in f.readlines():
                temp = line.split()
                priorspot[temp[0]] = [temp[-1], temp[1], temp[2]]
                select.append(temp)

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
            plt.scatter(float(rawspot[each][0]), float(rawspot[each][1]), c='white', marker='o', s=pointsize+2, edgecolors='black', linewidths=0.1)
        for each in priorspot.keys():
            plt.scatter(float(priorspot[each][1]), float(priorspot[each][2]), c=colors[priorspot[each][0]], marker='o', s=pointsize)
        plt.savefig(output_path + "/Check_PriorSpot.pdf")

        selectnum = list(set([x[-1] for x in select]))
        Label_selectnum = {}
        for each in selectnum:
            Label_selectnum[each] = 0
        for each in select:
            Label_selectnum[each[-1]] = Label_selectnum[each[-1]] + 1
        info_select = json.dumps(Label_selectnum, sort_keys=False)

        f = open(output_path + "stat.txt", "w")
        f.write(info_select)
        f.close()



    def Check_ImputedSpot(self, adjust_raw_spot, imputed_spot, image_path, output_path, pointsize=5, back_image=False):

        """
        Visualization of imputing.

        Parameters
        ----------
        adjust_raw_spot
            Path of the adjusted raw spot file.
        imputed_spot
            Path of the imputed spot file.
        image_path
            Path of the original H&E staining image.
        output_path
            Path of the output files.
        pointsize
            Size of scatter points.
        back_image
            Shown of the original H&E staining image.
        """

        Image = imgplt.imread(image_path)
        h, w, _ = Image.shape

        rawspot = {}
        with open(adjust_raw_spot) as f:
            for line in f.readlines():
                temp = line.split()
                rawspot[temp[-1]] = temp[0:2]

        imputedspot = {}
        with open(imputed_spot) as f:
            for line in f.readlines():
                temp = line.split()
                imputedspot[temp[-1]] = temp[0:2]

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
            plt.scatter(float(rawspot[each][0]), float(rawspot[each][1]), c='red', marker='o', s=pointsize)
        for each in imputedspot.keys():
            plt.scatter(float(imputedspot[each][0]), float(imputedspot[each][1]), c='blue', marker='o', s=pointsize)
        plt.savefig(output_path + "/Check_ImputedSpot.pdf")

