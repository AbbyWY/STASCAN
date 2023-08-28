import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(device=gpu, enable=True)
    
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

from tensorflow.keras.models import load_model
import numpy as np
import os

import itertools
import glob





class BuildModel():

    def __init__(self):

        self.name = "BuildModel"

        self.train_datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1 / 255,
            shear_range=20,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
        )

        self.test_datagen = ImageDataGenerator(
            rescale=1 / 255,
        )

        self.batch_size = 32



    def base_model(self, traindata_path, testdata_path, output_path, n_class, lr=1e-3, epochs=50, model_name="base_model.h5"):

        """
        Build base model.

        Parameters
        ----------
        traindata_path
            Path of training set.
        testdata_path
            Path of testing set.
        output_path
            Path of output files.
        n_class
            Number of cell classes.
        lr
            Learning rate for SGDOptimizer.
        epochs
            Number of total epochs in training.
        model_name
            Name of trained model
        """

        vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        top_model = Sequential()
        top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(n_class, activation='softmax'))
        model = Sequential()
        model.add(vgg16_model)
        model.add(top_model)

        train_generator = self.train_datagen.flow_from_directory(
            traindata_path,
            target_size=(224, 224),
            batch_size=self.batch_size,
        )

        test_generator = self.test_datagen.flow_from_directory(
            testdata_path,
            target_size=(224, 224),
            batch_size=self.batch_size,
        )

        print(train_generator.class_indices)
        print(test_generator.class_indices)


        model.compile(optimizer=SGD(lr=lr, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_generator, epochs=epochs, validation_data=test_generator)

        loss, val_loss, accuracy, val_accuracy = model.history.history['loss'], model.history.history['val_loss'], model.history.history['accuracy'], model.history.history['val_accuracy']

        os.makedirs(output_path + "/Models/")

        log = [loss, val_loss, accuracy, val_accuracy]
        np.savetxt(output_path + "/Models/" + "Log_BaseModel.txt", log, fmt='%s', delimiter='\t')

        model.save(output_path + "/Models/" + model_name)



    def finetuned_model(self, basemodel, traindata_path, testdata_path, output_path, lr=1e-4, epochs=50, model_name="finetuned_model.h5"):

        """
        Build finetuned model.

        Parameters
        ----------
        basemodel
            Path of base model
        traindata_path
            Path of training set.
        testdata_path
            Path of testing set.
        output_path
            Path of output files.
        lr
            Learning rate for SGDOptimizer.
        epochs
            Number of total epochs in training.
        model_name
            Name of trained model
        """

        model = load_model(basemodel)

        train_generator = self.train_datagen.flow_from_directory(
            traindata_path,
            target_size=(224, 224),
            batch_size=self.batch_size,
        )

        test_generator = self.test_datagen.flow_from_directory(
            testdata_path,
            target_size=(224, 224),
            batch_size=self.batch_size,
        )

        print(train_generator.class_indices)
        print(test_generator.class_indices)

        model.compile(optimizer=SGD(lr=lr, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_generator, epochs=epochs, validation_data=test_generator)

        loss, val_loss, accuracy, val_accuracy = model.history.history['loss'], model.history.history['val_loss'], model.history.history['accuracy'], model.history.history['val_accuracy']

        if not os.path.exists(output_path + "/Models/"):
            os.makedirs(output_path + "/Models/")

        log = [loss, val_loss, accuracy, val_accuracy]
        np.savetxt(output_path + "/Models/" + "Log_FinetunedModel.txt", log, fmt='%s', delimiter='\t')

        model.save(output_path + "/Models/" + model_name)



    def alltrain_model(self, traindata_path, output_path, n_class, lr=1e-3, epochs=50, model_name="alltrain_model.h5"):

        """
        Build base model.

        Parameters
        ----------
        traindata_path
            Path of training set.
        output_path
            Path of output files.
        n_class
            Number of cell classes.
        lr
            Learning rate for SGDOptimizer.
        epochs
            Number of total epochs in training.
        model_name
            Name of trained model
        """

        vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

        top_model = Sequential()
        top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
        top_model.add(Dense(256, activation='relu'))
        top_model.add(Dropout(0.5))
        top_model.add(Dense(n_class, activation='softmax'))
        model = Sequential()
        model.add(vgg16_model)
        model.add(top_model)

        train_generator = self.train_datagen.flow_from_directory(
            traindata_path,
            target_size=(224, 224),
            batch_size=self.batch_size,
        )


        print(train_generator.class_indices)


        model.compile(optimizer=SGD(lr=lr, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_generator, epochs=epochs)

        os.makedirs(output_path + "/Models/")
        model.save(output_path + "/Models/" + model_name)





class BuildPrediction():

    def __init__(self, model, result_dir):

        """
        Build prediction.

        Parameters
        ----------
        model
            Path of the trained model used for prediction.
        result_dir
            Path of the output folder.
        """

        self.name = "Prediction"
        self.model = load_model(model)
        self.result_dir = result_dir

        if not os.path.exists(result_dir + "/Predict/"):
            os.makedirs(result_dir + "/Predict/")



    def predict(self, image):

        """
        Build prediction.

        Parameters
        ----------
        image
            Image for prediction.
        """

        image = load_img(image)
        image = image.resize((224, 224))
        image = img_to_array(image)
        image = image / 255
        image = np.expand_dims(image, 0)
        predict = self.model.predict(image)
        classes = np.argmax(predict, axis=1)

        return predict, classes



    def prediction(self, input_spot, input_image, input_position, predicting_type):

        """
        Build prediction.

        Parameters
        ----------
        input_spot
            Spots in the dataset, which used to determine labels.
        input_image
            Path of images for prediction.
        input_position
            Position file of images.
        predicting_type
            Type of predicting spot images.
        """

        celltype = list(set([x[-1] for x in input_spot]))
        celltype.sort()
        label = {}
        for i in range(len(celltype)):
            label[i] = celltype[i]
        print(label)

        imgsLib = []
        imgsLib.extend(glob.glob(os.path.join(input_image + "/", "*.png")))

        pre_type = []
        pre_type_detail = []

        for each in imgsLib:
            pre_detail, pre = self.predict(each)
            imgname = each.split("/")[-1].split(".")[0]
            pre_type.append([imgname, int(pre), label[int(pre)]])
            pre_temp = list(itertools.chain.from_iterable(pre_detail.tolist()))
            temp = [imgname]
            temp.extend(pre_temp)
            pre_type_detail.append(temp)

        coor = {}
        with open(input_position) as f:
            for line in f.readlines():
                temp = line.split()
                coor[temp[2]] = [temp[0], temp[1]]

        filltype = []
        for each in pre_type:
            if each[0] in coor.keys():
                temp = each
                temp.extend(coor[each[0]])
                filltype.append(temp)
        np.savetxt(self.result_dir + "/Predict/" + predicting_type + "_predict.txt", filltype, fmt='%s', delimiter='\t')

        filltype_detail = []
        for each in pre_type_detail:
            if each[0] in coor.keys():
                temp = each
                temp.extend(coor[each[0]])
                filltype_detail.append(temp)
        np.savetxt(self.result_dir + "/Predict/" + predicting_type + "predict_detail.txt", filltype_detail, fmt='%s', delimiter='\t')

