import io
import os
import time
import random
import itertools
import numpy as np
from scipy.io import wavfile
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from .FeatureSelector import FeatureSelector
from .WindowClassifier import WindowClassifier
from .ClassificationCleaner import ClassificationCleaner

class Data():
    def __init__(self, analysisParameters, file):
        self.analysisParameters = analysisParameters
        self.predList = None
        self.file = file
        # print("ANALYSIS", analysisObjects, analysisParameters)

    def analyseData(self):
        print("Data => analyseData()")
        analysisParameters = self.analysisParameters

        featureSelector = FeatureSelector(self.file)
        featureMatrix = featureSelector.getFeatures(windowSize = analysisParameters["classifierWindow"], sampleProportion = analysisParameters["classifierProportion"])

        X = list(featureMatrix[i][0:6] for i in range(0, len(featureMatrix))) # Features
        Y = list(featureMatrix[i][6] for i in range(0, len(featureMatrix))) # Labels; 0 -> False 1 -> True
        
        windowClassifier = WindowClassifier(X, Y) # INPUT TRAINING / VALIDATION

        if (analysisParameters["classifierType"] == 'nn' or analysisParameters["classifierType"] == 'svm' or analysisParameters["classifierType"] == 'rf'):
            self.accuracy = windowClassifier.handleEvaluate(modelName = analysisParameters["classifierType"], extraParameter= analysisParameters["extraParameter"])
        else:
            self.accuracy = windowClassifier.handleEvaluate(modelName = analysisParameters["classifierType"])

        ampliData = featureSelector.getData()

        windowSize = self.analysisParameters["classifierWindow"]
        stepSize = self.analysisParameters["classifierStep"]

        toPredIndices = []
        toPred = []

        for i in range(0, len(ampliData) - windowSize, stepSize):
            toPredIndices.append(i)
            windowFeatures = featureSelector.selectFeatures(i, i+windowSize)
            toPred.append(windowFeatures)

        toPred = np.array(toPred)
        # print(toPred[:, -1])
        toPred = toPred[:, :-1]
        self.predList = windowClassifier.handlePredict(toPred, self.analysisParameters["classifierType"])

        self.sampleTime = len(ampliData) / featureSelector.getSampleFreq() # 41000 x/s, 
        # print("============self.sampleTime=============")
        # print(self.sampleTime)
        

        self.analysisObjects = {
            "featureSelector": featureSelector,
            "windowClassifier": windowClassifier,
        }
    def deleteData(self):
        self.analysisObjects = {}
        self.predList = []
        return

    def cleanPrediction(self): #, analysisObjects, analysisParameters
        print("Data => cleanPrediction()")
        cCleaner = ClassificationCleaner(self.analysisParameters["cleanerWindow"], self.analysisParameters["cleanerProportion"])
        newPreds = list()
        for rawPred in self.predList:
            newPred = cCleaner.streamThis(rawPred)
            if newPred is None:
                newPreds.append(False)
                continue
            newPreds.append(newPred)

        # print("CLASS NEWPREDS", newPreds)
        self.predList = newPreds
        # print(self.predList)
        # print("THE COUNT!")
        # print(cCleaner.countThis())
        self.thisCount = cCleaner.countThis()
        return

    def get_count(self):
        return self.thisCount

    def get_count_minute(self):
        return round(self.thisCount * 60 / self.sampleTime, 2)

    def get_pred_list(self):
        return self.predList

    def get_accuracy_raw(self):
        return self.accuracy

    def get_accuracy(self):
        return round(100 * self.accuracy, 2)

    def get_classifier_window(self):
        return self.analysisParameters["classifierWindow"]

    def get_classifier_proportion(self):
        return self.analysisParameters["classifierProportion"]

    def get_cleaner_window(self):
        return self.analysisParameters["cleanerWindow"]

    def get_cleaner_proportion(self):
        return self.analysisParameters["cleanerProportion"]

    def get_classifier_type(self):
        # print(self.analysisParameters["classifierType"])
        return self.analysisParameters["classifierType"]

    def get_extra_parameter(self):
        # print(self.analysisParameters["classifierType"])
        if "extraParameter" in self.analysisParameters:
            # print(self.analysisParameters["extraParameter"])
            return self.analysisParameters["extraParameter"]
        else:
            return None

    def createFigure(self, isTruth = True):
        # print("Data => createFigure()")

        stepSize = 100 # TODO: link

        time.sleep(1)
        fig = Figure()
        axis = fig.add_subplot(1, 1, 1)

        # needs to be changed to USER_INPUT later
        featureSelector = self.analysisObjects["featureSelector"]

        Y = featureSelector.getData()
        peakTuples = featureSelector.getPeaks()
        # print(peakTuples)

        newPredList = list(itertools.chain.from_iterable(itertools.repeat(x, stepSize) for x in self.predList))

        excessCount = len(Y) - len(newPredList)
        newPredList.extend([newPredList[-1]] * excessCount)

        isEvent = False

        noEventY = list()
        eventY = list()
        noEventX = list()
        eventX = list()

        predEventX = list()
        predEventY = list()
        predNoEventX = list()
        predNoEventY = list()

        peakMarkers = list()

        for peakA, peakB in peakTuples:
            peakMarkers.append(peakA)
            peakMarkers.append(peakB)

        # print()
        # print("PEAKMARKERS", peakMarkers)

        # print(len(Y))
        # print(len(newPredList))
        # print(sum(newPredList))

        for i in range(len(Y)):
            if i in peakMarkers:
                # print(i)
                # print(peakMarkers)
                # print("isEvent")
                # print(i)
                isEvent = not isEvent

            if isEvent:
                eventX.append(i)
                eventY.append(Y[i])
            else:
                noEventX.append(i)
                noEventY.append(Y[i])

            if int(newPredList[i]) == 0:
                predNoEventX.append(i)
                predNoEventY.append(Y[i])
            else:
                # print("predList[i]")
                # print(predList[i])
                predEventX.append(i)
                predEventY.append(Y[i])

        def convertsec(ls_x):
            output = []
            for x in ls_x:
                output.append(x/10000)
            return output

        if isTruth:
            axis.scatter(convertsec(noEventX), noEventY, s = 0.5, c='lightblue')
            axis.scatter(convertsec(eventX), eventY, s = 0.5, c='coral')
        else:
            axis.scatter(convertsec(predNoEventX), predNoEventY, s = 0.1, c='lightgreen')
            axis.scatter(convertsec(predEventX), predEventY, s = 0.1, c='darkgreen')
        # axis.set_title('Predictions of Eye-Movement (Dark-Green) and \nNon-Eye-Movement (Light-Green)')
        axis.set_xlabel('Time (s)')
        axis.set_ylabel('Amplitude From Brainbox')

        return fig
