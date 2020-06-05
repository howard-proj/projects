import math
import statistics
import numpy as np
from random import sample
from scipy.io import wavfile
from scipy.stats import gmean
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
from models.gaussianFilter import filterData

class FeatureSelector():
    def __init__(self, fileName):
        # print("FeatureSelector => constructor()")
        peakNum = 3
        # peakNum = len(fileName.split(".")[0].split("/")[-1])
        # if peakNum > 4:
        #     peakNum = int(peakNum*3/4)
        self.sampleFreq, self.rawData = wavfile.read(fileName)
        self.ampliData = filterData(self.rawData)

        self.peakTuples = self.findKPeaks(peakNum)
        return

    def getData(self):
        return self.ampliData

    def getSampleFreq(self):
        return self.sampleFreq

    def selectFeatures(self, startIndex, endIndex):
        # print("FeatureSelector => selectFeatures()")
        sampleList = self.ampliData[startIndex:endIndex]
        amplitudeArray = np.array(sampleList)

        amplitudeIndices = np.arange(startIndex, endIndex)

        isEvent = False
        for peakTuple in self.peakTuples:
            isEvent = isEvent or ((amplitudeIndices > peakTuple[0]).any() and (amplitudeIndices < peakTuple[1]).any())
        # print(isEvent)

        featuresDict = {
            # "geomean": gmean(amplitudeArray),
            # "geomedian": min(map(lambda p1:(p1, sum(map(lambda p2:euclidean(p1,p2),amplitudeArray))), amplitudeArray), key = lambda x:x[1])[0],
            "mean": amplitudeArray.mean(),
            "min": amplitudeArray.min(),
            "iq1": np.percentile(amplitudeArray, 25),
            "median": np.median(amplitudeArray),
            "iq3": np.percentile(amplitudeArray, 75),
            "max": amplitudeArray.max(),
            "label": bool(isEvent),
        }
        # print(self.peakTuples)
        # print(startIndex, endIndex)
        # print(featuresDict)

        outputFeatures = np.array(list(featuresDict.values()))
        return outputFeatures

    def getFeatures(self, windowSize = 500, sampleProportion = 0.001):
        print("FeatureSelector => getFeatures()")
        indicesNum = len(self.ampliData) - windowSize
        dataIndices = list(range(indicesNum))
        eventIndices = list()

        for i in range(len(self.peakTuples), 0, -1):
            peakTuple = self.peakTuples[i-1]
            eventIndices.extend(dataIndices[peakTuple[0]:peakTuple[1]])
            del dataIndices[peakTuple[0]:peakTuple[1]]

        sampleEventless = sample(dataIndices, int(sampleProportion * len(eventIndices)))
        sampleEventful = sample(eventIndices, int(sampleProportion * len(eventIndices)))

        sampleIndices = list()
        sampleIndices.extend(sampleEventless)
        sampleIndices.extend(sampleEventful)

        featuresList = list()
        for startIndex in sampleIndices:
            endIndex = startIndex + windowSize
            featuresDict = self.selectFeatures(startIndex, endIndex)
            featuresList.append(featuresDict)
        # print(np.array(featuresList))
        
        return np.array(featuresList)

    def findKPeaks(self, k = 3):
        print("FeatureSelector => findKPeaks()")
        peakFinderData = np.array(self.ampliData)
        peakFinderData.setflags(write = 1)
        zeroCrossings = np.where(np.diff(np.sign(peakFinderData)))[0]

        eventIntervals = list()
        newIntervals = list()

        for i in range(k):
            maxIndex = np.argmax(peakFinderData)
            zeroIndex = np.argmax(zeroCrossings > maxIndex)
            
            lowerIndex = max(zeroIndex - 1, 0)
            upperIndex = min(zeroIndex + 2, len(zeroCrossings)-1)

            eventInterval = (zeroCrossings[lowerIndex], zeroCrossings[upperIndex])
            peakFinderData[eventInterval[0]:eventInterval[1]] = 0
            eventIntervals.append(eventInterval)

            newIntervals.append(zeroCrossings[lowerIndex])
            newIntervals.append(zeroCrossings[upperIndex])

        newIntervals.sort()

        outputEvents = list()

        for i in range(len(newIntervals)-1):
            if newIntervals[i] == newIntervals[i+1]:
                newIntervals[i] -= 1

        for i in range(int(len(newIntervals)/2)):
            index = 2*i
            newInterval = (newIntervals[index], newIntervals[index+1])
            outputEvents.append(newInterval)

        self.eventIntervals = outputEvents

        return outputEvents

    def getPeaks(self):
        return self.eventIntervals

# FeatureSelector("assets/RLR.wav")