from itertools import groupby

##GLOBAL INPUT
windowLength = 3
classifiedX = [False, True, True, False, False, True, False, True, True, True, True, False, False, True, True, True, False, False, False, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True]
threshold = 0.5
count_thresh=3

class ClassificationCleaner:
    def __init__(self, windowlength=10, threshold=0.8):
        self.history = []
        self.historyClean = []
        self.count=[]

        self.window = windowlength
        self.threshold = threshold

        self.minContinuous = 1

    def streamThis(self, classification):
        self.history.append(classification)
        classif = False
        if len(self.history) >= self.window:
            if sum(self.history[-self.window:]) / len(self.history[-self.window:]) >= self.threshold:
                classif = True

        self.historyClean.append(classif)
        return classif

    def countThis(self):
        # i=0
        # counting=[0]*len(self.historyClean)
        count=0
        for i in range(0,len(self.historyClean)-1):
            if self.historyClean[i+1]==True and self.historyClean[i]==False:
                count+=1
        # for k, g in groupby(self.historyClean):
        #     g = list(g)
        #     if k==1 and len(g) >= self.minContinuous:
        #         continuous = [1]*len(g)
        #         counting[i:i+len(g)] = continuous

        #     i+=len(g)
        return count # continuous


###INPUTS X FOR EACH FRAME
c = ClassificationCleaner(windowLength, threshold)
for i in classifiedX:
    c.streamThis(i)

#countinuous is the list of length classifiedX of 1 and 0. I indicates continuous if threshold or above
#count is number of times from false to true of historyclean list. 
count = c.countThis()

print(count)
