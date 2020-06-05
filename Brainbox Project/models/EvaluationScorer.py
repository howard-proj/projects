def getF1(predList, realList):
    Tp = 0
    Fn = 0
    Tn = 0
    Fp = 0
    for(i in range(len(predList))):
        for(y in len(realList)):
            if(predList[i] == "True" and predList[i] == realList[y]):
                Tp += 1
            elif(predList[i] == "True" and predList[i] != realList[y]):
                Fp += 1
            elif(predList[i] == "False" and predList[i] == realList[y]):
                Fn += 1
            elif(predList[i] == "False" adn predList[i] != Fn[y]):
                Tn += 1;

    accuracy = (Tp + Tn)/(Tp + Tn + Fp + Fn)
    recall = Tp/(Tp + Fn)
    precision = Tp/(Tp + Fp)
    F_measure = (2*recall * precision)/(recall + precision)
    return accuracy,recall,precision,F_measure


