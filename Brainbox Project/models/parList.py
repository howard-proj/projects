import random
import numpy as np

'''
windowSize
sampleProportion
cynthiaWindow
threshold
'''

def rand_par_list(par_dic):
    iterations = par_dic['iterations']
    parameters_list = []

    for k in range(iterations):
        windowSize = round(random.uniform(par_dic['classifierWindow'][0],par_dic['classifierWindow'][1]))
        sampleProportion = round(random.uniform(par_dic['classifierProportion'][0],par_dic['classifierProportion'][1]),3)
        cynthiaWindow = round(random.uniform(par_dic['cleanerWindow'][0],par_dic['cleanerWindow'][1]))
        threshold = round(random.uniform(par_dic['cleanerProportion'][0],par_dic['cleanerProportion'][1]),2)

        if par_dic['extraParameter'] is not None:
            extra = round(random.uniform(par_dic['extraParameter'][0],par_dic['extraParameter'][1]),2)
            temp_list = [windowSize, sampleProportion, cynthiaWindow, threshold, extra]
        else:
            temp_list = [windowSize, sampleProportion, cynthiaWindow, threshold]
        parameters_list.append(temp_list)

    return parameters_list

def lin_par_list(iterations):
    parameters_list = []

    windowSize = np.linspace(1000,100000,iterations,dtype = int)
    sampleProportion = np.around(np.linspace(0,1,iterations,dtype = float),3)
    cynthiaWindow = np.linspace(10,10000,iterations, dtype = int)
    threshold = np.around(np.linspace(0,1,iterations,dtype = float),2)

    for i in range(0,iterations):
        temp_list = [windowSize[i], sampleProportion[i], cynthiaWindow[i], threshold[i]]
        parameters_list.append(temp_list)

    return parameters_list
