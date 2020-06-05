def iterate_Pipeline(parameters_list):
    saved_values = []

    for i, paramaters in enumerate(parameters_list):
        print('paramaters', paramaters)
        print('iteration {} out of {}'.format(i,len(parameters_list)))
        analysisParameters = {
            "classifierType": 'svm',
            "classifierStep": 100,
            "classifierWindow": paramaters[0],
            "classifierProportion": paramaters[1],
            "cleanerWindow": paramaters[2],
            "cleanerProportion": paramaters[3],
        }
        user = doPipe(analysisParameters)
    return
