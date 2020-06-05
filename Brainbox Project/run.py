import io
import os
import json
import time
import random
import itertools
import numpy as np
from scipy.io import wavfile
import pickle

import plotly
import plotly.express as px
import plotly.graph_objs as go
import chart_studio.plotly as py

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from flask import Flask, Response, render_template, request, jsonify, send_file, redirect

from models.FeatureSelector import FeatureSelector
from models.WindowClassifier import WindowClassifier
from models.ClassificationCleaner import ClassificationCleaner
from models.Data import Data
from models.parList import rand_par_list
from models.parList import lin_par_list

app = Flask(__name__)

# Don't touch this
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

#====@ANALYSIS: Use this variable into feature selector <- gets the user's wav file
USER_INPUT = None
#=== Formated: å[Data, accuracy] <- User's history
USER_ANALYSIS = []
#===

# import pickle
# with open("temp/RLR.remdata",'rb') as inputFile:
#     uA = pickle.load(inputFile)
# with open("temp/RF_RLR.remdata",'rb') as inputFile:
#     uB = pickle.load(inputFile)
# print(len(uA))
# print(len(uB))
# print(len(USER_ANALYSIS))
# USER_ANALYSIS.extend(uA)
# USER_ANALYSIS.extend(uB)
# exit()

# Create a new form; our main page gets messy!
@app.route('/index', methods=["GET"])
def index_get():
    return redirect('/')
@app.route('/index', methods=["POST"])
def index_post():
    global USER_INPUT
    try:
        target = os.path.join(APP_ROOT, 'data/')
        #print("TARGET: ", target)

        if not os.path.isdir(target):
            os.mkdir(target)

        for file in request.files.getlist("inputFile"):
            #print(file)
            filename = file.filename
            destination = "/".join([target, filename])
            #print("DESTINATION", destination)
            file.save(destination)
            USER_INPUT = destination
            print(USER_INPUT.split('//')[-1])

        load_target = os.path.join(APP_ROOT, 'saved_data/')

        if os.path.isdir(load_target):
            print(filename[:-4])
            for item in os.listdir(load_target):
                if item == filename[:-4] + '.remdata':
                    load()
                    break
    except Exception as e:
        print(e)
        return redirect('/')

    # parameters_list = rand_par_list(5)
    # iteratePipeline(parameters_list)

    tableResults = [
        {
            "ClassifierType": "SVM",
            "ClassifierWindow": 1000,
            "SampleProportion": 0.01,
            "CleanerWindow": 100,
            "CleanerThreshold": 0.8,
            "ExtraParameter": 1,
            "Accuracy": 0.9888,
            "Count": 1,
            "CountMinute": 13.93,
            "REMState": "REM",
        },
    ]

    return render_template('index.html', tableResults = tableResults)

@app.route('/')
def upload():
    return render_template('upload.html')

@app.route('/clear')
def clear():
    global USER_ANALYSIS
    USER_ANALYSIS = []
    graphJSON = getHelix()
    return jsonify(result = True, graphJSON=graphJSON)


@app.route('/save')
def save():
    global USER_ANALYSIS
    global USER_INPUT

    print('saving')

    target = os.path.join(APP_ROOT, 'saved_data/')

    if not os.path.isdir(target):
        os.mkdir(target)

    filename = USER_INPUT.split('//')[-1]
    filename = filename.split('.')[0]
    destination = "/".join([target, filename])+'.remdata'

    with open(destination,'wb') as output:
        pickle.dump(USER_ANALYSIS, output, pickle.HIGHEST_PROTOCOL)

    return jsonify(result = True)

@app.route('/get_graph_only')
def get_graph_only():
    graphJSON = retrieveGraph(request)
    return jsonify(graphJSON = graphJSON)

import math
def getHelix():
    t = np.linspace(0, 20, 100)
    x, y, z = np.cos(t), np.sin(t), t

    trace = go.Scatter3d(x=x, y=y, z=z,
        name='Accuracy',
        mode='markers',
        marker=dict(
            size=12,
            color=(  abs(x)/4 + abs(y)/4 + abs(t)/50 + abs(np.cos(3*t - math.pi/3))  + abs(np.cos(8*t - math.pi/2))/2  )/2,
            colorscale='YlGn',
            colorbar=dict(
                title="Accuracy",
                tickvals=[0,0.5,1],
                # range=[0.5, 1],
            ),
            cauto=False,
            cmin=0.25,
            cmax=1,
            opacity=0.8
        )
    )
    data = [trace]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def retrieveGraph(request):
    print("retrieveGraph()")
    ### PLOTLY STUFF ###
    toX = request.args.get('plot_x')
    toY = request.args.get('plot_y')
    toZ = request.args.get('plot_z')
    classifyType = request.args.get('classify_type')

    graphX = getPlotData(toX, classifyType)
    graphY = getPlotData(toY, classifyType)
    graphZ = getPlotData(toZ, classifyType)
    graphA = getPlotData("A", classifyType) # A for Accuracy

    if len(graphA) == 0:
        graphJSON = getHelix()
    else:
        graphJSON = getGraph(graphX, graphY, graphZ, graphA, getName(toX), getName(toY), getName(toZ))
    return graphJSON
### END OTHER PLOTLY STUFF ###

@app.route('/load')
def load():
    global USER_ANALYSIS
    global USER_INPUT

    print('loading')

    target = os.path.join(APP_ROOT, 'saved_data/')

    filename = USER_INPUT.split('//')[-1]
    filename = filename.split('.')[0]
    destination = "/".join([target, filename])+'.remdata'

    with open(destination,'rb') as inputFile:
        USER_ANALYSIS = pickle.load(inputFile)

    best_par, best_pars = get_best()
    print(best_par)
    best_pars = sorted(best_pars, key = lambda x: x.get_accuracy(), reverse=True)[:10]
    print(best_pars)

    outputBest = list()

    for best_p in best_pars:
        resultMap = {
            "ClassifierType": best_p.get_classifier_type(),
            "ClassifierWindow": best_p.get_classifier_window(),
            "SampleProportion": best_p.get_classifier_proportion(),
            "CleanerWindow": best_p.get_cleaner_window(),
            "CleanerThreshold": best_p.get_cleaner_proportion(),
            "ExtraParameter": best_p.get_extra_parameter(),
            "Accuracy": best_p.get_accuracy(),
            "Count": best_p.get_count(),
            "CountMinute": best_p.get_count_minute(),
        }
        outputBest.append(resultMap)

    # TODO: SORT #
    # sorted() #

    # for item in best_pars:
    #     print_dat(item)
    # print_dat(best_par)
    graphJSON = retrieveGraph(request)

    return jsonify(result = True, graphJSON=graphJSON, outputBest=outputBest)

@app.route('/process')
def process():
    try:
        classifyType = request.args.get('classify_type').lower()
        analysisParameters = {
            "classifierType": classifyType,
            "classifierStep": 100,
            "classifierWindow": int(request.args.get('classify_window')),
            "classifierProportion": float(request.args.get('classify_proportion')),
            "cleanerWindow": int(request.args.get('cleaner_window')),
            "cleanerProportion": float(request.args.get('cleaner_proportion')),

        }

        if request.args.get('special_parameter') != "":
            if classifyType == "nn" or classifyType == "rf":
                analysisParameters["extraParameter"] = int(request.args.get('special_parameter'))
            elif classifyType == "svm":
                analysisParameters["extraParameter"] = float(request.args.get('special_parameter'))
        else:
            analysisParameters["extraParameter"] = None

        isValid = analysisParameters["classifierWindow"] and analysisParameters["classifierProportion"] and analysisParameters["cleanerWindow"] and analysisParameters["cleanerProportion"]
        if isValid:

            print("AhoyB")

            global USER_ANALYSIS
            user = doPipe(analysisParameters, True)
            print("AhoyC")
            predList = user.get_pred_list()
            accuracyScore = user.get_accuracy()
            print("AhoyD")
            thisCount = user.get_count()
            graphJSON = retrieveGraph(request)
            print("AhoyE")

            my_history = {
                "ClassifierType": USER_ANALYSIS[-1].get_classifier_type(),
                "ClassifierWindow": USER_ANALYSIS[-1].get_classifier_window(),

                "SampleProportion": USER_ANALYSIS[-1].get_classifier_proportion(),

                "CleanerWindow": USER_ANALYSIS[-1].get_cleaner_window(),
                "CleanerThreshold": USER_ANALYSIS[-1].get_cleaner_proportion(),

                "Accuracy": USER_ANALYSIS[-1].get_accuracy(),
                "Count": USER_ANALYSIS[-1].get_count(),
                "CountMinute": USER_ANALYSIS[-1].get_count_minute()
            }

            history_user = [ my_history ]

            return jsonify(result = accuracyScore, prediction = predList, count = thisCount, graphJSON=graphJSON, history_user = history_user)
        else:
            return jsonify(result = 'Error: Please input proper value')
    except Exception as e:
        print(e)
        return jsonify(result = "Something went wrong")

def get_best():
    global USER_INPUT
    global USER_ANALYSIS
    try:
        correct_count = int(USER_INPUT.split('.')[0].split('_')[-1])
    except:
        correct_count = len(USER_INPUT.split('.')[0].split('//')[-1])
    best_pars = []

    for item in USER_ANALYSIS:
        if item.get_count() == correct_count:
            best_pars.append(item)

    best_par = best_pars[0]

    for item in best_pars:
        if item.get_accuracy() > best_par.get_accuracy():
            best_par = item
    return best_par, best_pars

def print_dat(data):
    par = {'Classifier Type': data.get_classifier_type(),
            'Classifier Window Size': data.get_classifier_window(),
            'Classifier Sample Proportion': data.get_classifier_proportion(),
            'Cleaner Window Size': data.get_cleaner_window(),
            'Cleaner Threshold': data.get_cleaner_proportion(),
            'Extra Parameter': data.get_extra_parameter()
            }
    print(par)

### NEW PLOT STUFF ###
def getGraph(x, y, z, a, toX, toY, toZ):
    textLines = []
    for i in range(len(a)):
        stringRow = ""
        if type(x[i]) == float:
            stringRow += "%s: %.2f<br>" % (toX, x[i])
        else:
            stringRow += "%s: %d<br>" % (toX, x[i])

        if type(y[i]) == float:
            stringRow += "%s: %.2f<br>" % (toY, y[i])
        else:
            stringRow += "%s: %d<br>" % (toY, y[i])

        if type(z[i]) == float:
            stringRow += "%s: %.2f<br>" % (toZ, z[i])
        else:
            stringRow += "%s: %d<br>" % (toZ, z[i])
        textLines.append(stringRow)
    trace = go.Scatter3d(x=x, y=y, z=z,
        name='Accuracy',
        mode='markers',
        text = textLines,
        hoverinfo = 'text',
        marker = dict(
            size=12,
            color=a,
            colorscale='YlGn',
            colorbar=dict(
                title="Accuracy",
                tickvals=[0,0.5,1],
                # range=[0.5, 1],
            ),
            cauto=False,
            cmin=0.25,
            cmax=1,
            opacity=0.8
        )
    )
    data = [trace]
    graphJSON = json.dumps(data, cls=plotly.utils.PlotlyJSONEncoder)
    return graphJSON

def getPlotData(selectedOption, whatType):
    global USER_ANALYSIS
    # print(USER_ANALYSIS)
    outputData = list()
    if whatType == None:
        return outputData
    whatType = whatType.lower()
    if selectedOption == "A":
        for dataStruct in USER_ANALYSIS:
            thisStat = dataStruct.get_accuracy_raw()
            thisType = dataStruct.get_classifier_type()
            if thisType == whatType:
                outputData.append(thisStat)
    elif selectedOption == "C":
        for dataStruct in USER_ANALYSIS:
            thisStat = dataStruct.get_count()
            thisType = dataStruct.get_classifier_type()
            if thisType == whatType:
                outputData.append(thisStat)
    elif selectedOption == "CW":
        for dataStruct in USER_ANALYSIS:
            thisStat = dataStruct.get_classifier_window()
            thisType = dataStruct.get_classifier_type()
            if thisType == whatType:
                outputData.append(thisStat)
    if selectedOption == "CP":
        for dataStruct in USER_ANALYSIS:
            thisStat = dataStruct.get_classifier_proportion()
            thisType = dataStruct.get_classifier_type()
            if thisType == whatType:
                outputData.append(thisStat)
    elif selectedOption == "CCW":
        for dataStruct in USER_ANALYSIS:
            thisStat = dataStruct.get_cleaner_window()
            thisType = dataStruct.get_classifier_type()
            if thisType == whatType:
                outputData.append(thisStat)
    elif selectedOption == "CCP":
        for dataStruct in USER_ANALYSIS:
            thisStat = dataStruct.get_cleaner_proportion()
            thisType = dataStruct.get_classifier_type()
            if thisType == whatType:
                outputData.append(thisStat)
    elif selectedOption == "EX":
        for dataStruct in USER_ANALYSIS:
            thisStat = dataStruct.get_extra_parameter()
            thisType = dataStruct.get_classifier_type()
            if thisType == whatType:
                outputData.append(thisStat)
    return outputData
### END NEW PLOT STUFF ###

def getName(nameCode):
    if (nameCode == "A"):
        return "Accuracy"
    if (nameCode == "C"):
        return "Count"
    if (nameCode == "CW"):
        return "Classifier Window"
    if (nameCode == "CP"):
        return "Classifier Proportion"
    if (nameCode == "CCW"):
        return "Cleaner Window"
    if (nameCode == "CCP"):
        return "Cleaner Threshold"
    if (nameCode == "EX"):
        return "Extra Parameter"

#FILTER SVM, etc.
@app.route('/random_process')
def random_process():
    print("random_process()")
    try:
        classifyType = request.args.get('classify_type').lower()
        print(classifyType)
        if (classifyType == "nn"):
            extraRange = [int(request.args.get('extra_parameter_lower')), int(request.args.get('extra_parameter_upper'))]
        elif (classifyType == "svm"):
            extraRange = [float(request.args.get('extra_parameter_lower')), float(request.args.get('extra_parameter_upper'))]
        elif (classifyType == "rf"):
            extraRange = [int(request.args.get('extra_parameter_lower')), int(request.args.get('extra_parameter_upper'))]
        else:
            extraRange = None
        analysisParameters = {
            "classifierType": classifyType,
            "classifierStep": 100,
            "classifierWindow": [int(request.args.get('random_classify_window_lower')), int(request.args.get('random_classify_window_upper'))],
            "classifierProportion": [float(request.args.get('random_classify_proportion_lower')), float(request.args.get('random_classify_proportion_upper'))],
            "cleanerWindow": [int(request.args.get('random_cleaner_window_lower')), int(request.args.get('random_cleaner_window_upper'))],
            "cleanerProportion": [float(request.args.get('random_cleaner_proportion_lower')), float(request.args.get('random_cleaner_proportion_upper'))],
            "iterations": int(request.args.get('random_iterations')),

            "extraParameter": extraRange,

        }

        # return jsonify(result = True)
        isValid = analysisParameters["classifierWindow"] and analysisParameters["classifierProportion"] and analysisParameters["cleanerWindow"] and analysisParameters["cleanerProportion"]
        if isValid:
            print(analysisParameters)

            parameters_list = rand_par_list(analysisParameters)
            print(parameters_list)
            iteratePipeline(parameters_list, classifyType) # disabled randomisation

            graphJSON = retrieveGraph(request)

            return jsonify(result = True, graphJSON=graphJSON)
        else:
            return jsonify(result = 'Error: Please input proper value')
    except Exception as e:
        print(e)
        return jsonify(result = "Something went wrong")

def doPipe(analysisParameters, isIndividual=False):
    print("doPipe()")
    global USER_ANALYSIS
    global USER_INPUT

    #Store the data
    user = Data(analysisParameters = analysisParameters, file = USER_INPUT)
    user.analyseData()
    user.cleanPrediction()

    # USER
    if not isIndividual:
        user.deleteData()

    # user.deleteData()
    USER_ANALYSIS.append(user)

    print("END doPipe()")
    return user

classifierTypes = ["lr", "mlp", "gpc", "dtc", "ada", "gnb", "qd"]

def iteratePipeline(parameters_list, classifierType = None):
    print("iteratePipeline()")
    saved_values = []

    for i, paramaters in enumerate(parameters_list):
        if classifierType is None:
            classifierType = random.choice(classifierTypes)
        print('paramaters', paramaters)
        print('iteration {} out of {}'.format(i+1,len(parameters_list)))
        analysisParameters = {
            "classifierType": classifierType,
            "classifierStep": 100,
            "classifierWindow": paramaters[0],
            "classifierProportion": paramaters[1],
            "cleanerWindow": paramaters[2],
            "cleanerProportion": paramaters[3],
        }
        if classifierType == "nn" or classifierType == "rf":
            analysisParameters["extraParameter"] = int(paramaters[4])
        elif classifierType == "svm":
            analysisParameters["extraParameter"] = paramaters[4]

        user = doPipe(analysisParameters)
    return

@app.route('/plotTruth')
def plotTruth():
    print("plotTruth()")
    global USER_ANALYSIS
    isLoaded = request.args.get('index')
    if isLoaded is None:
        return Response()
    fig = USER_ANALYSIS[-1].createFigure()
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plotAccuracies')
def plotAccuracies():
    print("plotAccuracies()")
    global USER_ANALYSIS
    isLoaded = request.args.get('index')
    if isLoaded is None:
        return Response()
    statDistribution = list()
    for dataStruct in USER_ANALYSIS:
        thisStat = dataStruct.get_accuracy()
        statDistribution.append(thisStat)
    image = makeHistogram(statDistribution)
    return image

@app.route('/plotClassifierWindows')
def plotClassifierWindows():
    print("plotClassifierWindows()")
    global USER_ANALYSIS
    isLoaded = request.args.get('index')
    if isLoaded is None:
        return Response()
    statDistribution = list()
    for dataStruct in USER_ANALYSIS:
        thisStat = dataStruct.get_classifier_window()
        statDistribution.append(thisStat)
    image = makeHistogram(statDistribution)
    return image

@app.route('/plotClassifierProportions')
def plotClassifierProportions():
    print("plotClassifierProportions()")
    global USER_ANALYSIS
    isLoaded = request.args.get('index')
    if isLoaded is None:
        return Response()
    statDistribution = list()
    for dataStruct in USER_ANALYSIS:
        thisStat = dataStruct.get_classifier_proportion()
        statDistribution.append(thisStat)
    image = makeHistogram(statDistribution)
    return image

@app.route('/plotCleanerWindows')
def plotCleanerWindows():
    print("plotCleanerWindows()")
    global USER_ANALYSIS
    isLoaded = request.args.get('index')
    if isLoaded is None:
        return Response()
    statDistribution = list()
    for dataStruct in USER_ANALYSIS:
        thisStat = dataStruct.get_cleaner_window()
        statDistribution.append(thisStat)
    image = makeHistogram(statDistribution)
    return image

@app.route('/plotCleanerProportions')
def plotCleanerProportions():
    print("plotCleanerProportions()")
    global USER_ANALYSIS
    isLoaded = request.args.get('index')
    if isLoaded is None:
        return Response()
    statDistribution = list()
    for dataStruct in USER_ANALYSIS:
        thisStat = dataStruct.get_cleaner_proportion()
        statDistribution.append(thisStat)
    image = makeHistogram(statDistribution)
    return image

@app.route('/plotClassifierCount')
def plotClassifierCount():
    print("plotClassifierCount()")
    global USER_ANALYSIS
    isLoaded = request.args.get('index')
    if isLoaded is None:
        return Response()
    statDistribution = list()
    for dataStruct in USER_ANALYSIS:
        thisStat = dataStruct.get_classifier_type()
        statDistribution.append(thisStat)
    # print(statDistribution)
    image = makeHistogram(statDistribution)
    return image

def makeHistogram(distributionList):
    img = io.BytesIO()
    fig = Figure()
    axis = fig.add_subplot(1, 1, 1)
    axis.hist(distributionList, bins = 5, rwidth=0.8)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/plotPrediction')
def plotPrediction():
    print("plotPrediction()")
    global USER_ANALYSIS
    isLoaded = request.args.get('index')
    if isLoaded is None:
        return Response()
    fig = USER_ANALYSIS[-1].createFigure(isTruth = False)
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    USER_ANALYSIS[-1].deleteData()
    return Response(output.getvalue(), mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)

# windowSizeA = the window size used for sample from the raw data (fed into classifier)
# windowSizeB = the window size used to get consus during LiveStreamer
