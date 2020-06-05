import numpy as np
import random
from sklearn import preprocessing, neighbors, svm
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

class WindowClassifier():
    def __init__(self, X, Y):  # X refers to the features, Y the labels
        self._X = X
        self._Y = Y

        # Test 40% Train 60%
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self._X, self._Y, test_size=0.4)

    def handleEvaluate(self, modelName, extraParameter=None): # X -> features to predict
        if modelName == "nn":
            self.knn_metrics = {}
            self._knn = neighbors.KNeighborsClassifier(extraParameter)
            self._knn.fit(self.X_train, self.y_train)
            self.knn(5)
            return self.knn_metrics['accuracy']
        elif modelName=="svm":
            self.svm_metrics = {}
            self._svm = self._svm = svm.SVC(C=extraParameter)
            self._svm.fit(self.X_train, self.y_train)
            self.svm(5)
            return self.svm_metrics['accuracy']
        elif modelName=="rf":
            self.rff_metrics = {}
            self._rff = RandomForestClassifier(extraParameter)
            self._rff.fit(self.X_train, self.y_train)
            self.random_forest(5)
            return self.rff_metrics['accuracy']
        elif modelName=="lr":
            self.lr_metrics = {}
            self._lr = LogisticRegression()
            self._lr.fit(self.X_train, self.y_train)
            self.lr(5)
            return self.lr_metrics['accuracy']
        elif modelName=="mlp":
            self.mlp_metrics = {}
            self._mlp = MLPClassifier()
            self._mlp.fit(self.X_train, self.y_train)
            self.mlp(5)
            return self.mlp_metrics['accuracy']
        elif modelName=="gpc":
            self.gaupc_metrics = {}
            self._gaupc = GaussianProcessClassifier()
            self._gaupc.fit(self.X_train, self.y_train)
            self.gaupc(5)
            return self.gaupc_metrics['accuracy']
        elif modelName=="dtc":
            self.detc_metrics = {}
            self._detc = DecisionTreeClassifier()
            self._detc.fit(self.X_train, self.y_train)
            self.detc(5)
            return self.detc_metrics['accuracy']
        elif modelName=="ada":
            self.adab_metrics = {}
            self._adab = AdaBoostClassifier()
            self._adab.fit(self.X_train, self.y_train)
            self.adab(5)
            return self.adab_metrics['accuracy']
        elif modelName=="gnb":
            self.ganb_metrics = {}
            self._ganb = GaussianNB()
            self._ganb.fit(self.X_train, self.y_train)
            self.ganb(5)
            return self.ganb_metrics['accuracy']
        elif modelName=="qd":
            self.qud_metrics = {}
            self._qud = QuadraticDiscriminantAnalysis()
            self._qud.fit(self.X_train, self.y_train)
            self.qud(5)
            return self.qud_metrics['accuracy']
        # call other methods here
        return None

    def handlePredict(self, X, modelName): # X -> features to predict
        print("WindowClassifier => handlePredict()")
        if modelName == "nn":
            return self.predict_knn(X)
        elif modelName=="svm":
            return self.predict_svm(X)
        elif modelName=="rf":
            return self.predict_forest(X)
        elif modelName=="mlp":
            return self.predict_mlp(X)
        elif modelName=="gpc":
            return self.predict_gaupc(X)
        elif modelName=="dtc":
            return self.predict_detc(X)
        elif modelName=="ada":
            return self.predict_adab(X)
        elif modelName=="gnb":
            return self.predict_ganb(X)
        elif modelName=="qd":
            return self.predict_qud(X)
        return None

    # Add new classifers here
    def logistic(self, CV, n_estimate):
        # Create KNN classifier
        clf = neighbors.KNeighborsClassifier(n_neighbors=n_estimate)

        # Train model with a specified cv
        cv_scores = cross_val_score(clf, self._X, self._Y, cv=CV, scoring='accuracy')
        cv_precision = cross_val_score(clf, self._X, self._Y, cv=CV, scoring='precision')
        cv_recall = cross_val_score(clf, self._X, self._Y, cv=CV, scoring='recall')

        values = {'accuracy': np.mean(cv_scores),
                  'precision': np.mean(cv_precision),
                  'recall': np.mean(cv_recall)}
        return values

    def X(self):
        return self._X if self._X is not None else None

    def Y(self):
        return self._Y if self._Y is not None else None

    def Y_test(self):
        return self.y_test if self.y_test is not None else None

    '''Next few methods get the accuracy, until the webpage is developed we will redefine this again'''
    def knn_accuracy(self):
        if len(self.knn_metrics) != 0:
            return self.knn_metrics['accuracy']
        raise ValueError('Knn has not been evaluated')

    def rff_accuracy(self):
        if len(self.rff_metrics) != 0:
            return self.knn_metrics['accuracy']
        raise ValueError('Random Forest has not been evaluated')

    def svm_accuracy(self):
        if len(self.svm_metrics) != 0:
            return self.knn_metrics['accuracy']
        raise ValueError('SVM has not been evaluated')

    '''All classifiers along with metrics in interest and other predictions'''
    def knn(self, CV):
        # Train model with a specified cv
        cv_scores = cross_val_score(self._knn, self._X, self._Y, cv=CV, scoring='accuracy')
        cv_precision = cross_val_score(self._knn, self._X, self._Y, cv=CV, scoring='precision')
        cv_recall = cross_val_score(self._knn, self._X, self._Y, cv=CV, scoring='recall')

        values = {'accuracy': np.mean(cv_scores),
                  'precision': np.mean(cv_precision),
                  'recall': np.mean(cv_recall)}

        self.knn_metrics = values
        return values 

    def predict_knn(self, matrix = None):  # matrix -> features
        print("WindowClassifier => predictKNN()")
        # Make prediction with knn
        # print(matrix[0])
        predX = self.X_test if matrix is None else matrix
        # print(predX.shape)
        predY = self._knn.predict(predX)  # => [0, 1, 0, 1, 1, 1, 1]
        # print(predY)
        return predY

    def svm(self, CV):
        # Train model with a specified cv
        cv_scores = cross_val_score(self._svm, self._X, self._Y, cv=CV, scoring='accuracy')
        cv_precision = cross_val_score(self._svm, self._X, self._Y, cv=CV, scoring='precision')
        cv_recall = cross_val_score(self._svm, self._X, self._Y, cv=CV, scoring='recall')

        values = {'accuracy': np.mean(cv_scores),
                  'precision': np.mean(cv_precision),
                  'recall': np.mean(cv_recall)}
        self.svm_metrics = values
        return values

    def predict_svm(self, matrix = None):  # matrix -> features
        print("WindowClassifier => predictSVM()")
        # Make prediction with svm
        Y_pred = self._svm.predict(self.X_test if matrix is None else matrix)
        return Y_pred

    def random_forest(self, CV):
        # Train model with a specified cv
        cv_scores = cross_val_score(self._rff, self._X, self._Y, cv=CV, scoring='accuracy')
        cv_precision = cross_val_score(self._rff, self._X, self._Y, cv=CV, scoring='precision')
        cv_recall = cross_val_score(self._rff, self._X, self._Y, cv=CV, scoring='recall')

        values = {'accuracy': np.mean(cv_scores),
                  'precision': np.mean(cv_precision),
                  'recall': np.mean(cv_recall)}
        self.rff_metrics = values
        return values

    def predict_forest(self, matrix = None):  # matrix -> features
        # Make prediction with random forest
        Y_pred = self._rff.predict(self.X_test if matrix is None else matrix)
        return Y_pred


    def lr(self, CV):
        # Train model with a specified cv
        cv_scores = cross_val_score(self._lr, self._X, self._Y, cv=CV, scoring='accuracy')
        cv_precision = cross_val_score(self._lr, self._X, self._Y, cv=CV, scoring='precision')
        cv_recall = cross_val_score(self._lr, self._X, self._Y, cv=CV, scoring='recall')

        values = {'accuracy': np.mean(cv_scores),
                  'precision': np.mean(cv_precision),
                  'recall': np.mean(cv_recall)}

        self.lr_metrics = values
        return values

    def mlp(self, CV):
        # Train model with a specified cv
        cv_scores = cross_val_score(self._mlp, self._X, self._Y, cv=CV, scoring='accuracy')
        cv_precision = cross_val_score(self._mlp, self._X, self._Y, cv=CV, scoring='precision')
        cv_recall = cross_val_score(self._mlp, self._X, self._Y, cv=CV, scoring='recall')

        values = {'accuracy': np.mean(cv_scores),
                  'precision': np.mean(cv_precision),
                  'recall': np.mean(cv_recall)}

        self.mlp_metrics = values
        return values
    def predict_mlp(self, matrix = None):  # matrix -> features
        print("WindowClassifier => predictMLP()")
        # Make prediction with svm
        Y_pred = self._mlp.predict(self.X_test if matrix is None else matrix)
        return Y_pred

    def gaupc(self, CV):
        # Train model with a specified cv
        cv_scores = cross_val_score(self._gaupc, self._X, self._Y, cv=CV, scoring='accuracy')
        cv_precision = cross_val_score(self._gaupc, self._X, self._Y, cv=CV, scoring='precision')
        cv_recall = cross_val_score(self._gaupc, self._X, self._Y, cv=CV, scoring='recall')

        values = {'accuracy': np.mean(cv_scores),
                  'precision': np.mean(cv_precision),
                  'recall': np.mean(cv_recall)}

        self.gaupc_metrics = values
        return values
    def predict_gaupc(self, matrix = None):  # matrix -> features
        print("WindowClassifier => predictGPC()")
        # Make prediction with svm
        Y_pred = self._gaupc.predict(self.X_test if matrix is None else matrix)
        return Y_pred

    def detc(self, CV):
        # Train model with a specified cv
        cv_scores = cross_val_score(self._detc, self._X, self._Y, cv=CV, scoring='accuracy')
        cv_precision = cross_val_score(self._detc, self._X, self._Y, cv=CV, scoring='precision')
        cv_recall = cross_val_score(self._detc, self._X, self._Y, cv=CV, scoring='recall')

        values = {'accuracy': np.mean(cv_scores),
                  'precision': np.mean(cv_precision),
                  'recall': np.mean(cv_recall)}
        self.detc_metrics = values
        return values

    def predict_detc(self, matrix = None):  # matrix -> features
        print("WindowClassifier => predictDT()")
        # Make prediction with svm
        Y_pred = self._detc.predict(self.X_test if matrix is None else matrix)
        return Y_pred

    def adab(self, CV):
        # Train model with a specified cv
        cv_scores = cross_val_score(self._adab, self._X, self._Y, cv=CV, scoring='accuracy')
        cv_precision = cross_val_score(self._adab, self._X, self._Y, cv=CV, scoring='precision')
        cv_recall = cross_val_score(self._adab, self._X, self._Y, cv=CV, scoring='recall')

        values = {'accuracy': np.mean(cv_scores),
                  'precision': np.mean(cv_precision),
                  'recall': np.mean(cv_recall)}
        self.adab_metrics = values
        return values

    def predict_adab(self, matrix = None):  # matrix -> features
        print("WindowClassifier => predictADA()")
        # Make prediction with svm
        Y_pred = self._adab.predict(self.X_test if matrix is None else matrix)
        return Y_pred


    def ganb(self, CV):
        # Train model with a specified cv
        cv_scores = cross_val_score(self._ganb, self._X, self._Y, cv=CV, scoring='accuracy')
        cv_precision = cross_val_score(self._ganb, self._X, self._Y, cv=CV, scoring='precision')
        cv_recall = cross_val_score(self._ganb, self._X, self._Y, cv=CV, scoring='recall')

        values = {'accuracy': np.mean(cv_scores),
                  'precision': np.mean(cv_precision),
                  'recall': np.mean(cv_recall)}
        self.ganb_metrics = values
        return values

    def predict_ganb(self, matrix = None):  # matrix -> features
        print("WindowClassifier => predictGNB()")
        # Make prediction with svm
        Y_pred = self._ganb.predict(self.X_test if matrix is None else matrix)
        return Y_pred


    def qud(self, CV):
        # Train model with a specified cv
        cv_scores = cross_val_score(self._qud, self._X, self._Y, cv=CV, scoring='accuracy')
        cv_precision = cross_val_score(self._qud, self._X, self._Y, cv=CV, scoring='precision')
        cv_recall = cross_val_score(self._qud, self._X, self._Y, cv=CV, scoring='recall')

        values = {'accuracy': np.mean(cv_scores),
                  'precision': np.mean(cv_precision),
                  'recall': np.mean(cv_recall)}
        self.qud_metrics = values
        return values

    def predict_qud(self, matrix = None):  # matrix -> features
        print("WindowClassifier => predictQUD()")
        # Make prediction with svm
        Y_pred = self._qud.predict(self.X_test if matrix is None else matrix)
        return Y_pred


    '''Tuning'''
    def pick_the_best_knn(self):
        knn2 = neighbors.KNeighborsClassifier()
        param_grid = {"n_neighbors": np.arange(1, 25)}
        knn_gscv = GridSearchCV(knn2, param_grid, cv=5)
        knn_gscv.fit(self._X, self._Y)
        return knn_gscv.best_params_

    def pick_the_best_random_forest(self):
        rff= RandomForestRegressor()
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        min_samples_split = [2, 5, 10]
        min_samples_leaf = [1, 2, 4]
        bootstrap = [True, False]
        random_grid = {'n_estimators': n_estimators,
                       'max_features': max_features,
                       'max_depth': max_depth,
                       'min_samples_split': min_samples_split,
                       'min_samples_leaf': min_samples_leaf,
                       'bootstrap': bootstrap}
        rf_random = RandomizedSearchCV(estimator=rff, param_distributions=random_grid, n_iter=100, cv=3, verbose=2,
                                       random_state=42, n_jobs=-1)
        rf_random.fit(self._X,self._Y)
        print(rf_random.best_params_)