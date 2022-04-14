import abc
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,scale, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from concurrent.futures import ProcessPoolExecutor
class classification:
    """description of class"""

    def __init__(self,file,models_parameters):
        self.file = file
        self.models_parameters = models_parameters

    def read_file(self):
        try:
            df = pd.read_csv(self.file)
            return df

        except FileNotFoundError as err:
            print(f"Error Message: {err.args[1]}")

    def split_data(self,df):
        for key,value in self.models_parameters.items():
            if key == "train_data":
                test_size = value.get('test_size')
                train_size = value.get('train_size')
            elif key == "X_columns":
                 features = value
            elif key == "Y_column":
                 label = value
        X = df[features].to_numpy()
        y = np.array(df[label])
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, train_size=train_size, shuffle=False)
        return X_train, X_test, y_train, y_test

    def scale_data(self,X_train,X_test):
       for key,value in self.models_parameters.items():
           if key == "scale_data":
              if value == "StandardScaler":
                 scaler = StandardScaler()
                 scaler.fit(X_train)
                 X_train = scaler.transform(X_train)
                 X_test = scaler.transform(X_test)
       return X_train, X_test

    def classifier_fit(self,X_train,y_train):
       all_models = []
       for class_key,classifiers in self.models_parameters.items():
           if class_key == "Classifiers":
              for model, models in classifiers.items():
                  if model == "DecisionTree_Classifiers":
                     with ProcessPoolExecutor() as executor:
                          futures = [executor.submit(tree.DecisionTreeClassifier(**parameters).fit,X_train, y_train) for dtree, parameters in models.items()]
                          [all_models.append(f.result()) for f in futures]
                  elif model == "RandomForest_Classifiers":
                       with ProcessPoolExecutor() as executor:
                          futures = [executor.submit(RandomForestClassifier(**parameters).fit,X_train, np.ravel(y_train)) for dtree, parameters in models.items()]
                          [all_models.append(f.result()) for f in futures]
                  elif model == "KNeighbors_Classifiers":
                       with ProcessPoolExecutor() as executor:
                          futures = [executor.submit(KNeighborsClassifier(**parameters).fit,X_train, np.ravel(y_train)) for dtree, parameters in models.items()]
                          [all_models.append(f.result()) for f in futures]
       return all_models

    def classifier_predict(self,models,X_test):
        predicted_models = []
        for model in models:
            model = model.predict(X_test)
            predicted_models.append(model)
        return predicted_models

    def classifier_score(self, models, X_test, y_test):
        scores = []
        for score in models:
            score = score.score(X_test, y_test)
            scores.append(score)
        return scores

