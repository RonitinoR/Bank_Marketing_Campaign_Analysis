from loading import load_data
from data_tranformation import Transformation
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings('ignore')
class model(Transformation):
    def __init__(self, filename: str) -> None:
        super().__init__(filename)
        self.train , self.test = self.TTSplit()
        
        #for debugging purpose
        #print("Columns in the train dataframe: ", list(self.train.columns))
        
        if 'y' in self.train.columns:
            self.X_train = self.train.drop(columns = ['y'])
            self.X_test = self.test.drop(columns = ['y'])
            self.Y_train = self.train['y']
            self.Y_test = self.test['y']
        else: raise KeyError("'y' column is not found in the dataset.")

    def decisiontree(self):
        #icluding all the hyper parameter tuning as well pruning the tree for no overfitting
        pipeline = Pipeline(steps = [('preprocessor', self.chaining()),
            ('classifier', DecisionTreeClassifier())
            ])
        
        #assigning the parameters for hyper-parameter tuning
        parameter_tune = {
            'classifier__max_depth' : [None, 10, 20, 30],
            'classifier__min_samples_split' : [2, 5, 10],
            'classifier__min_samples_leaf' : [1, 2, 4],
            'classifier__criterion' : ['gini', 'entropy'],
            'classifier__ccp_alpha' : [0.0, 0.01, 0.05] #cost complexity pruning
        }

        grid_search = GridSearchCV(pipeline, parameter_tune, cv = 5, scoring = 'accuracy', n_jobs = -1)
        grid_search.fit(self.X_train, self.Y_train)

        #updating the pipeline with the best_estimator found
        pipeline = grid_search.best_estimator_

        y_prediction = pipeline.predict(self.X_test)

        accuracy = accuracy_score(self.Y_test, y_prediction)
        report = classification_report(self.Y_test, y_prediction)
        matrix = confusion_matrix(self.Y_test, y_prediction)

        return {
            'accuracy' : accuracy,
            'classification_report' : report,
            'confusion_matrix': matrix
        }