import pandas as pd
from loading import load_data
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

class Transformation(load_data):
    def __init__(self, filename: str) -> None:
        super().__init__(filename)
    
    def labelencoding(self, df: pd.DataFrame, Column1: str) -> pd.DataFrame:
        le = LabelEncoder()
        #fit and tranform the column, label encoding them
        df[Column1] = le.fit_transform(df[Column1])
        return df
    def categorical(self) -> Pipeline:
        #creating a pipeline to transform the categorical variables
        #performing one hot encoding and then chi-squared test for dependency for dimensionality
        categorical_transformer = Pipeline(
            steps = [('encoder', OneHotEncoder(handle_unknown='ignore')),
                     ('selector', SelectPercentile(chi2, percentile=50)),
                     ]
        )
        return categorical_transformer
    
    def numerical(self) -> Pipeline:
        #Performing simple imputer and standard scalar later for numerical columns
        numerical_transformer = Pipeline(
            steps = [('imputer', SimpleImputer(strategy='median')),
                     ('scaler', StandardScaler())]
        )
        return numerical_transformer
   
    def chaining(self) -> ColumnTransformer:
        #setting the numerical and categorical columns
        numeric_features = ['age','balance','duration','campaign','pdays','previous']
        categorical_features = ['job', 'marital', 'education', 'default', 
        'housing', 'loan', 'contact', 'month', 'poutcome']
        
        # creating a pipeline that can set up a preprocessor chaining the previous transformers
        preprocessor = ColumnTransformer(
            transformers = [
                ('num', self.numerical(), numeric_features),
                ('cat', self.categorical(), categorical_features),
            ]
        )

        return preprocessor
    
    def TTSplit(self):
        #splitting the data into train and test 
        df = self.df
        train, test = train_test_split(df, test_size = 0.2, random_state = 0)

        return train, test
    