import pandas as pd
from scipy import stats

class load_data:
    #setting up the constructors
    def __init__(self, filename) -> None:
        self.__filename = filename
        self.df = self.load()

    def load(self) -> pd.DataFrame:
        data = pd.read_csv(self.__filename, delimiter= ";")
        return data
    
    def clean(self) -> pd.DataFrame:
        #dealing with outliers using z-transform method
        df = self.df.copy()

        #removing any whitespaces from columns
        df.columns = df.columns.str.strip()
        df['zscores'] = stats.zscore(df['balance'])
        #filtering the outliers and removing duplicate entries(if there are any)
        df = df.loc[df['zscores'].abs() <= 3 ].drop_duplicates()

        #dropping of the columns that have more than 40% of nas
        treshold = len(df) * 0.6
        df.dropna(axis=1, thresh = treshold, inplace = True) 

        #dropping the zscore column since it might skew the data a bit later
        df.drop(columns=['zscores'], inplace = True)
        return df
    
    def save_cleaned_csv(self, outputpath: str = 'src/processed_goods/data_clean.csv') -> None:
        df = self.clean()
        #creating a duplicate data copy for the cleaned data not changing the original data
        return df.to_csv(outputpath, index = False, sep = ";")
    
