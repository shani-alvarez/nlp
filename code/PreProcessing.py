# Data pre-processing class

import os
import glob
import numpy as np
import pandas as pd
from datetime import timezone, datetime
import pytz

class PreProcessing:
    
    def __init__(self, directory):
        self.directory = directory # Working directory
    
    def loadMergeDatasets(self):
        #Load datasets
        path = self.directory
        list_dfs = []
        csv_files = glob.glob(os.path.join(self.directory, "*.csv"))
        print("Loading datasets.............................", "\n")
        for file in csv_files:
            df = pd.read_csv(file)
            list_dfs.append(df)
            print('Folder:', file)
            print('File Name:', file.split("\\")[-1])
            print('Rows: ' + str(df.shape[0]) + ', columns: ' + str(df.shape[1]), '\n')
        #Merge datasets
        data = pd.concat(list_dfs)
        print('Shape of merged dataset: ' + str(data.shape), '\n')
        return data
    
    def cleanDataset(self, data):
        #Eliminates repeated tweets by id
        df = data.drop_duplicates(['status_id'])
        print('Non-duplicated tweets: ' + str(df.shape[0]) + ', and columns: ' + str(df.shape[1]), '\n')
        df = df[df.user_id == 634191591]
        print('UNODC Mexico tweets with non-duplicated rows: ' + str(df.shape[0]) + ', and columns: ' + str(df.shape[1]), '\n')
        print('Unique id user: ' + str(df.user_id.unique().shape[0]), '\n')
        print("NaNs in 'text' column: " + str(df.text.isna().sum()), '\n')
        return df

    # Dates to datetime
    def datesToMexicoCityTimezone(self, data):
        data['created_at'] = pd.to_datetime(data['created_at'])
        # Convert to local time (MXCity). The point in time does not change, only the associated timezone
        my_timezone = pytz.timezone('America/Mexico_City')
        data["date_time_mxc"] = data.created_at.dt.tz_convert(my_timezone)
        # Crea columna con fechas
        data["date_mxc"] = pd.to_datetime(data["date_time_mxc"].dt.date).dt.date
        # Remover NAs
        print("Number of NAs in the datetime column:")
        print(data.isna().sum().head(3), "\n")
        df = data.dropna(subset=['created_at']).reset_index(drop=True)
        print("Number of NAs in the datetime column after removal of NAs:")
        print(df.isna().sum().head(3), "\n")
        print('Shape of merged dataset: ' + str(df.shape) + "\n")
        return df
    
    # Establece las semanas del año a la que pertenecen los tweets
    def weekNumber(self, data):
        df = data.copy()
        df["week_number"] = df.date_time_mxc.dt.isocalendar().week
        return df
    
    # Calculate rates for the favorites and retweets fields
    def ratesFavoritesRetweets(self, data):
        df = data.copy()
        df["fav_rate"] = df["favorite_count"] / df.favorite_count.sum()
        df["ret_rate"] = df["retweet_count"] / df.retweet_count.sum()
        return df
    
    # Tokeniza los hashtags de cada tweet
    def hashtagsCountAndRates(self, data):
        df = data.copy()
        
        def countHashtagsList(list_hashtags):
            count = 0;
            if list_hashtags[0] == 'No hashtag':
                count = 0
            else:
                count = len(list_hashtags)
            return count
        
        #Substitute NAs
        df["hashtags"] = df["hashtags"].fillna("No hashtag")
        df["hashtags_sep"] = df["hashtags"].apply(lambda x: x.split(sep = '|'))
        df["hashtags_count"] = df["hashtags_sep"].apply(lambda x: countHashtagsList(x))
        df["hashtags_rate"] =  df["hashtags_count"] /  df["hashtags_count"].sum()
        return df