import os
import csv
import pandas as pd
from surprise import Reader, Dataset
import constants


#to load the rating csv file onto a surprise datset
def ml_small_rating_to_dataset():

    def ml_small_rating_to_pandas():
        df = pd.read_csv(constants.RATINGSMALL, sep=",")
        df.drop("timestamp", axis=1, inplace=True)
        return df

    df = ml_small_rating_to_pandas()
    reader = Reader(rating_scale=(1,5))
    dataset = Dataset.load_from_df(df[["userId", "movieId", "rating"]], reader=reader)
    return(dataset)

    


def load_model(model_filename):
    from surprise import dump
    import os
    file_name = os.path.expanduser(model_filename)
    _, loaded_model = dump.load(file_name)
    return loaded_model

    

def movieid_to_name(movieID):

    movieID_to_name = {}

    with open(constants.MOVIESSMALL, newline='', encoding='iso-8859-1') as csvfile:
        movies_reader = csv.reader(csvfile)
        next(movies_reader)
        for row in movies_reader:
            movie_ID = int(row[0])
            movie_name = row[1]
            movieID_to_name[movie_ID] = movie_name
        if int(movieID) in movieID_to_name:
            return movieID_to_name[int(movieID)]
        else:
            return ""