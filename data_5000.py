import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from matplotlib import rcParams


#tell pandas to display wide tables as pretty HTML tables
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)

names = ['budget', 'production_countries', 'release_date', 'runtime', 'vote_average']
data = pd.read_csv('tmdb_5000_movies.csv').dropna()
#print("Number of rows: %i" % data.shape[0])
print(data.head(10))  # Это выведет 5 первых строчек