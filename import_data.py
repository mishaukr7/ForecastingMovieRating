from pandas import read_csv, DataFrame
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt



data = read_csv("IMDB-Movie-Data.csv")

data.head()

data = data.drop(['Rank', 'Title', 'Director', 'Revenue (Millions)', 'Genre', 'Description', 'Actors', 'Votes', 'Metascore'], axis=1)


trg = data['Rating']
trn = data.drop(['Rating'], axis = 1)

models = [LinearRegression(), # метод наименьших квадратов
          RandomForestRegressor(n_estimators=100, max_features ='sqrt'), # случайный лес
	      KNeighborsRegressor(n_neighbors=6), # метод ближайших соседей
	      SVR(kernel='linear'), # метод опорных векторов с линейным ядром
	      LogisticRegression() # логистическая регрессия
	          ]

Xtrn, Xtest, Ytrn, Ytest = train_test_split(trn, trg, test_size=0.4)
'''
TestModels = DataFrame()
tmp = {}
for model in models:
    m = str(model)
    tmp['Model'] = m[:m.index('(')]
    for i in range(Ytrn.shape[1]):
        model.fit(Xtrn, Ytrn[:, i])
        tmp['R2_Y%s' % str(i + 1)] = r2_score(Ytest[:, 0], model.predict(Xtest))
    TestModels = TestModels.append([tmp])
TestModels.set_index('Model', inplace=True)

fig, axes = plt.subplots(ncols=2, figsize=(10, 4))
TestModels.R2_Y1.plot(ax=axes[0], kind='bar', title='R2_Y1')
TestModels.R2_Y2.plot(ax=axes[1], kind='bar', color='green', title='R2_Y2')
'''

model = models[0]
model.fit(Xtrn, Ytrn)
print(model.score(Xtrn, Ytrn))

