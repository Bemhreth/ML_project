import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
#loading the data
oecd_bli = pd.read_csv("oecd_bli_2015.csv", thousands=',')
gdp_per_capital = pd.read_csv("gdp_per_capita.csv",delimiter='\t',encoding='latin1',na_values='n/a', index_col="Country")

#preparing the data for calculation
country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]y = np.c_[country_stats["Life satisfaction"]]
y = np.c_[country_stats["Life satisfaction"]]

#visualizing
country_stats.plot(kind='scatter', x="GDP per capita",y="life satisfaction")
plt.show()

#selection of the model
model=sklearn.linear_model.LinearRegression()
# model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3) #use this to pridict by the instance based learning

#tranning the model
model.fit(x,y)

#make a pridiction for another country_
x_new = [["22587"]]#enter the gdp of the country you want to train
print(model.predict(x_new))
#transmission
