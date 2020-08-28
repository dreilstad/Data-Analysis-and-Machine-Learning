# Common imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error
import os

# Where to save the figures and data files
PROJECT_ROOT_DIR = "Results"
FIGURE_ID = "Results/FigureFiles"
DATA_ID = "DataFiles/"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)

if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)

if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)

def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)

def data_path(dat_id):
    return os.path.join(DATA_ID, dat_id)

def save_fig(fig_id):
    plt.savefig(image_path(fig_id) + ".png", format='png', dpi=300)

infile = open(data_path("winemag-data-130k-v2.csv"),'r')

#cleaning data
wine_data = pd.read_csv(infile, index_col=[0])
wine_data = wine_data.dropna()
wine_data = wine_data[wine_data.price < 1000]

# assign relevant data
points = wine_data['points']
price = wine_data['price']

X = np.zeros((len(price),2))
X[:,0] = 1
X[:,1] = price

linreg = LinearRegression().fit(X,points)
points_predict = linreg.predict(X)

# The mean squared error                               
print("Mean squared error: %.2f" % mean_squared_error(points, points_predict))
# Explained variance score: 1 is perfect prediction                                 
print('Variance score: %.2f' % r2_score(points, points_predict))
# Mean absolute error                                                           
print('Mean absolute error: %.2f' % mean_absolute_error(points, points_predict))
plt.plot(price, points_predict, "r-")
plt.plot(price, points ,'ro')
plt.axis([wine_data.min()['price'], wine_data.max()['price'], wine_data.min()['points'], wine_data.max()['points']])
plt.xlabel(r'$price$')
plt.ylabel(r'$points$')
plt.title(r'Linear Regression fit ')
save_fig("wine_price_points_predict")
plt.show()