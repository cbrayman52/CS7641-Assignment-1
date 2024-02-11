In order to run this code, all you have to do is run the Submission.py file.

This will perform all experiments and generate all images used in the report.
Images are saved in the 'Images' directory including a subfolder for the specific model being analyzed.

The Wine Quality Dataset was provided as a csv. However it can also be found online here:
https://www.kaggle.com/datasets/uciml/red-wine-quality-cortez-et-al-2009

Note: Running the code will take 1-2 hours to fully compile depending on the machine. 
Comment out the function calls in the "Training and Testing Error Rates" section to speed up run time.

Libraries used:
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from imblearn.over_sampling import SMOTE
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score