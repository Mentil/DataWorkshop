import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report
from sklearn.linear_model import LinearRegression

import tensorflow as tf

data= make_regression(n_samples=1000, n_features=20, n_informative=15, n_targets=1)
