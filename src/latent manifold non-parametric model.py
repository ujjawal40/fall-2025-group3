import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('sub_sample.csv')

# Explore the data
print(data.head(5))