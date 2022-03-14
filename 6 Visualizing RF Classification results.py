import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")


"""
load data
"""
path_2 = "file path/results for k=2.xlsx"
path_4 = "file path/results for k=4.xlsx"
path_8 = "file path/results for k=8.xlsx"
k_2 = pd.read_excel(path_2)
k_4 = pd.read_excel(path_4)
k_8 = pd.read_excel(path_8)

"""
Create a relationship plot for each data frame
"""


def rel_plot(dataframe):
    sns.relplot(
        x='n',
        y='value',
        hue='evaluation metric',
        style='parameter tuning',
        kind='line',
        data=dataframe)
    plt.show()


"""
Show plots
"""
# rel_plot(k_2)
# rel_plot(k_4)
# rel_plot(k_8)
