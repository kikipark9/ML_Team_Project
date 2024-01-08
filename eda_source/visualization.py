import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.graphics.mosaicplot import mosaic


def create_mosaic_plot(df, col1, col2):

    """
+    Generates a mosaic plot using the provided data frame, column 1, and column 2.
+
+    Parameters:
+    - df: The data frame containing the data.
+    - col1: The name of the first column.
+    - col2: The name of the second column.
+
+    Returns:
+    None
+    """
    col1_values = df[col1].unique()
    col2_values = df[col2].unique()

    color_palette = plt.cm.get_cmap('Pastel1', len(col1_values))
    colors = {str(val): color_palette(i) for i, val in enumerate(col1_values)}
    
    alphas = np.linspace(0.3, 0.7, len(col2_values))
    alpha_dict = {str(val): alpha for val, alpha in zip(col2_values, alphas)}
    
    def props(key):
        color = colors.get(str(key[0]), (0, 0, 0))
        alpha = alpha_dict.get(str(key[1]), 1)
        return {'color': color[:3] + (alpha,)}

    def labels(key):
        lab1 = {str(val): str(val) for val in col1_values}
        lab2 = {str(val): str(val) for val in col2_values} if col2 != 'Exited' else {'1': 'exited', '0': 'not exited'}

        label = f'{lab1.get(key[0], "unknown")} & {lab2.get(key[1], "unknown")}'
        return label

    fig, ax = plt.subplots(figsize=(10, 8))
    mosaic(df, [col1, col2], properties=props, labelizer=labels, ax=ax)
    ax.set_xlabel(col1)
    ax.set_ylabel(col2)
    ax.grid(False)
    plt.show()


def draw_countplot(df, x, hue, figsize=(10, 8)):
    """
+    Generate a countplot using seaborn library.
+
+    Parameters:
+        - df (pandas.DataFrame): The DataFrame containing the data.
+        - x (str): The column in the DataFrame to plot on the x-axis.
+        - hue (str): The column in the DataFrame to group the data by.
+        - figsize (tuple, optional): The size of the figure. Defaults to (10, 8).
+
+    Returns:
+        None
+
+    Raises:
+        None
+    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.countplot(x=x, hue=hue, data=df, palette=["#abc9ea","#98daa7","#f3aba8","#d3c3f7","#f3f3af","#c0ebe9"])
    ax.bar_label(ax.containers[0], label_type='edge', fontsize=11)
    ax.bar_label(ax.containers[1], label_type='edge', fontsize=11)
    plt.show()


def draw_histplot(df, x, hue=None, figsize=(10, 8), bins=25):
    """
+    Generate a histogram plot using seaborn library.
+
+    Parameters:
+        x (array-like): The data values to be plotted on the x-axis.
+        hue (str, optional): The variable used for grouping the data. Defaults to None.
+        figsize (tuple, optional): The size of the figure. Defaults to (10, 8).
+        bins (int, optional): The number of bins to use for the histogram. Defaults to 25.
+
+    Returns:
+        None
+    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.histplot(x=x, hue=hue, data=df, kde=True, bins=bins, palette=["#abc9ea","#98daa7","#f3aba8","#d3c3f7","#f3f3af","#c0ebe9"])
    plt.show()

