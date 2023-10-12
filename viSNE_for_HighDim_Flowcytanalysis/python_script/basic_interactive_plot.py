import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import display

# Sample DataFrame with values and columns for coloring
data = {'X': np.random.rand(50),
        'Y': np.random.rand(50),
        'Color_Column1': np.random.randint(1, 4, size=50),
        'Color_Column2': np.random.randint(1, 4, size=50)}
df = pd.DataFrame(data)

# Create a figure and axis for the scatter plot
fig, ax = plt.subplots(figsize=(8, 6))

# Initialize a colormap
cmap = plt.cm.get_cmap('viridis', df['Color_Column1'].nunique())

# Function to plot scatter based on the selected color column
def plot_scatter(color_column):
    ax.clear()
    scatter = ax.scatter(df['X'], df['Y'], c=df[color_column], cmap=cmap, s=100)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(f'Scatter Plot Colored by {color_column}')
    plt.colorbar(scatter, label=color_column)
    plt.show()

# Create a dropdown widget for selecting the color column
color_options = df.columns[df.columns.str.startswith('Color')]
color_dropdown = widgets.Dropdown(options=color_options, description='Color by')

# Create an interactive plot
interact(plot_scatter, color_column=color_dropdown)

# Display the interactive plot
display()
