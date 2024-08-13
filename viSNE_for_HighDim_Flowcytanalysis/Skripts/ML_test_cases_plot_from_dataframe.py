

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import ML_tsne_lib as MLtsne


df= pd.read_excel(r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\TestResult\Test_simpel_1.xlsx")


cols_to_plot=['Cyanobacteria', 'Diatom',
       'Grünalge', 'MP', 'Sediment']

MLtsne.create_stacked_barplot_from_ML(input_df=df,cols_to_plot=cols_to_plot)    