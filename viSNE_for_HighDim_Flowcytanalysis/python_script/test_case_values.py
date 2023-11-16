
import ML_tsne_lib  as MLtsne
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt


dir_path_test_data = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\Pyrenäen_Data_FCM\partialdata_fcm"


fcs_df,fcs_data=MLtsne.load_fcs_from_dir(directory_path=dir_path_test_data,label_data_frames=False)


