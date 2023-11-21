import ML_tsne_lib  as MLtsne
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np

#load
dir_path_ML = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\NewProtocolSingleSpecies\iLink_species\Species"

dir_path_test_data = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\Pyrenäen_Data_FCM\partialdata_fcm"

# develop ML Model here with correct labels
definitions_dict = {
        "3B": "Cyanobacteria",
        "3D": "Diatom",
        "3G": "GreenAlgea",
        "A26": "AlgeaMix",
        "MP": "MP",
        "Sediment": "Sediment_low_TOC"
    }


fcs_data_for_ML_label = MLtsne.label_data_according_to_definitions(dir_path_ML,definitions_dict)
rf_class = MLtsne.develop_ML_model_RF(fcs_data_for_ML_label,test_size=0.2)
fcs_data=MLtsne.load_fcs_from_dir(directory_path=dir_path_test_data,label_data_frames=True)

df= MLtsne.evaluate_dir_with_ML_classifier(dir_evaluate=dir_path_test_data,classifier=rf_class,dir_save=None,definitions_dict=None)

MLtsne.create_stacked_barplot_from_ML(df=df)