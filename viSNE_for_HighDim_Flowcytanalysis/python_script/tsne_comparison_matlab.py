
import ML_tsne_lib  as MLtsne
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np


# import data 
dir_path_ML = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\Data_Xinje\Some Data from Xinjie Project\DataMicroplasticSedimentalgea"

dir_path_test_data = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\Pyrenäen_Data_FCM\partialdata_fcm"
# create viSNE maps for each 
fcs_data=MLtsne.load_fcs_from_dir(directory_path=dir_path_test_data,label_data_frames=True)
fcs_data_input_tsne_one = fcs_data[1]
fcs_data_transform = MLtsne.asinh_transform(fcs_data_input_tsne_one)

fcs_data_input_tsne = fcs_data_transform.iloc[:,:-1]
filename = fcs_data_transform["Label"]
tnse_result = TSNE(perplexity=30,verbose=True).fit_transform(fcs_data_input_tsne)

MLtsne.create_visne_with_dropdown(tsne_result=tnse_result,subsampling_df=fcs_data_input_tsne)

# Develop ML Model

definitions_dict = {
        "3B": 1,
        "3D": 2,
        "3G": 3,
        "A26": 4,
        "MP": 5,
        "Sediment": 6
    }
fcs_data_for_ML_label = MLtsne.label_data_according_to_definitions(dir_path_ML,definitions_dict)
rf_class = MLtsne.develop_ML_model_RF(fcs_data_for_ML_label,test_size=0.2)
# get clusters

Labels = rf_class.predict(fcs_data_input_tsne)

unique_labels, label_count = np.unique(Labels,return_counts=True)

for label, count in zip(unique_labels,label_count):
    print(f"Label {label} apears {count} times")
# compare clustersizes 

fcs_data_input_tsne["Label"] = Labels
fcs_data_input_tsne["filenames"] = filename

MLtsne.create_visne_with_dropdown(tsne_result=tnse_result,subsampling_df=fcs_data_input_tsne)

MLtsne.create_stacked_barplot_from_ML(fcs_data_input_tsne)
