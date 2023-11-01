
import ML_tsne_lib  as MLtsne
import pandas as pd
from sklearn.manifold import TSNE



# import data 
dir_path_ML = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\Data_Xinje\Some Data from Xinjie Project\Algae and microplastics"

dir_path_test_data = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\Pyrenäen_Data_FCM\partialdata_fcm"
# create viSNE maps for each 
fcs_data=MLtsne.load_fcs_from_dir(directory_path=dir_path_test_data,label_data_frames=True)
fcs_data_input_tsne_one = fcs_data[0]
fcs_data_transform = MLtsne.asinh_transform(fcs_data_input_tsne_one)

fcs_data_input_tsne = fcs_data_transform.iloc[:,:-1]

tnse_result = TSNE(perplexity=30,verbose=True).fit_transform(fcs_data_input_tsne)

MLtsne.create_visne_with_dropdown(tsne_result=tnse_result,subsampling_df=fcs_data_input_tsne)

# Develop ML Model

definitions_dict = {
        "3B": 1,
        "3D": 2,
        "3G": 3,
        "A26": 4,
        "MP": 5
    }
fcs_data_for_ML_label = MLtsne.label_data_according_to_definitions(dir_path_ML,definitions_dict)
rf_class = MLtsne.develop_ML_model_RF(fcs_data_for_ML_label,test_size=0.2)
# get clusters

Labels = rf_class.predict(fcs_data_input_tsne_one)



# compare clustersizes 


# 