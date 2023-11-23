import ML_tsne_lib as MLtsne
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np

#load
dir_path_ML = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\ML_Model_TOTAL"

dir_path_test_data = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\Pyrenäen_Data_FCM\partialdata_fcm"

# develop ML Model here with correct labels

fcs_data_for_ML_label=MLtsne.load_data_from_structured_directory(dir_path_ML)
rf_class = MLtsne.develop_ML_model_RF(fcs_data_for_ML_label,test_size=0.2)
fcs_data=MLtsne.load_fcs_from_dir(directory_path=dir_path_test_data,label_data_frames=True)


summary_df,df= MLtsne.evaluate_dir_with_ML_classifier(dir_evaluate=dir_path_test_data,classifier=rf_class,dir_save=None)
summary_df.to_excel(r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\ML_eval\Test.xlsx")
MLtsne.create_stacked_barplot_from_ML(df=summary_df)