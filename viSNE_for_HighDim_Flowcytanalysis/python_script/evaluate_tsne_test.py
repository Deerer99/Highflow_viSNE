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

dir_save=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\ML_eval\Test.xlsx"
summary_df,df= MLtsne.evaluate_dir_with_ML_classifier(dir_evaluate=dir_path_test_data,classifier=rf_class,dir_save=dir_save)

df_list= []
for data in df:
    df_trans_labels= MLtsne.asinh_transform(data)
    df_trans_labels["Label"]= data["Label"]
    df_numeric_values,label_value_dict = MLtsne.generate_unique_ML_values_from_labels(normalized_matrix_with_labels=df_trans_labels,max_t=5)

    df_list.append(df_numeric_values)

tsne_result_list = []
input_list =[]
for i in range(0,len(df_numeric_values),3):

    input = pd.concat(df_list[i:i+3])
    input = input.drop(["filename","Label"],axis=1)
    tsne_result = TSNE(perplexity=30,verbose=True).fit_transform(input)
    tsne_result_list.append(tsne_result)
    input_list.append(input)

MLtsne.create_visne_with_dropdown(tsne_result_list[0],input_list[0])