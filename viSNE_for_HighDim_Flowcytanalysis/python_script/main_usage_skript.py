import ML_tsne_lib as MLtsne
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np


dir_path_ML = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\NewProtocolSingleSpecies\iLink_species\SpeciesPopSep"
dir_path_test_data = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\NewProtocolSingleSpecies\iLink_species\iLink_total"
dir_save=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\ML_eval\iLinktest_PopSep.xlsx"

# develop ML Model here with correct labels

#fcs_data_for_ML_label=MLtsne.load_data_from_structured_directory(dir_path_ML)
# MP_EPS_data = MLtsne.label_data_according_to_definitions(path_csv)
# MP_EPS_data=MP_EPS_data[0]
# MP_EPS_data=MP_EPS_data.drop(MP_EPS_data.columns[MP_EPS_data.columns.str.contains("unnamed",case=False)],axis=1)
fcs_data_for_ML_label = MLtsne.load_fcs_from_dir(dir_path_ML,label_data_frames=True,data_from_matlab=True)
#MLtsne.export_loaded_fcs_data_col_A_as_filename_csv(fcs_data_list=fcs_data_for_ML_label,dir_save=dir_save_excel)


rf_class = MLtsne.develop_ML_model_RF(fcs_data_for_ML_label,test_size=0.3,random_state=42,additional_df_non_transformed=None)

summary_df,df= MLtsne.evaluate_dir_with_ML_classifier(dir_evaluate=dir_path_test_data,classifier=rf_class,dir_save=dir_save,subdir=False,triplicates=True)

MLtsne.ML_statistic(df,dir_save=dir_save)

df_list= []
for data in df:
    df_trans_labels= MLtsne.asinh_transform(data)
    df_trans_labels["Label"]= data["Label"]
    df_numeric_values,label_value_dict = MLtsne.generate_unique_ML_values_from_labels(normalized_matrix_with_labels=df_trans_labels,max_t=5)

    df_list.append(df_numeric_values)

tsne_result_list = []
input_list =[]

for i in range(0,len(df_numeric_values),3):

    input_tsne = pd.concat(df_list[i:i+3])
    input_tsne = input_tsne.drop(["filename","Label"],axis=1)
    tsne_result = TSNE(perplexity=30,verbose=True).fit_transform(input)
    tsne_result_list.append(tsne_result)
    input_list.append(input)

MLtsne.create_visne_with_dropdown(tsne_result_list[0],input_list[0])