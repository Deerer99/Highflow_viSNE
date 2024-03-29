
# %%
import ML_tsne_lib  as MLtsne
import pandas as pd
from sklearn.manifold import TSNE


dir_path = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\fcm_field_data"
dir_path_ML = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\SingleSpeciesData"


fcs_data=MLtsne.load_fcs_from_dir(directory_path=dir_path,label_data_frames=True)

fcs_data_for_ML = MLtsne.load_csv_with_ending(dir_path_ML,ending="")

fcs_data_subsample = MLtsne.subsample_from_list_of_df(fcs_data,subsampling_number=1000)

transformed_fcs_data = MLtsne.asinh_transform(fcs_data_subsample,min_max_trans=True)

# %%
rf_class = MLtsne.develop_ML_model_RF(labeld_dfs=fcs_data_for_ML,test_size=0.2)

# %%
column_mapping = {
    'FL1 510/20': 'FL1_Area',
    'FL10 525': 'FL10_area',
    'FL2 542/27': 'FL2_area',
    'FL3 575': 'FL3_area',
    'FL4 620/30': 'FL4_area',
    'FL5 695/30': 'FL5_area',
    'FL6 660': 'FL6_area',
    'FL7 725/20': 'FL7_area',
    'FL8 755': 'FL8_area',
    'FL9 450/50': 'FL9_area',
    "FS":"FS_area",
    "SS": "SS_area"

}

filename = transformed_fcs_data["Label"]
transformed_fcs_data = transformed_fcs_data.iloc[:,:-1]
transformed_fcs_data = MLtsne.rename_columns(transformed_fcs_data,column_mapping)
Labels = rf_class.predict(transformed_fcs_data)
transformed_fcs_data["Label"] = Labels
transformed_fcs_data["filename"] = filename

#%%

print(transformed_fcs_data)

normalized_matrix_with_labels, label_value_dict = MLtsne.generate_unique_ML_values_from_labels(transformed_fcs_data,max_t=0.5)
input_matrix_for_tsne = normalized_matrix_with_labels.drop("Label",axis=1)
tnse_result = TSNE(perplexity=30,verbose=True).fit_transform(input_matrix_for_tsne)

MLtsne.create_visne_with_dropdown(tsne_result=tnse_result,subsampling_df=normalized_matrix_with_labels)


# %%
feature_names = fcs_data_for_ML[0].columns
feature_names = feature_names[:-1]
MLtsne.get_feature_importance(rf_class,feature_names)
# %%
