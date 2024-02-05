
import ML_tsne_lib as MLtsne
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import seaborn as sns
import os 

dir_path_train_data_species = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\ML_training\Models\V08_spiked_model_simpel\1_raw_data\Species"
dir_path_train_data_MP_sed=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\ML_training\Models\V08_spiked_model_simpel\1_raw_data\MP_Sediment"
dir_path_test_data = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\Länder\1_rawdata"
dir_save=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\Länder\2_results\Excel\Länder_V08.xlsx"
dir_save_hist=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\Länder\2_results\Histogramms"
dir_save_conf_matrix = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\ML_training\ML_training_params\conf_matrix.xlsx"
accepted_col_names = [    
    'FL1 510/20',
    "FL10 675",
    'FL2 542/27',
    "FL3 575",
    'FL4 620/30',
    'FL5 695/30',
    'FL6 660',
    'FL7 725/20',
    'FL8 755',
    'FL9 460/20',
    "FS",
    "SS"
]

fcs_data_for_ML_label = MLtsne.load_data_from_structured_directory(
    dir_path_train_data_MP_sed,data_from_matlab=False,
    accepted_col_names=accepted_col_names
)

fcs_data_for_ML_label_species = MLtsne.load_data_from_structured_directory(
        rootdir=dir_path_train_data_species,
        data_from_matlab=True,
        accepted_col_names=accepted_col_names


)

rf_class, conf_matrix, train_df_all, report = MLtsne.develop_ML_model_RF(
    fcs_data_for_ML_label,test_size=0.5,
    random_state=70,
    frac=1,
    additional_df_non_transformed=fcs_data_for_ML_label_species
)


summary_df,pred_df_list= MLtsne.evaluate_dir_with_ML_classifier(
    dir_evaluate=dir_path_test_data,classifier=rf_class,dir_save=dir_save,
    subdir=False,triplicates=False,conf_interval=True,
    transform=True,accepted_col_names=accepted_col_names,
    data_from_matlab=False
)

# disp= ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=rf_class.classes_)
# disp.plot(xticks_rotation="vertical",include_values=True)
# MLtsne.save_conf_matrix(conf_matrix=conf_matrix,rf_class=rf_class,dir_save=dir_save_conf_matrix)
# #plot hisogramms
# pred_df_all = MLtsne.asinh_transform(pd.concat(pred_df_list),min_max_trans=False)
# pure =pred_df_all.drop("filename",axis=1)
# train_df_all = train_df_all.rename(columns={"filename":"Label"})
# MLtsne.create_hist_comparison_for_experiment(train_df=train_df_all,pred_df=pure,dir_save=dir_save_hist)

# test

names= ["AR","DE","DK","Gronol","Immez","SUR4","Q","SP","Sw","Z"]
MLtsne.create_tsne_for_species_percent(summary_df=summary_df,plot=True,names=names)

subplot_identifier= ["AR","DE","DK","Gronol","Immez","SUR4","Q","SP","Sw","Z"]
fig_save_path = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\Länder\2_results\Populations"


for iden in subplot_identifier:
    fig_save= os.path.join(fig_save_path,f"{iden}.png")
    MLtsne.stacked_barplot_for_certain_rows(df=summary_df,identifer=[iden],cols_to_plot=['Cyanobacteria', 'Diatom', 'Grünalge', 'MP', 'Sediment'],fig_save=fig_save)




                                      