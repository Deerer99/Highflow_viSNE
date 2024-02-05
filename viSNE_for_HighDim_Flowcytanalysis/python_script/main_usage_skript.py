import ML_tsne_lib as MLtsne
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns

dir_path_ML_Species = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\ML_training\Models\V06_mp_confusion\MLModel_V06\Species"
dir_path_ML_Sediment_MP =r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\ML_training\Models\V06_mp_confusion\MLModel_V06\MP_Sediment"
dir_path_test_data = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\ML_training\TestCases"
dir_save=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\biofilm_control_no_sed_no_mp\ML_result\biofilm_control_V06.xlsx"

dir_save_conf_matrix = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\ML_training\Models\V06_mp_confusion\Preformance\conf_matrix.xlsx"
accepted_col_names = [    'FL1 510/20',
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
    "SS"]

fcs_data_for_ML_label = MLtsne.load_data_from_structured_directory(dir_path_ML_Species,data_from_matlab=True,accepted_col_names=accepted_col_names)
fcs_data_for_ML_label_MP_Sediment = MLtsne.load_data_from_structured_directory(dir_path_ML_Sediment_MP)
  

rf_class, conf_matrix, train_df_all, report = MLtsne.develop_ML_model_RF(fcs_data_for_ML_label_MP_Sediment,test_size=0.5,
                                                                         random_state=70,additional_df_non_transformed=fcs_data_for_ML_label,
                                                                         frac=1)


summary_df,pred_df_list= MLtsne.evaluate_dir_with_ML_classifier(dir_evaluate=dir_path_test_data,classifier=rf_class,dir_save=dir_save,
                                                                subdir=False,triplicates=False,conf_interval=True,transform=False,accepted_col_names=accepted_col_names,data_from_matlab=True)



#code for tsne map pf the ML predicted species
# tsne_cords = MLtsne.create_tsne_for_species_percent(summary_df=summary_df)
# df= MLtsne.add_location(summary_df)


#code for histogramms

# pred_df_all = MLtsne.asinh_transform(pd.concat(pred_df_list),min_max_trans=False)
# pure =pred_df_all.drop("filename",axis=1)
# train_df_all = train_df_all.rename(columns={"filename":"Label"})
# MLtsne.create_hist_comparison_for_experiment(train_df=train_df_all,pred_df=pure,dir_save=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\Plots experiment")





                                      