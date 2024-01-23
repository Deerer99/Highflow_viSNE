import ML_tsne_lib as MLtsne
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns

dir_path_ML_Species = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\ML_training\Models\V06_mp_confusion\MLModel_V06\Species"
dir_path_ML_Sediment_MP =r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\ML_training\Models\V06_mp_confusion\MLModel_V06\MP_Sediment"
dir_path_test_data = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\Länder\1_rawdata"
dir_save=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\Länder\2_results\Excel\Länder_Model_total_pop_sep.xlsx"

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

fcs_data_for_ML_label = MLtsne.load_fcs_from_dir(dir_path_ML_Species,data_from_matlab=True,accepted_col_names=accepted_col_names,label_data_frames=True)
fcs_data_for_ML_label_MP_Sediment = MLtsne.load_data_from_structured_directory(dir_path_ML_Sediment_MP)
  

rf_class, conf_matrix, train_df_all, report = MLtsne.develop_ML_model_RF(fcs_data_for_ML_label_MP_Sediment,test_size=0.5,
                                                                         random_state=70,additional_df_non_transformed=fcs_data_for_ML_label,
                                                                         frac=1)


summary_df,pred_df_list= MLtsne.evaluate_dir_with_ML_classifier(dir_evaluate=dir_path_test_data,classifier=rf_class,dir_save=dir_save,
                                                                subdir=False,triplicates=True,conf_interval=True)








# test

summary_df= MLtsne.add_location(summary_df)
loc = MLtsne.add_good_location_identifier(summary_df=summary_df)
summary_df["location"]= loc
summary_df.reset_index(inplace=True, drop=True)

water_df = pd.read_excel(r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\Water_data_pyr\Besos catchment summer 2022.xlsx",index_col=0)
clean_df = summary_df.loc[:, ~summary_df.columns.isin(["filename", "event_count","location_label"])]
water_df_clean = water_df.loc[:, water_df.iloc[0,:].isin(["DO","Temp","pH"])]
water_df_clean.reset_index(drop=True,inplace=True)
water_df_clean = water_df_clean.iloc[0:40]
correlation_matrix = pd.concat([water_df_clean,clean_df],axis=1).corr()

#code for tsne map pf the ML predicted species
# tsne_cords = MLtsne.create_tsne_for_species_percent(summary_df=summary_df)
# df= MLtsne.add_location(summary_df)


#code for histogramms

# pred_df_all = MLtsne.asinh_transform(pd.concat(pred_df_list),min_max_trans=False)
# pure =pred_df_all.drop("filename",axis=1)
# train_df_all = train_df_all.rename(columns={"filename":"Label"})
# MLtsne.create_hist_comparison_for_experiment(train_df=train_df_all,pred_df=pure,dir_save=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\Plots experiment")



MLtsne.single_fcs_file_to_csv(dir_fcs=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\NewProtocolSingleSpecies\iLink_species\7DW_2",dir_save=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\NewProtocolSingleSpecies\iLink_species\7DW_2",transform=False)





                                      