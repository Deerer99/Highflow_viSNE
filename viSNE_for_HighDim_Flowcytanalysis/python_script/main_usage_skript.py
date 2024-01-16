import ML_tsne_lib as MLtsne
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns

dir_path_ML_Species = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MLModelwithpop\Species"
dir_path_ML_Sediment_MP =r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MLModelwithpop\MP_Sediment"
dir_path_test_data = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\NewProtocolSingleSpecies\iLink_species\iLink_total"
dir_save=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\ML_eval\iLinktest_PopSep_com0l.xlsx"


fcs_data_for_ML_label = MLtsne.load_fcs_from_dir(dir_path_ML_Species,label_data_frames=True,data_from_matlab=True)
fcs_data_for_ML_label_MP_Sediment = MLtsne.load_data_from_structured_directory(dir_path_ML_Sediment_MP)



rf_class, conf_matrix, train_df_all, report = MLtsne.develop_ML_model_RF(fcs_data_for_ML_label_MP_Sediment,test_size=0.5,random_state=42,additional_df_non_transformed=fcs_data_for_ML_label,frac=1)




summary_df,pred_df_list= MLtsne.evaluate_dir_with_ML_classifier(dir_evaluate=dir_path_test_data,classifier=rf_class,dir_save=dir_save,subdir=False,triplicates=True,conf_interval=True)








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





                                      