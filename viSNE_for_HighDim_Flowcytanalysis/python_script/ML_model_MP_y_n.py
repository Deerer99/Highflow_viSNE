import ML_tsne_lib as MLtsne
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns



dir_path_MP = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\ML_microplastic_yn\MP"
dir_path_non_MP= r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\ML_microplastic_yn\Non_MP"

dir_path = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\ML_microplastic_yn"
dir_path_test_data = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\NewProtocolSingleSpecies\iLink_species\iLink_total"
dir_save=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\ML_eval\iLinktest_MP_nonMP.xlsx"


fcs_data = MLtsne.load_data_from_structured_directory(r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\ML_microplastic_yn")



rf_class, conf_matrix, train_df_all, report = MLtsne.develop_ML_model_RF(fcs_data,test_size=0.5,random_state=42,frac=1)




summary_df,pred_df_list= MLtsne.evaluate_dir_with_ML_classifier(dir_evaluate=dir_path_test_data,classifier=rf_class,dir_save=dir_save,subdir=False,triplicates=True,conf_interval=True)

pred_df_all = MLtsne.asinh_transform(pd.concat(pred_df_list),min_max_trans=False)
pure =pred_df_all.drop("filename",axis=1)
train_df_all = train_df_all.rename(columns={"filename":"Label"})
MLtsne.create_hist_comparison_for_experiment(train_df=train_df_all,pred_df=pure,dir_save=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\Plots experiment")



# test

summary_df= MLtsne.add_location(summary_df)
loc = MLtsne.add_good_location_identifier(summary_df=summary_df)
summary_df["location"]= loc
summary_df.reset_index(inplace=True, drop=True)

water_df = pd.read_excel(r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\Water_data_pyr\Besos catchment summer 2022.xlsx",index_col=0)
clean_df = summary_df.loc[:, ~summary_df.columns.isin(["filename", "event_count","location_label"])]
clean_df_sorted = clean_df.sort_values("location")
clean_df_sorted.reset_index(inplace=True,drop=True)

location_MP_df = location_MP_df= clean_df_sorted.iloc[:,0:2]

location_MP_df.drop(index=18,inplace=True)
location_MP_df.reset_index(inplace=True,drop=True)




water_df_clean = water_df.loc[:, water_df.iloc[0,:].isin(["DO","Temp","pH"])]
water_df_clean.reset_index(drop=True,inplace=True)
water_df_clean = water_df_clean.iloc[0:40]
water_df_clean.drop(index=28,inplace=True)
water_df_clean=water_df_clean[1:]
water_df_clean.reset_index(inplace=True,drop=True)


concat_df = pd.concat([location_MP_df,water_df_clean],axis=1)
concat_df_float = concat_df.astype(float)
col1 = concat_df_float["-"]     
col2 = concat_df_float["MP"]

col1.corr(col2)
                                      