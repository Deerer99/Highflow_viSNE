import ML_tsne_lib as MLtsne
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



dir_path_train_data = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\ML_models\ML_Model_TOTAL_simpel"
dir_path_test_data = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\NewProtocolSingleSpecies\iLink_species\20231206_LightDarkPeri"
dir_save=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\ML_eval\LighDarkPeri_simpel.xlsx"


fcs_data = MLtsne.load_data_from_structured_directory(dir_path_train_data)



rf_class, conf_matrix, train_df_all, report = MLtsne.develop_ML_model_RF(fcs_data,test_size=0.5,random_state=42,frac=1)

# Confusion Matrix
# disp= ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=rf_class.classes_)
# disp.plot()

summary_df,pred_df_list= MLtsne.evaluate_dir_with_ML_classifier(dir_evaluate=dir_path_test_data,classifier=rf_class,dir_save=dir_save,subdir=False,triplicates=True,conf_interval=True)

pred_df_all = MLtsne.asinh_transform(pd.concat(pred_df_list),min_max_trans=False)
pure =pred_df_all.drop("filename",axis=1)
train_df_all = train_df_all.rename(columns={"filename":"Label"})
MLtsne.create_hist_comparison_for_experiment(train_df=train_df_all,pred_df=pure,dir_save=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\Plots experiment")



# test

summary_df_loc= MLtsne.add_location(summary_df)
loc = MLtsne.add_good_location_identifier(summary_df=summary_df)
summary_df["location_total"]= loc
loc_meter = MLtsne.create_meter_from_location(loc_vector=summary_df_loc["location"])
summary_df["location from WWTP"] = loc_meter
summary_df.reset_index(inplace=True, drop=True)
clean_df = summary_df.loc[:,summary_df.columns.isin(["location_total","location from WWTP","Cyanobacteria","Diatom","MP","Sediment","Grünalge"])]
clean_df_sorted = clean_df.sort_values("location_total")
clean_df_sorted.reset_index(inplace=True,drop=True)


water_df = pd.read_excel(r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\Water_data_pyr\Besos catchment summer 2022.xlsx",index_col=0,header=None)
water_df.columns=water_df.iloc[1,:]
water_df.reset_index(inplace=True,drop=True)
water_df= water_df.iloc[2:,:]

water_df_clean = water_df.loc[:, water_df.columns.isin(["DO","Temp","pH","Discharge","EC"])]
water_df_clean.reset_index(drop=True,inplace=True)
water_df_clean = water_df_clean.iloc[0:40]
water_df_clean.drop(index=28,inplace=True)
water_df_clean.reset_index(inplace=True,drop=True)
water_df_clean= water_df_clean.astype(float)



clean_df_sorted.drop("location_total",inplace=True,axis=1)
concat_df = pd.concat([clean_df_sorted,water_df_clean],axis=1)


corr_matrix =concat_df.corr()
corr_matrix_rounded = corr_matrix.round(2)
sns.heatmap(corr_matrix_rounded, cmap="Blues", annot=True)
           