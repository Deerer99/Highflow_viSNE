import ML_tsne_lib as MLtsne
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns


path= r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\ML_models\ML_Model_simpel_fcs"
dir_path_test_data = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\Länder\1_rawdata"
dir_save=r"c:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\Länder\2_results\Excel\Länder_V07_width_height.xlsx"



fcs_data_for_ML_label_MP_Sediment = MLtsne.load_data_from_structured_directory(path,data_from_matlab=True)



rf_class, conf_matrix, train_df_all, report = MLtsne.develop_ML_model_RF(fcs_data_for_ML_label_MP_Sediment,test_size=0.5,random_state=42,frac=0.2)



summary_df,pred_df_list= MLtsne.evaluate_dir_with_ML_classifier(
    dir_evaluate=dir_path_test_data,
    classifier=rf_class,
    dir_save=dir_save,
    subdir=False,
    triplicates=True,
    conf_interval=True,
    data_from_matlab=True
)








# test


#code for tsne map pf the ML predicted species
# tsne_cords = MLtsne.create_tsne_for_species_percent(summary_df=summary_df)
# df= MLtsne.add_location(summary_df)


#code for histogramms

# pred_df_all = MLtsne.asinh_transform(pd.concat(pred_df_list),min_max_trans=False)
# pure =pred_df_all.drop("filename",axis=1)
# train_df_all = train_df_all.rename(columns={"filename":"Label"})
# MLtsne.create_hist_comparison_for_experiment(train_df=train_df_all,pred_df=pure,dir_save=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\Plots experiment")






                                      