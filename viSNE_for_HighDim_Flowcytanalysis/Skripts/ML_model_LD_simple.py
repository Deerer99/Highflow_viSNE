import ML_tsne_lib as MLtsne
import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import statsmodels.api as sm
from matplotlib import pyplot as plt


dir_path_train_data = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\ML_models\ML_Model_TOTAL_simpel"
dir_path_test_data = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\NewProtocolSingleSpecies\iLink_species\20231206_LightDarkPeri"
dir_save=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\ML_eval\LighDarkPeri_simpel.xlsx"

MLtsne.single_fcs_file_to_csv(dir_fcs=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\NewProtocolSingleSpecies\iLink_species\4D", dir_save=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\NewProtocolSingleSpecies\iLink_species\4D",transform=False)
fcs_data = MLtsne.load_data_from_structured_directory(dir_path_train_data)



rf_class, conf_matrix, train_df_all, report = MLtsne.develop_ML_model_RF(fcs_data,test_size=0.5,random_state=42,frac=1)

# Confusion Matrix
# disp= ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=rf_class.classes_)
# disp.plot()

summary_df,pred_df_list= MLtsne.evaluate_dir_with_ML_classifier(dir_evaluate=dir_path_test_data,classifier=rf_class,dir_save=dir_save,subdir=False,triplicates=True,conf_interval=True)

# pred_df_all = MLtsne.asinh_transform(pd.concat(pred_df_list),min_max_trans=False)
# pure =pred_df_all.drop("filename",axis=1)
# train_df_all = train_df_all.rename(columns={"filename":"Label"})
# MLtsne.create_hist_comparison_for_experiment(train_df=train_df_all,pred_df=pure,dir_save=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\Plots experiment")



# test

filename_vec = summary_df["filename"]
match_dic = {
        "D" : 0,
        "L" : 1,
        "epi":"epi",
        "bio": "bio",
        "sed":"sed"

}

summary_df["cat_value"]=MLtsne.transform_light_dark_to_numerical(filename_vec=filename_vec,match_dic=match_dic)

summary_df_LD = summary_df[(summary_df["cat_value"]==0) | (summary_df["cat_value"]==1)]

X = sm.add_constant(summary_df_LD["cat_value"].astype(float))

dependet_vars= ["Cyanobacteria","MP","Grünalge","Sediment","Diatom"]

results=[]
for var in dependet_vars:
    model = sm.OLS(summary_df_LD[var],X)
    result= model.fit()
    results.append(result)

print(results[4].summary())