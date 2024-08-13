
import ML_tsne_lib as MLtsne
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import seaborn as sns
import os 
from matplotlib import pyplot as plt

dir_path_train_data =r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\2_ML_training\2_models\1_models\V07_final_model\ML_Model_TOTAL_simpel"
dir_path_test_data = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\1_datasets\LightDarkPeri\1_raw_data"
dir_save=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\LightDarkPeri\ML_result\Excel\LighDarkPeri_simpel.xlsx"


dir_save_conf_matrix = None
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

fcs_data_for_ML_label = MLtsne.load_data_from_structured_directory(dir_path_train_data,data_from_matlab=False,accepted_col_names=accepted_col_names)

  

rf_class, conf_matrix, train_df_all, report = MLtsne.develop_ML_model_RF(fcs_data_for_ML_label,test_size=0.5,
                                                                         random_state=70,
                                                                         frac=1)


summary_df,pred_df_list= MLtsne.evaluate_dir_with_ML_classifier(dir_evaluate=dir_path_test_data,classifier=rf_class,dir_save=None,
                                                                subdir=False,triplicates=True,conf_interval=True,transform=True,accepted_col_names=accepted_col_names,data_from_matlab=False)

# disp= ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=rf_class.classes_)
# disp.plot(xticks_rotation="vertical",include_values=True)
# MLtsne.save_conf_matrix(conf_matrix=conf_matrix,rf_class=rf_class,dir_save=dir_save_conf_matrix)
# #plot hisogramms
# pred_df_all = MLtsne.asinh_transform(pd.concat(pred_df_list),min_max_trans=False)
# pure =pred_df_all.drop("filename",axis=1)
# train_df_all = train_df_all.rename(columns={"filename":"Label"})
# MLtsne.create_hist_comparison_for_experiment(train_df=train_df_all,pred_df=pure,dir_save=dir_save_hist)

# # test

# names= ["L","D","epi","sed","bio"]
# MLtsne.create_tsne_for_species_percent(summary_df=summary_df,plot=True,names=names)


# fig_save_path = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\LightDarkPeri\ML_result\Barplot"



# fig_save= os.path.join(fig_save_path,f"plot.png")
# MLtsne.stacked_barplot_for_certain_rows(df=summary_df,identifer=None,cols_to_plot=['Cyanobacteria', 'Grünalge', 'MP', 'Sediment'],fig_save=fig_save)



# analysis pca
dir_excel=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\1_datasets\LightDarkPeri\Proben Pyrenäen2022_water_data.xlsx"
def analyze_pyr_project_pca(summary_df,dir_excel):
    excel=pd.read_excel(dir_excel,[1])[1]
    excel.drop(columns=["Estany"],inplace=True)
    excel_new=excel.rename(columns={"Unnamed: 0":"merge key"})
    
    clean_df= MLtsne.remove_cf_interval_from_df(summary_df)
    clean_df['merge key'] = clean_df['filename'].str.extract(r'(\d+)')
    excel_new["merge key"]= excel_new["merge key"].astype(str)
    merged_df = clean_df.merge(excel_new, on='merge key', how='left')
    merged_df.drop(columns=["filename","event_count","merge key"],inplace=True)
    merged_df.replace("-", np.nan, inplace=True)
    merged_df = merged_df.applymap(lambda x: pd.to_numeric(x.replace(',', '.'), errors='coerce') if isinstance(x, str) else x)
    merged_df.dropna(inplace=True)
    return merged_df

merged_df= analyze_pyr_project_pca(summary_df=summary_df,dir_excel=dir_excel)
df_pca,pca=MLtsne.preform_pca_on_dataframe(merged_df)   



explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = explained_variance_ratio.cumsum()   
eigenvalues = pca.explained_variance_
plt.plot(eigenvalues, marker='o')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.show() 


#biplot
def biplot(score, coeff, labels=None):
    plt.figure(figsize=(10, 8))
    xs = score.iloc[:, 0]
    ys = score.iloc[:, 1]
    n = coeff.shape[0]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())
    plt.scatter(xs * scalex, ys * scaley, c='blue')

    for i in range(n):
        plt.arrow(0, 0, coeff[i, 0], coeff[i, 1], color='r', alpha=0.5)
        if labels is not None:
            plt.text(coeff[i, 0] * 1.15, coeff[i, 1] * 1.15, labels[i], color='g', ha='center', va='center')

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.grid()
    plt.tight_layout()
    plt.show()

biplot(df_pca.iloc[:, :2], np.transpose(pca.components_[:2, :]), labels=merged_df.columns)