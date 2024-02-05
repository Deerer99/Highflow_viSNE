
import ML_tsne_lib as MLtsne
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import seaborn as sns
import os 
from matplotlib import pyplot as plt



dir_path_train_data =r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\2_ML_training\2_models\1_models\V09_viSNE_gate_model\gated_pop"
dir_path_test_data = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\2_ML_training\3_validation\3_eval_viSNE_Matlab\validation_1\Rawdata"
dir_save=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\2_ML_training\3_validation\3_eval_viSNE_Matlab\validation_1\excel_ML_result\VAL_09.xlsx"

dir_save_hist= None

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

fcs_data_for_ML_label = MLtsne.load_data_from_structured_directory(dir_path_train_data,data_from_matlab=True,accepted_col_names=accepted_col_names)

  
l=None
rf_class, conf_matrix, train_df_all, report = MLtsne.develop_ML_model_RF(
    l,
    test_size=0.5,
    random_state=70,
    frac=1,
    additional_df_non_transformed=fcs_data_for_ML_label
)

summary_df,pred_df_list= MLtsne.evaluate_dir_with_ML_classifier(
    dir_evaluate=dir_path_test_data,classifier=rf_class,dir_save=dir_save,
    subdir=False,triplicates=True,conf_interval=True,
    transform=True,accepted_col_names=accepted_col_names,
    data_from_matlab=False
)
# disp= ConfusionMatrixDisplay(confusion_matrix=conf_matrix,display_labels=rf_class.classes_)
# disp.plot(xticks_rotation="vertical",include_values=True)
# MLtsne.save_conf_matrix(conf_matrix=conf_matrix,rf_class=rf_class,dir_save=dir_save_conf_matrix)
#plot hisogramms
# pred_df_all = MLtsne.asinh_transform(pd.concat(pred_df_list),min_max_trans=False)
# pure =pred_df_all.drop("filename",axis=1)
# train_df_all = train_df_all.rename(columns={"filename":"Label"})
# MLtsne.create_hist_comparison_for_experiment(train_df=train_df_all,pred_df=pure,dir_save=dir_save_hist)

# # test

# names= ["UP","DW1","DW2"]
# MLtsne.create_tsne_for_species_percent(summary_df=summary_df,plot=True,names=names)

subplot_identifier= None
#fig_save_path = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\Pyrenäen\ML_results\Barplot"
MLtsne.stacked_barplot_for_certain_rows(df=summary_df,identifer=None,cols_to_plot=['Cyanobacteria',"Diatoms", 'Greenalgae', 'MP', 'Sediment'])


excel_path=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\Pyrenäen\Water_data_pyr\Besos catchment summer 2022.xlsx"
concat_df,loc= MLtsne.create_correl_heatmap_for_WWTP_dataset(summary_df=summary_df,excel_path=excel_path)


pca_df,pca=MLtsne.preform_pca_on_dataframe(concat_df)

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

biplot(pca_df.iloc[:, :2], np.transpose(pca.components_[:2, :]), labels=concat_df.columns)


#component loadings 
loadings = pca.components_
component_names = [f'PC{i+1}' for i in range(len(loadings))]
loading_df = pd.DataFrame(loadings, columns=concat_df.columns)
loading_df.index = component_names



#variance
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = explained_variance_ratio.cumsum()

# Plot explained variance
plt.plot(cumulative_variance, marker='o', linestyle='--', color='b', label='Cumulative Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')

# Annotate each point with markers
for i, value in enumerate(cumulative_variance):
    plt.scatter(i, value, color='b')
    plt.text(i, value, f'({i+1}, {value:.2f})', ha='right', va='bottom')

plt.legend()
plt.show()