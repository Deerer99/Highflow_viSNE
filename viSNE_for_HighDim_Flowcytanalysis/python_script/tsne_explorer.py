


import ML_tsne_lib as M
import pandas as pd
import numpy as np
import os 
from sklearn.manifold import TSNE

path= r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\Evaluation_viSNE_Matlab\Rawdata"
fcs_data = M.load_fcs_from_dir(path,label_data_frames=True,data_from_matlab=False)

fcs_data_concat=pd.concat(fcs_data)
subsampled_df = fcs_data_concat.sample(frac=0.2, random_state=42)
subsampled_df_trans= M.asinh_transform(subsampling_df=subsampled_df,min_max_trans=False)
tsne = TSNE(n_components=2, random_state=42).fit(subsampled_df_trans.iloc[:,:-1])
tsne_result= tsne.embedding_
M.create_visne_with_dropdown(tsne_result=tsne_result,subsampling_df=subsampled_df_trans)


for col in subsampled_df_trans.columns:
    ax= M.create_hist_for_channel_and_label(channel=[col],label=None,df=subsampled_df_trans)