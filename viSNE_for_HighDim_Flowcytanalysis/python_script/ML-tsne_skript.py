

import ML_tsne_lib  as MLtsne
import pandas as pd
from sklearn.manifold import TSNE

dir_path = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\fcm_field_data\test_data_visne"
fcs_data=MLtsne.load_fcs_from_dir(directory_path=dir_path)



fcs_data = pd.concat(fcs_data)

transformed_fcs_data = MLtsne.asinh_transform(fcs_data,min_max_trans=True)

tnse_result = TSNE(perplexity=30,verbose=True).fit_transform(transformed_fcs_data)

MLtsne.create_visne_with_dropdown(tsne_result=tnse_result,subsampling_df=transformed_fcs_data)


