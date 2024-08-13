
import ML_tsne_lib as ML
import numpy as np
import pandas as pd


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
path = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\1_datasets\Sgier\Sgier_field_saples"
dir_save = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\1_datasets\Sgier\Sgier_field_samples_excel"
fcs_data = ML.load_fcs_from_dir(path,accepted_col_names=None,data_from_matlab=False)

fcs_data = ML.export_loaded_fcs_data_col_A_as_filename_csv(fcs_data_list=fcs_data,dir_save=dir_save,triplicates=False)


