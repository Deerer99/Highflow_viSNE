


import ML_tsne_lib as MLtsne
import numpy as np
import pandas as pd
import os 


dir_path=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\NewProtocolSingleSpecies\iLink_species\Species"
dir_save=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\real_excel_species_data"

fcs_data_list= MLtsne.load_fcs_from_dir(directory_path=dir_path,label_data_frames=True)

MLtsne.export_loaded_fcs_data_col_A_as_filename_csv(fcs_data_list,dir_save)
