
import ML_tsne_lib as ML
import numpy as np
import pandas as pd



path = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\NewProtocolSingleSpecies\iLink_species\Species"
dir_save = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\excel_species_data_all"
fcs_data = ML.load_data_from_structured_directory(path)
fcs_data = ML.export_loaded_fcs_data_col_A_as_filename_csv(fcs_data_list=fcs_data,dir_save=dir_save)