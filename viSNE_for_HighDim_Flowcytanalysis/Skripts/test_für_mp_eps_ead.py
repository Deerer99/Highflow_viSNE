

import flowkit as fk


path = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MP_EPS_coated\MP-EPS_all.fcs"
fcs_data= fk.Sample(path,ignore_offset_error=True,cache_original_events=True)
fcs_data_df = fcs_data.as_dataframe(source="orig")
fcs_data_df= fcs_data_df.droplevel(0,axis=1)
fcs_data_df= fcs_data_df.iloc[:,0:12]
fcs_data_df.iloc[:,13]= "MP_EPS"

fcs_data_df.to_csv(r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MP_EPS_coated\MP_EPS.csv")