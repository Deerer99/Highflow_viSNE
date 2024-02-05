


import pandas as pd
import ML_tsne_lib as M
from matplotlib import pyplot as plt
path=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\1_datasets\LightDarkPeri\ML_result\V07\Excel\LighDarkPeri_simpel.xlsx"

df = pd.read_excel(path)

df_clean = M.remove_cf_interval_from_df(df)

df_clean_ = df_clean.drop("event_count",inplace=True,axis=1)
identifier = ["D","L","epi","bio"]


df_D = df[df['filename'].str.contains('|'.join(identifier)) & df['filename'].str.contains("D")]
df_L = df[df['filename'].str.contains('|'.join(identifier)) & df['filename'].str.contains("L")]
df_epi = df[df['filename'].str.contains('|'.join(identifier)) & df['filename'].str.contains("epi")]
df_bio = df[df['filename'].str.contains('|'.join(identifier)) & df['filename'].str.contains("bio")]

df_list= [df_D,df_L,df_bio,df_epi]
def plot_stacked_bar_with_errorbars(df, identifier):
    df= df.filter(items=identifier)
    mean_values = df.mean()
    std_values = df.std()

    plt.bar(mean_values.index, mean_values, yerr=std_values, label=identifier)
     