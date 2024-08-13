


import pandas as pd
import ML_tsne_lib as M
from matplotlib import pyplot as plt
path=r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\MasterDataFolder\1_datasets\LightDarkPeri\ML_result\V07\Excel\LighDarkPeri_simpel.xlsx"

df = pd.read_excel(path,index_col=0)

df_clean = M.remove_cf_interval_from_df(df)

df_clean_ = df_clean.drop("event_count",inplace=True,axis=1)

ident= ["D","L","epi","bio"]

def get_df_for_ident(df,iden):

    df_i = df[df['filename'].str.contains('|'.join(iden))]

    return df_i

def create_ax_obj_for_ident(df: pd.DataFrame, iden) -> plt.axes:
    # Define custom colors for each column
    colors = ['blue', 'orange', 'green', 'red', 'purple']

    fig, ax = plt.subplots()

    for i, each_iden in enumerate(iden):
        df_i = get_df_for_ident(df, each_iden)
        colnames = ["Cyanobacteria", "Diatom", "Grünalge", "MP", "Sediment"]
        df_i = df_i.filter(items=colnames)
        mean_values = df_i.mean()
        std_values = df_i.std()

        bars_list = []
        bottom_value = 0
        for j, col in enumerate(colnames):
            bar = ax.bar(x=i, height=mean_values[col], yerr=std_values[col],
                          label=col, bottom=bottom_value, color=colors[j])
            bars_list.append(bar)
            bottom_value += mean_values[col]

    # Set labels and legend
    ax.set_xlabel('Identifiers')
    ax.set_ylabel('Percentages')
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[0:5], labels[0:5], title='Columns',bbox_to_anchor=(1, 1))
    ax.set_xticks(range(len(iden)))
    ax.set_xticklabels(["D","L","epi","bio"])
    return ax