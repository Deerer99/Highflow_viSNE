# %%
# import relevant libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import os
from sklearn.model_selection import train_test_split
rd.seed(100)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import ipywidgets as widgets
from matplotlib.cm import get_cmap
import flowkit as fk
from scipy.stats import t
import matplotlib.colors as mcolors
import hdbscan
from keras import Sequential
from keras.layers import Dense


# %%
# import data for viSNE
csv_path = r"C:\Users\bruno\OneDrive\Desktop\Programmer\viSNE_maps_and_data\Data\SingleSpeciesData"
definitions_dict = {
        "3B": 1,
        "3D": 2,
        "3G": 3,
        "A26": 4,
        "MP": 5,
        "CK":6,
        "CM":7,
        "FeOx":8,
        "Q12":9

    }

def load_data_from_structured_directory(rootdir):
    """
    Function to load data out of structured directories where the structure represents the measurements of each instance for which the label shoud be represented
    The label column is going to be determined from the name of the direcory in the root dir. Purpose is to label data according to dir names and prepare it for ML application

    Parameters:
    - rootdir: dir in which the structured data resides

    returns:
    - concatenated_list: each instance in each dir is contatenated and put into a overarching list which represents each directory
    
    
    
    
    """

    list_of_labeld_df = []
    # loop through the direcotry
    for subdir, dirs, f in os.walk(rootdir):
        for each_dir in dirs:
        # check the files inside if csv or fcs ending 
            files = os.listdir(os.path.join(subdir,each_dir))
                               
            if files[0].endswith(".csv"):
                labeld_df = label_data_according_to_definitions(os.path.join(subdir,each_dir))
                for i, df in enumerate(labeld_df):
                    df['filename'] = each_dir
                list_of_labeld_df.append(labeld_df)
            elif files[0].endswith(".fcs"):
                labeld_df = load_fcs_from_dir(os.path.join(subdir,each_dir))
                for i, df in enumerate(labeld_df):
                    df['filename'] = each_dir
                list_of_labeld_df.append(labeld_df)
        
    concatenated_list = []
    for sublist in list_of_labeld_df:
        concatenated_list.extend(sublist)
                

    return concatenated_list


def label_data_according_to_definitions(csv_path,definitions_dict= None):
    """
    Function creates matrix and labels them according to filename naming.
    If no definitions dict is given then the whole filename is used to name the events

    csv_path: contains the path of your csv files
    definitions_dict: contains the matching suffix or prefix (unique), which matches corresponding labels
    
    return: returns a labeld data set from all the csv files in the dictionary
    """
    labeled_dfs = []
    

    for filename in os.listdir(csv_path):
        file_path = os.path.join(csv_path, filename)
        if filename.endswith(".csv"):  # Check if the file is a CSV file
            df = pd.read_csv(file_path)
            if definitions_dict is not None:
                for key, label in definitions_dict.items():
                    if key in filename:
                        df['filename'] = label  # Add a new column 'Label' with the corresponding label
                        labeled_dfs.append(df)
                        break  # No need to check other keys once a match is found
            elif definitions_dict is None:
                df['filename']= filename
                labeled_dfs.append(df)

    return labeled_dfs

def remove_overflow(fcs_df):

    value_to_remove = 1048575.0
    mask = ~fcs_df.isin([value_to_remove]).any(axis=1)
    df= fcs_df[mask]
    return df 

# %%
def load_fcs_from_dir(directory_path,label_data_frames=False):
    """ 
    Loads in the events in the matrix from the given directory

    Parameters:
    - directory_path: path of the directory

    Return: 
    - returns list of dataframes of all Area columns in question, no transformation is applied
    """
    fcs_files = []
    # Iterate through files in the directory
    for file in os.listdir(directory_path):
        if file.endswith(".fcs"):
            file_path = os.path.join(directory_path, file)
            try:
                fcs_data = fk.Sample(file_path,cache_original_events=True)
                fcs_data_df = fcs_data.as_dataframe(source="orig")
                fcs_data_df = remove_overflow(fcs_data_df)
                fcs_data_df = fcs_data_df[[col for col in fcs_data_df.columns if col[0].endswith('-A')]]
                fcs_data_df.columns = fcs_data_df.columns.droplevel(0)
                if label_data_frames is True:
                    fcs_data_df["filename"] = file
                fcs_files.append(fcs_data_df)
            except Exception as e:
                print(f"An error occurred while processing {file}: {e}")
    return fcs_files



# %%
# write subsampling function

def subsample_from_list_of_df(list_of_dataframes, subsampling_number=300, random_seed=None):
    """
    Subsample rows from each DataFrame in a list and concatenate them together.

    Parameters:
    - list_of_dataframes: List of pandas DataFrames to subsample from.
    - subsampling_number: Number of rows to subsample from each DataFrame.
    - random_seed: Seed for random number generation (for reproducibility).

    Returns:
    - Concatenated DataFrame containing the subsampled rows.

    Example:
    #subsampling_df = subsample_from_list_of_df(labeld_dfs,subsampling_number=2000)
    #subsampling_df_fcs_transformed = subsample_from_list_of_df(fcs_files,subsampling_number=3000,random_seed=42)
    """
    # Set random seed for reproducibility
    if random_seed is not None:
        np.random.seed(random_seed)
    
    subsampled_dfs = []
    
    for df in list_of_dataframes:
        if subsampling_number >= len(df):
            subsampled_dfs.append(df)  
        else:
            subsampled_df = df.sample(n=subsampling_number, replace=False)
            subsampled_dfs.append(subsampled_df)
    
    concatenated_df = pd.concat(subsampled_dfs, ignore_index=True)
    
    return concatenated_df



# %%
def asinh_transform(subsampling_df, factor=150,min_max_trans = True):
    """
    Apllies the asinh transformation on each column in one dataframe, also apllies a min max normalization

    Parameters:
    - subsampling_df : Dataframe
    - factor : Scaling factor that apllies like x= asinh(x/factor)

    Return:
    - transformed dataframe
    """
    transformed_df = subsampling_df.copy()  # Create a copy to store the transformed data
    for col_name in subsampling_df.columns:
        if col_name != 'filename' and pd.api.types.is_numeric_dtype(subsampling_df[col_name]):
            col = subsampling_df[col_name]
            transformed_df[col_name] = np.arcsinh(col / factor)

            # Min-max normalization
            if min_max_trans is True:
                col_min = transformed_df[col_name].min()
                col_max = transformed_df[col_name].max()
                transformed_df[col_name] = (transformed_df[col_name] - col_min) / (col_max - col_min)

    return transformed_df



# %%
def get_distinct_colors(n):
    """
    Gets unique colors

    Parameters:
    - n : amount of colors to create

    Return
    - Different colors 
    """
    colors = list(mcolors.TABLEAU_COLORS)
    while len(colors) < n:
        colors += colors
    return colors[:n]


def plot_vsne_result_generic(tsne_result, label_col):
    """
    Creates a generic vsne map from a tsne implementation 

    Parameters:
    - tsne_result : embedded tsne_result from TSNE function 
    - label_col : column that shoud be used to label the events in the viSNE

    Returns:
    - ()
    
    """
    x = tsne_result[:, 0]
    y = tsne_result[:, 1]
    if label_col is not None:
        distinct_colors = get_distinct_colors(len(label_col))  # Adjust the number of colors as needed
    else:
        distinct_colors= get_cmap("tab10")
    # Create a scatter plot with automatic colors for each label
    plt.figure(figsize=(8, 6))

    # Get unique labels
    if label_col is not None:
        unique_labels = label_col.unique()
        # Scatter plot with automatic colors
        for i, label in enumerate(unique_labels):
            plt.scatter(x[label_col == label], y[label_col == label], label=label, color=distinct_colors[i], alpha=0.7, s=0.05)
    else:
            plt.scatter(x, y, alpha=0.7, s=0.05)

    # Customize plot labels and title
    plt.xlabel('Values')
    plt.ylabel('Labels')
    plt.title('Scatter Plot with Automatic Colors for Each Label')
    plt.tight_layout()

    # Add a legend outside the figure
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Show the plot
    plt.show()


# %%
# create interactive plot to change coloring scheme with a dropdown menu 
def create_visne_with_dropdown(tsne_result,subsampling_df):
        """
        creates viSNE Map with coloring according to different input columns which can be switched through a dropdown menu

        Parameters:
        - tsne_result: Coordidnates in the viSNE Map created from subsampling_df
        - subsampling_df: a df of subsampled accordingly from original dataframes
        
        Returns:
        - interactive plot object 
        
        """
        data = {
                "X1":tsne_result[:,0],
                "X2":tsne_result[:,1]

        }

        df = pd.DataFrame(data)

        color_matrix = subsampling_df

        def update_scatter_plot(selected_col_for_coloring):
                plt.figure(figsize=(8,6))
                scatter = plt.scatter(df["X1"], df["X2"], c = color_matrix[selected_col_for_coloring], cmap="rainbow",s=0.01)
                plt.colorbar(scatter,label="Strenght")
                plt.show()

        coloring_scheme = widgets.Dropdown(
                options = color_matrix.columns,
                description = "Select Col for coloring: ",
                continuous_update = False

        )

        widgets.interactive(update_scatter_plot,selected_col_for_coloring=coloring_scheme)
        interactive_plot = widgets.interactive(update_scatter_plot, selected_col_for_coloring=coloring_scheme)
        return interactive_plot
    
    # Display the interactive widget
# %%
# automatic clustering 

def cluster_tsne_map(tsne_result,m=30,s=5):
    """
    Functions to automatically cluster the viSNE map via HDBSCAN

    Parameters:
    - tnse_result : embedded TNSE coordinates 
    - m : value for the HDBSAN, minimal cluster size
    - s: value minimal cluster


    Returns:
    - ()
    
    
    """

    clusterer = hdbscan.HDBSCAN(min_cluster_size=m, min_samples=s)
    cluster_labels = clusterer.fit_predict(tsne_result)

    # Visualize the clusters
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels, cmap='viridis', marker='.', s= 0.5)
    plt.colorbar()
    plt.title('HDBSCAN Clustering of t-SNE Data')
    plt.show()

# %%
# implement clustering based on mchine learning$

# implement machine learning model, use Y to predict labels
def develop_ML_model_RF(labeld_dfs, random_state = 42, test_size= 0.2):

    """
    Creates a machine learning model with the random forest classifier and prints a report

    Parameters:
    - labeld_dfs : labeled dataframe conataining a label column
    - random state : seed to use 
    - test_size : amount of data to use as a test case 

    Returns:
    - random forest classifier
    
    
    
    """

    combined_df = pd.concat(labeld_dfs,ignore_index=True)
    combined_df_trans = asinh_transform(combined_df)
    combined_df_rand= combined_df_trans.sample(frac=1,random_state=random_state).reset_index(drop=True)
    
    train_df, test_df = train_test_split(combined_df_rand,test_size=test_size,random_state=42)
    X_train = train_df.drop("filename",axis=1)
    y_train = train_df["filename"]

    X_test = test_df.drop("filename", axis=1)
    y_test = test_df["filename"]


    rf_class= RandomForestClassifier(n_estimators=100,max_depth=50,random_state=42,verbose=True,n_jobs= 3)

    rf_class.fit(X_train,y_train)


    y_pred = rf_class.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)
    report = classification_report(y_test,y_pred)

    print(f"accuracy:{accuracy:.2f}")
    print("Classification Report:")
    print(report)
    return rf_class


# %%

def load_csv_with_ending(dir_data,ending="_e"):
    """
    Loads csv files with a certain ending 

    Parameters:
    - dir_data : where the files are located, directory paths
    - ending : the ending to look out for 

    Returns:
    - dataframe from the direcotry 
    
    Example:
    input_df = load_csv_with_ending(dir_data)
    subsampling_input_df= subsample_from_list_of_df(input_df,1000,42)
    input_df_transformed= asinh_transform(subsampling_input_df,factor=150)
    """
    input_df = []
    for filename in os.listdir(dir_data):
        file_path = os.path.join(dir_data, filename)
        if filename.endswith(f"{ending}.csv"):  # Check if the file is a CSV file
            df = pd.read_csv(file_path)
            
            # Add a new column 'filename' with the current filename
            df['Label'] = filename
            
            input_df.append(df)
    return input_df


# %%
def rename_columns(input_df, column_mapping):

    """
    Renames columns if feature names are not equivalent

    Parameters:
    - input_df : Dataframe to rename 
    - column_mapping: dict with old:new names 

    Returns:
    - renamed dataframes
    
    """
    # Create a copy of the DataFrame to avoid modifying the original
    renamed_df = input_df.copy()

    # Rename columns based on the provided mapping
    renamed_df.rename(columns=column_mapping, inplace=True)

    return renamed_df

# Define the mapping from current column names to expected feature names
column_mapping = {
    'FL1 510/20': 'FL1_Area',
    'FL10 525': 'FL10_area',
    'FL2 542/27': 'FL2_area',
    'FL3 575': 'FL3_area',
    'FL4 620/30': 'FL4_area',
    'FL5 695/30': 'FL5_area',
    'FL6 660': 'FL6_area',
    'FL7 725/20': 'FL7_area',
    'FL8 755': 'FL8_area',
    'FL9 450/50': 'FL9_area',
    "FS":"FS_area",
    "SS": "SS_area"

}

# # %%


# ML_labels = rf_class.predict(subsampling_df_fcs_transformed)

# subsampling_df_fcs_transformed["ML_Labels"]= ML_labels

# # %%
# # create viSNE map with dropdown for all cols, hdbscan analysis, ML labeling

# tsne_result_input = TSNE(n_components=2, perplexity=30, verbose=True,method="barnes_hut").fit_transform(subsampling_df_fcs_transformed.iloc[:,:-1])

# # %%
# plot_vsne_result_generic(tsne_result_input,label_col=subsampling_df_fcs_transformed["ML_Labels"])

# # %%
# i_plot = create_visne_with_dropdown(tsne_result_input,input_matrix)
# display(i_plot)

# # %%


# %%

def create_stacked_barplot_from_ML(df, triplicates=True):
   
    filenames = list(np.concatenate(list(df['filename'])))
    unique_labels = df['unique_labels'][1]
    label_count = list(df['label_count'])
    colors = plt.cm.Paired(range(len(unique_labels[0])))
    # x positions
    fig,ax = plt.subplots()
    percentage_label_count = [[count_i / sum(count) * 100 for count_i in count] for count in label_count]
    bottom = np.zeros(len(filenames))
    
    for i,label in enumerate(unique_labels):
    
        position_i_values = [inner_list[i] for inner_list in percentage_label_count]
        print(position_i_values)
        ax.bar(filenames, position_i_values, color=colors[i],bottom=bottom,label=label)
        bottom = [sum(x) for x in zip(bottom, position_i_values)]




    ax.set_xlabel('Filename')
    ax.set_ylabel('Percentage of Unique Label Occurrences')
    ax.set_title('Stacked Barplot of Label Occurrences')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))


    plt.xticks(rotation=45, ha='right')
        
    plt.show()




# %%

def generate_unique_ML_values_from_labels(normalized_matrix_with_labels, max_t=1):

    """
    Generates numerical values for labels in order to include them in numerical operations

    Parameters:
    - normalized_matrix_with_labels : normalized (to 1) matrix of feature values
    - max_t : maximum value to give for values

    Returns:
    - normalized_matrix_with_labels : same matrix with label cols and value col
    - label value dict : mapping for values and labels
    
    """
    unique_labels = pd.unique(normalized_matrix_with_labels["Label"])
    values_for_tsne = np.linspace(0, max_t, len(unique_labels))
    label_value_dict = {}

    for i, each_entry in normalized_matrix_with_labels.iterrows():
        label = each_entry["Label"]
        if label in unique_labels:
            normalized_matrix_with_labels.at[i, "tsne_labels"] = values_for_tsne[np.where(unique_labels == label)[0][0]]
            label_value_dict[label] = values_for_tsne[np.where(unique_labels == label)[0][0]]

    return normalized_matrix_with_labels, label_value_dict



# %%

def get_feature_importance(clf=None,feature_names=None):

    """
    Gets the feature importance for a trained classifier, feature names can be adjusted
    Also creates a barplot to evaluate 

    Parameters:
    - clf : ML classifier
    - feature names : names of the features

    Returns:
    - feature importances 
    
    
    
    """
    importances = clf.feature_importances_

    
    feature_importances = pd.DataFrame(importances, index=feature_names, columns=['importance'])

    
    feature_importances = feature_importances.sort_values(by='importance', ascending=False)

    
    print(feature_importances)

   
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importances.index, feature_importances['importance'])
    plt.xticks(rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.show()

    return feature_importances



def create_deepL_classifier(X_matrix, y_pred):

    """
    Test function for future application if ML Model does not suffice
    
    """

    model = Sequential()
    model.add(Dense(10,input_dim=12,activation="relu"))
    model.add(Dense(8,activation="relu"))
    model.add(Dense(1,activation="sigmoid"))
    model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])
    model.fit(X_matrix,y_pred,epochs=50,batch_size=32)

    loss,accuracy = model.evaluate(X_test,y_test)





    return model


def evaluate_dir_with_ML_classifier(dir_evaluate,classifier,dir_save):

    """
    Combination function to evaluate a set of data in a directory.

    Parameters:
    - 



    Returns:
    - Label_filename_df: Returns a df which includes the name of a file, the labels and the associated counts
    
    
    """
    #load fcs files
    fcs_data=load_fcs_from_dir(directory_path=dir_evaluate,label_data_frames=True)
    Label_filename_df = pd.DataFrame()
    frames = []
    for i in range(0,len(fcs_data),3):

        calculating_df = pd.DataFrame()
        for frame in fcs_data[i:i+3]:
            new_row = pd.DataFrame()
            transform_frame = asinh_transform(frame)
            Labels = classifier.predict(transform_frame.iloc[:,:-1])
            unique_labels, label_count = np.unique(Labels, return_counts=True)   
            for label, count in zip(unique_labels, label_count):
                
                new_row[label] = [count]

                
            calculating_df = calculating_df.append(new_row, ignore_index=True)
            calculating_df.fillna(value=0,axis=0)
            frame["Label"]=Labels 
            frames.append(frame)

        Mean = np.mean(calculating_df).T
        std = np.std(calculating_df,ddof=1) 
        confidence_interval = t.interval(0.95, 2, Mean, std / np.sqrt(2))
        cf_lower = confidence_interval[0]
        cf_upper = confidence_interval[1]
        new_row2= pd.DataFrame()
        filename = np.unique(frame["filename"])
        new_row2["filename"]=filename

        for each_label,each_mean,each_cf_upper,each_cf_lower in zip(unique_labels,Mean,cf_upper,cf_lower):
            new_row2[each_label]= [None]
            new_row2[each_label] = each_mean
            new_row2[each_label + "cf_0.95_lower"] = each_cf_lower
            new_row2[each_label + "cf_0.95_upper"] = each_cf_upper
        

    
        Label_filename_df = Label_filename_df.append(new_row2, ignore_index=True)

    Label_filename_df.fillna(value=0,axis=0)
        
    if dir_save is not None:
        Label_filename_df.to_excel(dir_save)

    return Label_filename_df, frames 


def ML_statistic(df):
     
    df=pd.concat(df)
    mean_value_list = []
    for label in np.unique(df["Label"]):
        # Create a DataFrame for each label
        label_df = df[df['Label'] == label]
        
        # Calculate mean for each column
        mean_values = label_df.mean(numeric_only=True)
        mean_value_list.append([mean_values,label])
    return mean_value_list



# %%
