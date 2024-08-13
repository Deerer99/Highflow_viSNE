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
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_data_from_structured_directory(
        
    rootdir:str,
    data_from_matlab:bool=False,
    accepted_col_names:list=None

)   ->list:
    """
    Function to load data out of structured directories were the 
    structure represents the measurements of each instance for which 
    the label shoud be represented
    The label column is going to be determined from 
    the name of the direcory in the root dir. 
    Purpose is to label data according to dir names and 
    prepare it for ML application

    Parameters:
    - rootdir: dir in which the structured data resides
    - data_from_matlab: if the data was imported from matlab, wont get transformed again
    - accepted_col_names: list of accepted col names

    returns:
    - concatenated_list: each instance in each dir is contatenated and put into a 
                         overarching list which represents each directory
    
    
    
    
    """
    # this code needs to be remodeled sometime, not good 

    list_of_labeld_df = []
    # loop through the direcotry
    for each_dir in os.listdir(rootdir):
        if os.path.isdir(os.path.join(rootdir,each_dir)):
            files = os.listdir(os.path.join(rootdir,each_dir))
            if os.path.isdir(os.path.join(rootdir,each_dir,files[0])) is False:
                # check the files inside if csv or fcs ending 
                files = os.listdir(os.path.join(rootdir,each_dir))
                                
                if files[0].endswith(".csv"):
                    labeld_df = label_data_according_to_definitions(os.path.join(rootdir,each_dir))
                    for i, df in enumerate(labeld_df):
                        df['filename'] = each_dir
                        print(f"loaded {each_dir}")
                    list_of_labeld_df.append(labeld_df)
                elif files[0].endswith(".fcs"):
                    labeld_df = load_fcs_from_dir(os.path.join(rootdir,each_dir),data_from_matlab=data_from_matlab,accepted_col_names=accepted_col_names)
                    for i, df in enumerate(labeld_df):
                        df['filename'] = each_dir
                        print(f"loaded {each_dir}")
                    list_of_labeld_df.append(labeld_df)


            elif os.path.isdir(os.path.join(rootdir,each_dir,files[0])) is True:
                    for each_subdir in os.listdir(os.path.join(rootdir,each_dir)):
                        files = os.listdir(os.path.join(rootdir,each_dir,each_subdir))
                                
                        if files[0].endswith(".csv"):
                            labeld_df = label_data_according_to_definitions(os.path.join(rootdir,each_dir,each_subdir))
                            for i, df in enumerate(labeld_df):
                                df['filename'] = each_dir
                                print(f"loaded {each_dir}")
                            list_of_labeld_df.append(labeld_df)
                        elif files[0].endswith(".fcs"):
                            labeld_df = load_fcs_from_dir(os.path.join(rootdir,each_dir,each_subdir),data_from_matlab=data_from_matlab,accepted_col_names=accepted_col_names)
                            for i, df in enumerate(labeld_df):
                                df['filename'] = each_dir
                                print(f"loaded {each_dir}")
                            list_of_labeld_df.append(labeld_df)
        elif os.path.isdir(os.path.join(rootdir,each_dir,files[0])) is False:
                # check the files inside if csv or fcs ending 
                files = os.listdir(os.path.join(rootdir,each_dir))
                                
                if files[0].endswith(".csv"):
                    labeld_df = label_data_according_to_definitions(os.path.join(rootdir,each_dir))
                    for i, df in enumerate(labeld_df):
                        df['filename'] = each_dir
                        print(f"loaded {each_dir}")
                    list_of_labeld_df.append(labeld_df)
                elif files[0].endswith(".fcs"):
                    labeld_df = load_fcs_from_dir(os.path.join(rootdir,each_dir),data_from_matlab=data_from_matlab,accepted_col_names=accepted_col_names)
                    for i, df in enumerate(labeld_df):
                        df['filename'] = each_dir
                        print(f"loaded {each_dir}")
                    list_of_labeld_df.append(labeld_df)

    concatenated_list = []
    for sublist in list_of_labeld_df:
        concatenated_list.extend(sublist)
                

    return concatenated_list


def label_data_according_to_definitions(
        
        csv_path:str,
        definitions_dict:dict= None

)       ->pd.DataFrame:
    
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
        if filename.endswith(".csv"):
            df = pd.read_csv(file_path)
            if definitions_dict is not None:
                for key, label in definitions_dict.items():
                    if key in filename:
                        df['filename'] = label 
                        labeled_dfs.append(df)
                        break 
            elif definitions_dict is None:
                df['filename']= filename
                labeled_dfs.append(df)
            df = remove_overflow(df)

    return labeled_dfs

def remove_overflow(fcs_df:pd.DataFrame)->pd.DataFrame:

    """
    removes overflow (max) values in each column of a dataframe 
    this needs to be done to ensure no wrong values are assesed later
    
    Parameters:
    - fcs_df: flowcytometry dataframe 

    returns:
    - df: cleaned dataframe
    
    """
    value_to_remove = 1048575.0
    mask = ~fcs_df.isin([value_to_remove]).any(axis=1)
    df= fcs_df[mask]
    return df 

# %%
def load_fcs_from_dir(
        directory_path,
        label_data_frames=True,
        data_from_matlab = False,
        accepted_col_names=None
)       ->list:
    """ 
    Loads in the events in the matrix from the given directory, if the found file shoud be a directory the structured directory method is applied

    Parameters:
    - directory_path: path of the directory
    - label_data_frames: if the dataframes shoud be labeld with the filename 
    - data from matlab: does not drop level, important if original files or already processed files are used

    Return: 
    - returns list of dataframes of all Area columns in question, no transformation is applied
    """
    if accepted_col_names is None:
        print("No accepted colnames given, is mandatory to prevent further errors. Just give colnames as argument.")

    fcs_files = []
    # Iterate through files in the directory
    for file in os.listdir(directory_path):
        file_path = os.path.join(directory_path, file)
        if file.endswith(".fcs"):
            try:
                fcs_data = fk.Sample(file_path,cache_original_events=True,ignore_offset_error=True)
                fcs_data_df = fcs_data.as_dataframe(source="orig")
                fcs_data_df = remove_overflow(fcs_data_df)
                if data_from_matlab is False:
                    fcs_data_df = fcs_data_df[[col for col in fcs_data_df.columns if col[0].endswith('-A')]]
                fcs_data_df.columns = fcs_data_df.columns.droplevel(0)
                
                if accepted_col_names is not None:
                    for name in fcs_data_df.columns:
                        if name not in accepted_col_names:
                            try:
                                fcs_data_df=fcs_data_df.drop(name,axis=1)
                            except:
                                continue


                if label_data_frames is True:
                    fcs_data_df.loc[:,"filename"] = file
                fcs_files.append(fcs_data_df)
            except Exception as e:
                print(f"An error occurred while processing {file}: {e}")

        elif os.path.isdir(file_path) is True:
            print("Structured directory detected, check if loading was correct")
            fcs_data = load_data_from_structured_directory(directory_path,data_from_matlab=data_from_matlab,accepted_col_names=accepted_col_names)
            fcs_files.extend(fcs_data)

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
def asinh_transform(subsampling_df, factor=150,min_max_trans = True, asinh_transform_arg=True):
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
            if asinh_transform_arg is True:
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
                scatter = plt.scatter(df["X1"], df["X2"], c = color_matrix[selected_col_for_coloring], cmap="rainbow",s=0.1)
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

def cluster_tsne_map(tsne_result,m=100,s=10):
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
def develop_ML_model_RF(labeld_dfs, random_state = 42, test_size= 0.2,additional_df_non_transformed=None,asinh_transformation=True,frac=1):

    """
    Creates a machine learning model with the random forest classifier and prints a report

    Parameters:
    - labeld_dfs : labeled dataframe conataining a label column
    - random state : seed to use 
    - test_size : amount of data to use as a test case 

    Returns:
    - random forest classifier
    
    
    
    """
    if labeld_dfs is not None:
        combined_df = pd.concat(labeld_dfs,ignore_index=True)
        
        if asinh_transformation is True:
            combined_df_trans = asinh_transform(combined_df,min_max_trans=False)
        else:
            combined_df_trans=combined_df
    else:
        combined_df_trans=pd.DataFrame()

    if additional_df_non_transformed is not None:
        if type(additional_df_non_transformed) is list:
            additional_df_non_transformed=pd.concat(additional_df_non_transformed,ignore_index=True)
            combined_df_trans=pd.concat([combined_df_trans,additional_df_non_transformed],ignore_index=True)
        else:
            print(f"Dataframe must be inputted as list but was given as {type(additional_df_non_transformed)}")

    combined_df_rand= combined_df_trans.sample(frac=frac,random_state=random_state).reset_index(drop=True)
    
    train_df, test_df = train_test_split(combined_df_rand,test_size=test_size,random_state=random_state)
    X_train = train_df.drop("filename",axis=1)
    y_train = train_df["filename"]

    X_test = test_df.drop("filename", axis=1)
    y_test = test_df["filename"]


    rf_class= RandomForestClassifier(n_estimators=250,max_depth=50,verbose=True,random_state=random_state,n_jobs=6)

    rf_class.fit(X_train,y_train)


    y_pred = rf_class.predict(X_test)

    accuracy = accuracy_score(y_test,y_pred)
    report = classification_report(y_test,y_pred)
    conf_matrix = confusion_matrix(y_pred=y_pred,y_true=y_test,labels=rf_class.classes_)
    

    print(f"accuracy:{accuracy:.2f}")
    print("Classification Report:")
    print(report)
    return rf_class, conf_matrix, combined_df_rand, report


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

def create_stacked_barplot_from_ML(input_df, cols_to_plot=[]):
    """
    df: put in the contatenated list of all events with filename and Labels
    
    
    """
    df= input_df.copy()
    if df is list:
        print("Input contatenated dataframe, not list")
        return 
    if "filename" not in df.columns:
        print("Either filename or Label not in columns. Check if cols got assigned correctly or right dataframe got used")
        print(f"cols present:{df.columns}")
        return
    if any("cf_0.95" in col for col in df.columns):
        df = remove_cf_interval_from_df(df)
        print("confidence interval removed")

    df.drop("event_count", inplace=True,axis=1)
    filename_vec= [name.split("-")[0] for name in df["filename"]]
    print(df.columns)
    df = df[cols_to_plot]
    ax = df.plot(kind="bar",stacked=True)



    ax.set_xticks(range(len(filename_vec)))
    ax.set_xticklabels(filename_vec)
    ax.set_ylabel('Percent of classified events')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4),ncol=len(df.columns)//2)
    
    plt.xticks(rotation=45, ha='right')


def stacked_barplot_for_certain_rows(df,identifer=[],cols_to_plot=[],fig_save=None):
    if identifer is not None:
        for each_identifier in identifer:
            subset_df = df[df["filename"].str.contains(each_identifier)]
            create_stacked_barplot_from_ML(input_df=subset_df,cols_to_plot=cols_to_plot)
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.4, right=0.8)
            if fig_save is not None:
                plt.savefig(fig_save)
            plt.show()
            plt.close()
    else:
        create_stacked_barplot_from_ML(input_df=df,cols_to_plot=cols_to_plot)
        plt.tight_layout()
        plt.xticks(fontsize=8)
        plt.subplots_adjust(bottom=0.4, right=0.8)
        if fig_save is not None:
            plt.savefig(fig_save, bbox_inches='tight', pad_inches=0.1)
        plt.show()
        plt.close()

    return


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


def evaluate_dir_with_ML_classifier(
        dir_evaluate:str,
        classifier,
        dir_save:str,
        subdir:bool=False,
        triplicates:bool=False,
        conf_interval:bool=True,
        transform:bool = True,
        data_from_matlab:bool=False,
        accepted_col_names=None):

    """
    Combination function to evaluate a set of data in a directory.

    Parameters:
    - dir evaluate : directory which is evaluated,
    


    Returns:
    - Label_filename_df: Returns a df which includes the name of a file, the labels and the associated counts
    
    
    """

    #load fcs files


    if subdir is True:
        dirs = os.listdir(dir_evaluate)
        list_dirs = []
        for each_dir in dirs:
            each_dir_path = os.path.join(dir_evaluate,each_dir)
            fcs_data=load_fcs_from_dir(each_dir_path,label_data_frames=True,data_from_matlab=data_from_matlab)
            list_dirs.extend(fcs_data)
        
        fcs_data=list_dirs
    elif subdir is False:
        fcs_data=load_fcs_from_dir(directory_path=dir_evaluate,label_data_frames=True,data_from_matlab=data_from_matlab,accepted_col_names=accepted_col_names)

        
    Label_filename_df = pd.DataFrame()
    frames = []
    k=1
    if triplicates is True:
        k=3

    for i in range(0,len(fcs_data),k):

        calculating_df = pd.DataFrame()
        num_events=0
        for frame in fcs_data[i:i+k]:
            new_row = pd.DataFrame()
            if transform is True:
                transform_frame = asinh_transform(frame,min_max_trans=False)
            else:
                transform_frame= frame
            Labels = classifier.predict(transform_frame.iloc[:,:-1])
            unique_labels, label_count = np.unique(Labels, return_counts=True)   
            for label, count in zip(unique_labels, label_count):
                
                new_row[label] = [count]

                
            calculating_df = calculating_df.append(new_row, ignore_index=True)
            calculating_df.fillna(value=0,axis=0)
            frame["Label"]=Labels 
            frames.append(frame)
            print(frame["filename"])


            num_events += len(frame)

        
        Mean = np.mean(calculating_df).T
        std = np.std(calculating_df,ddof=1) 
        confidence_interval = t.interval(0.95, 2, Mean, std / np.sqrt(2))
        cf_lower = confidence_interval[0]
        cf_upper = confidence_interval[1]
        new_row2= pd.DataFrame()
        filename = np.unique(frame["filename"])
        new_row2["filename"]=filename
        new_row2["event_count"] = num_events

        for each_label,each_mean,each_cf_upper,each_cf_lower in zip(calculating_df.columns,Mean,cf_upper,cf_lower):
            new_row2[each_label]= [None]
            
            new_row2[each_label] = ((each_mean*k)/num_events)*100
            if conf_interval is True:
                new_row2[each_label + "cf_0.95_lower"] = ((3*each_cf_lower)/num_events)*100
                new_row2[each_label + "cf_0.95_upper"] = (3*(each_cf_upper)/num_events)*100
            

    
        Label_filename_df = Label_filename_df.append(new_row2, ignore_index=True)

        Label_filename_df.fillna(value=0,axis=0)
        
    if dir_save is not None:
        Label_filename_df.to_excel(dir_save)

    Label_filename_df.dropna(inplace=True,axis=1,how="all")
    Label_filename_df.fillna(0,inplace=True)
    return Label_filename_df, frames 


def ML_statistic(df,dir_save=None):
     
    df=pd.concat(df)
    mean_value_list = []
    label_df_list =[]
    for label in np.unique(df["Label"]):
        
        label_df = df[df['Label'] == label]
        # add functionality for exporting event data (raw)
        
        mean_values = label_df.mean(numeric_only=True)
        mean_value_list.append([mean_values,label])
        label_df_list.append([label_df_list,label])

    if dir_save is not None:
        label_df_list.to_excel(dir_save)
        mean_value_list.to_excel(dir_save)
        
    return mean_value_list,label_df_list


def create_hist_for_channel_and_label(channel=None,label=None,df=None):

    """
    Creates a histogramm for the passed dataframe label and channel
    

    """
    # if channel or label or df is None:
    #     print("One of channel, label or df was not defined")
    #     return
    # if type(df) is not pd.DataFrame:
    #     print(f"df is no Dataframe but {type(df)}")
    #     return
    if len(df) == 0:
        return

    try:
        if label is not None:
            df_label= df[df['Label'] == label]
        else:
            df_label=df
        channel_label_df = df_label[channel]
        if len(channel_label_df)==0:
            return
        ax= channel_label_df.plot(kind="density")
        ax.set_xlabel(channel)
        
    except:
        print("Error in create_hist_for_channel_and_label")
        return None

    return ax

def compare_train_hist_to_pred_hist(train_df,pred_df,channel=None,label=None):

    """
    Takes as input the whole training dataframe and the whole prediction dataframe
    and generates a comparison histogramm for the channel and label specified

    
    
    """
    try:
        ax_train = create_hist_for_channel_and_label(df=train_df,channel=channel,label=label)
        ax_test = create_hist_for_channel_and_label(df=pred_df,channel=channel,label=label)
        ax_test.set_title(label)
        fig = plt.gcf()
    except:
        print("Error in compare_train_hist_to_pred_hist")
        return None
    return  fig


def create_hist_comparison_for_experiment(train_df, pred_df, dir_save):

    channels = train_df.columns[0:12]
    print(channels)
    labels = np.unique(train_df["Label"])


    
    for label in labels:
        path = os.path.join(dir_save,label)
        os.mkdir(path)
        print(label)
        for channel in channels:
            fig = compare_train_hist_to_pred_hist(train_df=train_df,pred_df=pred_df,label=label,channel=channel)
            if fig is None:
                plt.close()
                continue
            plt.legend(labels=["train","test"])
            channel_new = channel.replace("/","_",)
            fname= os.path.join(path,f"{channel_new,label}.png")
            print(fname)
            fig.savefig(fname=fname)
            plt.close()
        

def add_location(df):

    """
    Costum function to add meters upstream or down stream of the Wastewater treatment plant in order to have numerical data
    
    """
    
    names = ["DW1","DW2","UP"]
    if "location" not in df.columns:
        df.insert(1,"location",0)


    for index,row in df.iterrows():
        if names[0] in row["filename"]:
            df["location"][index]= 2
        elif names[1] in row["filename"]:
            df["location"][index]=3
        elif names[2] in row["filename"]:
            df["location"][index] = 1


    return df


def analyze_corr_for_condition(df,):
    """
    Function analyzes the linear correlation
    """



# %%
def export_loaded_fcs_data_col_A_as_filename_csv(fcs_data_list,dir_save,triplicates=True):

    concat_list = []
    if triplicates is True:
        for i in range(0,len(fcs_data_list),3):
            new = pd.concat(fcs_data_list[i:i+3])
            print(len(new))
            concat_list.append(new)
    else:
        concat_list=fcs_data_list
    
    for dataset in concat_list:
        dataset=remove_overflow(dataset)
        dataset.reset_index(inplace=True,drop=True)
        name =dataset["filename"][0]
        dataset=dataset.drop("filename",axis=1)
        dataset.to_csv(os.path.join(dir_save,f"{name}_A_export.csv"),index=False)
    
def single_fcs_file_to_csv(dir_fcs, dir_save,transform = True):
    
    fcs_file = load_fcs_from_dir(dir_fcs,label_data_frames=True)
    dataset=remove_overflow(fcs_file[0])
    name = np.unique(dataset["filename"])
    dataset=dataset.drop("filename",axis=1)

    if transform is True:
        dataset = asinh_transform(dataset,min_max_trans=False)
    dataset.to_csv(os.path.join(dir_save,f"{name}_A_export_trans_woindex={transform}.csv"),index=False)

    

    pass


def import_data_from_Matlab(path_to_dir):

    """ data shoud be seperated for each population and be in one dataframe, the filename get added to the end column, returns dataframe"""
    fcs_data_df = []
    for file in os.listdir(path_to_dir):
        try:
            file_path = os.path.join(path_to_dir, file)
            fcs_data = fk.Sample(file_path,cache_original_events=True,ignore_offset_discrepancy=True,ignore_offset_error=True)
            fcs_data_df = fcs_data.as_dataframe(source="orig")
        except:

            print(f"Error in file: {file}")
            fcs_data_df = None
            pass
    return fcs_data_df
        
    
def create_tsne_for_species_percent(summary_df,plot=True,names=[],label_plot=True):


    """ 
    Generates tsne map in order to get overview of the data
    This works only if names = ["DW1","DW2","UP"] in column "filename"
    Otherwise the code has to get changed
    Pretty undynamic coding in here so only use with caution
    
    """
    if "location_label" not in summary_df.columns:
        summary_df.insert(1,"location_label",0)


    for index,row in summary_df.iterrows():
        for i,name in enumerate(names):
            if name in row["filename"]:
                summary_df["location_label"][index]= name
 
    
    summ_df_wo_confinterval= remove_cf_interval_from_df(summary_df)
    text_labels=[s.split("_")[0] for s in summ_df_wo_confinterval["filename"]]

    clean_df = summ_df_wo_confinterval.loc[:, ~summary_df.columns.isin(["filename", "event_count","location_label"])]
    tsne_cords =TSNE(n_components=2, learning_rate='auto', perplexity=4).fit_transform(X=clean_df)
    location_label = summary_df["location_label"]
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i/len(names)) for i in range(len(names))]
    c=[]
    for i,name in enumerate(names):
    
        for each_label in location_label:
            if each_label is name:
                c.append(colors[i])

    if plot is True:    
        plt.scatter(tsne_cords[:,0],tsne_cords[:,1],label=names,c=c)
    
    if label_plot is True:
        for i, label in enumerate(text_labels):
            plt.annotate(label, (tsne_cords[i,0], tsne_cords[i,1]), textcoords="offset points", xytext=(0, 5), ha='center')

    legend_labels = {n: color for n,color in zip(names,colors)}
    handels = [plt.Line2D([0],[0],marker="o",color="w",markerfacecolor=color,markersize=10) for color in colors]
    plt.legend(handels,legend_labels.keys(),title="Location", bbox_to_anchor=(1.05,1),loc="upper left")

    plt.tight_layout()

    plt.show()  
    return 


def add_good_location_identifier(summary_df):
    """
    Non dynamic eg. specific identity creation in order to sort values for 
    further processing
    
    
    
    """
    filename = summary_df["filename"]
    loc=[] 
    for each_filename in filename:
        each_filename_rep=each_filename.replace("U","D")
        s= each_filename_rep.split("D")
        loc.append(s[0])
    
    new_loc_list =[]
    if "location" in summary_df.columns:
        for i,entry in enumerate(summary_df["location"]):
            new_loc = loc[i]+"."+str(entry)
            new_loc = float(new_loc)
            new_loc_list.append(new_loc)

    return new_loc_list


def create_meter_from_location(loc_vector=None):
    """
    Adds predetimined meter from WWTP to specific location indices
    """
    
    loc_meter_vec =[]
    for back in loc_vector:
        if back == 1:
            loc_meter_vec.append(-50)
        elif back ==2:
            loc_meter_vec.append(50)
        elif back ==3:
            loc_meter_vec.append(400)


    return loc_meter_vec


def transform_light_dark_to_numerical(filename_vec,match_dic):

    """
    Creates and matches filenames to specific values defined in the match_dic and gives back a 
    vector numerical which represents categorical values 
    
    """
    categorical_vec = []
    for element in filename_vec:
        list_of_strings = element.split("-")
        concrete_name = list_of_strings[0]
        for key, value in match_dic.items():
            if key in concrete_name:
                categorical_vec.append(value)

    return categorical_vec


def save_conf_matrix(conf_matrix,rf_class,dir_save):
    """
    Saves the confusion matrix of a classifier in a given directory

    conf_matrix: confusion matrix
    rf_class: ML classifier with the .classes_ attribute to name cols and rows
    dir_save: path to a directory which is empty
    
    """
    conf_matrix = pd.DataFrame(conf_matrix)

    conf_matrix.columns=rf_class.classes_
    conf_matrix.set_index(rf_class.classes_,inplace=True)
    conf_matrix.to_excel(dir_save)
    return



def create_correl_heatmap_for_WWTP_dataset(
        
        summary_df:pd.DataFrame,
        excel_path:str=None

)       ->tuple[pd.DataFrame,pd.Series]:
    

    summary_df_loc= add_location(summary_df)
    loc = add_good_location_identifier(summary_df=summary_df)
    summary_df["location_total"]= loc
    loc_meter = create_meter_from_location(loc_vector=summary_df_loc["location"])
    summary_df["location from WWTP"] = loc_meter
    summary_df.reset_index(inplace=True, drop=True)
    clean_df = summary_df.loc[:,summary_df.columns.isin(["location_total","location from WWTP","Cyanobacteria","Diatom","MP","Sediment","Grünalge"])]
    clean_df_sorted = clean_df.sort_values("location_total")
    clean_df_sorted.reset_index(inplace=True,drop=True)


    water_df = pd.read_excel(excel_path,index_col=0,header=None)
    water_df.columns=water_df.iloc[1,:]
    water_df.reset_index(inplace=True,drop=True)
    water_df= water_df.iloc[2:,:]

    water_df_clean = water_df.loc[:, water_df.columns.isin(["DO","Temp","pH","Discharge","EC","O2sat","DONestimated","CH4","CO2","N2O"])]
    water_df_clean.reset_index(drop=True,inplace=True)
    water_df_clean = water_df_clean.iloc[0:40]
    water_df_clean.drop(index=28,inplace=True)
    water_df_clean.reset_index(inplace=True,drop=True)
    water_df_clean= water_df_clean.astype(float)


    location = clean_df_sorted["location_total"]
    clean_df_sorted.drop("location_total",inplace=True,axis=1)
    concat_df = pd.concat([clean_df_sorted,water_df_clean],axis=1)


    corr_matrix =concat_df.corr()
    corr_matrix_rounded = corr_matrix.round(2)
    sns.heatmap(corr_matrix_rounded, cmap="Blues", annot=True,annot_kws={"size": 8})

    return concat_df,location

def remove_cf_interval_from_df(df):

    df.drop(columns=[col for col in df.columns if "cf_0.95" in col],inplace=True)
    return df


def preform_pca_on_dataframe(
        
        dataframe: pd.DataFrame

)       ->pd.DataFrame:

    scaler = StandardScaler()
    X_scaled=scaler.fit_transform(dataframe)
    pca= PCA(n_components=5)
    X_pca= pca.fit_transform(X_scaled)
    df_pca=pd.DataFrame(X_pca)
    return df_pca,pca


