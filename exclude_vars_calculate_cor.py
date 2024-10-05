import pandas as pd
import numpy as np
import os

#get file path
def list_directories(path):
    try:
        items = os.listdir(path)
        
        directories = [item for item in items if (os.path.isfile(os.path.join(path, item)))]
        
        return directories
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

path_to_check = 'Final_dataset'
directories = list_directories(path_to_check)

#get countries' common varibles
path_to_check = 'Final_dataset'
data_files = os.listdir(path_to_check)
common_columns =set()
datas =[]
for data_file in data_files:
    data = pd.read_csv(os.path.join(path_to_check, data_file))
    datas.append(data)
    common_columns =common_columns.union(set(data.columns))
print(len(common_columns))

#Keep common vars
all_columns = set()
common_columns = None

for file in directories:
    df = pd.read_csv(path_to_check + '/'+  file, nrows = 0)
    columns = set(df.columns)
    all_columns.update(columns)
    
    if common_columns is None:
        common_columns = columns  
    else:
        common_columns.intersection_update(columns)  # get intersection
# the vars that be excluded
columns_to_exclude = {'V001', 'V002', 'V003', 'BIDX','HV005', 'V005', 'Country_name','AWFACTT',
                      'VCAL$2','B7','B18','B19',}
                      #'VCAL$1','B3','B6','V008','V018','V008A'}

filtered_common_columns = common_columns - columns_to_exclude

path_to_save = 'public_var_dataset'

with open(os.path.join(path_to_save, 'common_columns.txt'), 'w') as f:
    for column in filtered_common_columns:
        f.write(column + '\n')
# all var's number
print(f"All columns that appear at least once: {len(all_columns)}")

# common var's number
print(f"Common columns across all datasets (excluding specified columns): {len(filtered_common_columns)}")

#filiter each country
path_to_save = 'public_var_dataset'

for file in directories:
    df = pd.read_csv(os.path.join(path_to_check, file), usecols=filtered_common_columns)
    output_path = os.path.join(path_to_save, file)  # use the original file name
    df.to_csv(output_path, index=False)
    print(f"Saved processed file: {output_path}")
   
#concat all countries' dataset into one dataset 
path_to_save = 'public_var_dataset'
directories = list_directories(path_to_save)
dataframes = []
for file in directories:
    df = pd.read_csv(os.path.join(path_to_save, file))
    dataframes.append(df)
concat_dataset = pd.concat(dataframes,axis = 0,ignore_index = True)
print(directories)

#define function to drop the outliers
def custom_sum(row):
    return row.replace(8, pd.NA).sum()
def custom_sum1(row):
    return row.replace(2, 1).sum()
def custom_sum2(row):
    return row.replace(8, pd.NA)
def merge_assistance_during_delivery_provider(df):
    if df['M3A'] == 1:
        return 5
    elif df['M3B'] == 1:
        return 4
    elif df['M3C'] == 1:
        return 3
    elif df['M3G'] == 1:
        return 2
    elif df[['M3H', 'M3I', 'M3J', 'M3K', 'M3L', 'M3M']].max() == 1:
        return 1
    elif df['M3N'] == 1:
        return 0
    else:
        return None
    
#integrate detialed varibles in to common one
concat_dataset['newborn_postnatal_components'] = concat_dataset[['M78A', 'M78B', 'M78C', 'M78D', 'M78E']].apply(custom_sum, axis=1)
concat_dataset['difficult_in_get_medical_help'] = concat_dataset[['V467B', 'V467C', 'V467D', 'V467F']].apply(custom_sum1, axis=1)
concat_dataset['health_insurance_coverage'] = concat_dataset[['V481A','V481B','V481C','V481D','V481E','V481F','V481G','V481H','V481X']].max(axis=1)
concat_dataset['receive_components_ANC'] = concat_dataset[['M42A','M42B','M42C','M42D','M42E']].replace(8, pd.NA).sum(axis = 1)
concat_dataset['drugs_for_malaria_during_pregnancy'] = concat_dataset[['M49A', 'M49B', 'M49C', 'M49D', 'M49E', 'M49F', 'M49G', 'M49X', 'M49Y', 'M49Z']].replace(8, pd.NA).max(axis=1)
concat_dataset['assistance_during_delivery_provider'] = concat_dataset.apply(merge_assistance_during_delivery_provider, axis = 1)

#calculate perinatal death
def cal_death(data):
    """
    Calculate the Perinatal Mortality According to DHS7 Guide
    :param data: a dataframe of one country
    :return: a dataframe with a new column 'Perinatal_Death' indicating perinatal mortality (1 for perinatal death, 0 otherwise)
    """
    data["V018"].fillna(0, inplace=True)
    data["V018"] = data["V018"].replace([np.inf, -np.inf], 0)
    data["V018"] = data["V018"].astype(int)
    data["VCAL$1"] = data["VCAL$1"].astype(str)
    data["Stillbirth"] = data.apply(lambda row: 1 if "TPPPPPP" in row['VCAL$1'][row['V018']: row["V018"]+60] else 0, axis=1)

    data["B6"].fillna(0, inplace=True)
    data["Early_Neonatal_Death"] = 0
    data.loc[(data["V008"] >= data["B3"]) & (data["B3"] >= data["V008"] - 59) & (data["B6"] >= 100) & (data["B6"] <= 106), "Early_Neonatal_Death"] = 1
    data.loc[( data['Early_Neonatal_Death'] == 1) & ( data['Stillbirth'] == 1), 'Early_Neonatal_Death'] = 0
    data["Perinatal_Death"] = data["Stillbirth"] | data["Early_Neonatal_Death"]

    return data

#calculate stillbirth
def cal_stillbirth(data):
    data["V018"].fillna(0, inplace=True)
    data["V018"] = data["V018"].replace([np.inf, -np.inf], 0)
    data["V018"] = data["V018"].astype(int)
    data["VCAL$1"] = data["VCAL$1"].astype(str)
    data["Stillbirth"] = data.apply(lambda row: 1 if "TPPPPPP" in row['VCAL$1'][row['V018']: row["V018"]+60] else 0, axis=1)
    return data

#calculate neonatal_death
def cal_neonatal_death(data):
    data["B6"].fillna(0, inplace=True)
    data["Neonatal_Death"] = 0
    data.loc[(data["V008"] >= data["B3"]) & (data["B3"] >= data["V008"] - 59) & (((data["B6"] >= 100) & (data["B6"] <= 130)) | (data["B6"] == 201)), "Neonatal_Death"] = 1 
    return data

#calculate early neonatal_death
def cal_early_neonatal_death(data):
    data["B6"].fillna(0, inplace=True)
    data["Early_Neonatal_Death"] = 0
    data.loc[(data["V008"] >= data["B3"]) & (data["B3"] >= data["V008"] - 59) & (data["B6"] >= 100) & (data["B6"] <= 106), "Early_Neonatal_Death"] = 1   
    return data

concat_dataset_stillbirth = cal_stillbirth(concat_dataset)
concat_dataset_neonatal_death = cal_neonatal_death(concat_dataset)
concat_dataset_early = cal_early_neonatal_death(concat_dataset)
concat_dataset1 = cal_death(concat_dataset)


#exclude varible
#perinatal
exclude_var1 = ['HV237A','HV237B','HV237C','HV237D','HV237E','HV237F','HV237G','HV237H','HV237I','HV237J','HV237K','HV237X','HV237Z',
               'M2N','HV270A',#'M4','V169B','HV201A','V743B','V743D',
               'HV232', 'HV232B', 'HV232C', 'HV232D', 'HV232E',
               'M78A', 'M78B', 'M78C', 'M78D', 'M78E',
               'V467B', 'V467C', 'V467D', 'V467F',
               'V481A','V481B','V481C','V481D','V481E','V481F','V481G','V481H','V481X',
               'M42A','M42B','M42C','M42D','M42E',
               'M49A', 'M49B', 'M49C', 'M49D', 'M49E', 'M49F', 'M49G', 'M49X', 'M49Y', 'M49Z',
               'M3A','M3B', 'M3C','M3D','M3F','M3G','M3H','M3I','M3J','M3K','M3L','M3M','M3N',
               'VCAL$1','B3','B5','B6','V008','V018','V008A','Stillbirth','Early_Neonatal_Death','Neonatal_Death']
print(len(exclude_var1))
    
#stillbirth
exclude_var2 = ['HV237A','HV237B','HV237C','HV237D','HV237E','HV237F','HV237G','HV237H','HV237I','HV237J','HV237K','HV237X','HV237Z',
               'M2N','HV270A',
               'HV232', 'HV232B', 'HV232C', 'HV232D', 'HV232E',
               'M78A', 'M78B', 'M78C', 'M78D', 'M78E',
               'V467B', 'V467C', 'V467D', 'V467F',
               'V481A','V481B','V481C','V481D','V481E','V481F','V481G','V481H','V481X',
               'M42A','M42B','M42C','M42D','M42E',
               'M49A', 'M49B', 'M49C', 'M49D', 'M49E', 'M49F', 'M49G', 'M49X', 'M49Y', 'M49Z',
               'M3A','M3B', 'M3C','M3D','M3F','M3G','M3H','M3I','M3J','M3K','M3L','M3M','M3N',
               'VCAL$1','B3','B5','B6','V008','V018','V008A','Neonatal_Death','Early_Neonatal_Death','Perinatal_Death']
print(len(exclude_var2))

#early_neonatal_death
exclude_var3 = ['HV237A','HV237B','HV237C','HV237D','HV237E','HV237F','HV237G','HV237H','HV237I','HV237J','HV237K','HV237X','HV237Z',
               'M2N','HV270A',
               'HV232', 'HV232B', 'HV232C', 'HV232D', 'HV232E',
               'M78A', 'M78B', 'M78C', 'M78D', 'M78E',
               'V467B', 'V467C', 'V467D', 'V467F',
               'V481A','V481B','V481C','V481D','V481E','V481F','V481G','V481H','V481X',
               'M42A','M42B','M42C','M42D','M42E',
               'M49A', 'M49B', 'M49C', 'M49D', 'M49E', 'M49F', 'M49G', 'M49X', 'M49Y', 'M49Z',
               'M3A','M3B', 'M3C','M3D','M3F','M3G','M3H','M3I','M3J','M3K','M3L','M3M','M3N',
               'VCAL$1','B3','B5','B6','V008','V018','V008A','Neonatal_Death','Stillbirth','Perinatal_Death']
print(len(exclude_var3))

#neonatal_death
exclude_var4 = ['HV237A','HV237B','HV237C','HV237D','HV237E','HV237F','HV237G','HV237H','HV237I','HV237J','HV237K','HV237X','HV237Z',
               'M2N','HV270A',
               'HV232', 'HV232B', 'HV232C', 'HV232D', 'HV232E',
               'M78A', 'M78B', 'M78C', 'M78D', 'M78E',
               'V467B', 'V467C', 'V467D', 'V467F',
               'V481A','V481B','V481C','V481D','V481E','V481F','V481G','V481H','V481X',
               'M42A','M42B','M42C','M42D','M42E',
               'M49A', 'M49B', 'M49C', 'M49D', 'M49E', 'M49F', 'M49G', 'M49X', 'M49Y', 'M49Z',
               'M3A','M3B', 'M3C','M3D','M3F','M3G','M3H','M3I','M3J','M3K','M3L','M3M','M3N',
               'VCAL$1','B3','B5','B6','V008','V018','V008A','Early_Neonatal_Death','Stillbirth','Perinatal_Death']
print(len(exclude_var4))

#keep columns that perinatal death need 
final_concat = concat_dataset.drop(columns = exclude_var1)

#exclude missing rate over 70%
miss_counts = final_concat.isnull().sum()
threshold = 0.7 * len(final_concat)
columns_to_keep = final_concat.columns[miss_counts <= threshold]
columns_to_drop = final_concat.columns[miss_counts > threshold]
columns_to_keep_list = columns_to_keep.tolist()
print("Columns to keep:", len(columns_to_keep_list))
final_concat_keep = final_concat[columns_to_keep_list]
columns_to_drop_list = columns_to_drop.tolist()
print("Columns to drop:", columns_to_drop_list)
review_drop = ['M4','V169B','HV201A','V743B','V743D','Perinatal_Death']
final_dataset = final_concat_keep.drop(columns = review_drop) #75 columns
final_dataset = final_dataset.drop(final_dataset.index[54351:54512]) #exclude the row with all missing value

#读入变量名称变换字典
df_rename_dict = pd.read_excel('final_var.xlsx')
rename_dict = dict(zip(df_rename_dict['Varible'], df_rename_dict['Name']))
#calculate corelation
correlation_matrix = final_dataset.corr(method='pearson')
correlation_matrix.rename(columns=rename_dict, inplace=True)
correlation_matrix.to_csv('value_correlation_name_75.csv',index = False)

#select cor over 0.8
upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
high_correlation_pairs = [(index, column, upper_tri.loc[index, column])
                          for column in upper_tri.columns 
                          for index in upper_tri.index 
                          if upper_tri.loc[index, column] > 0.8]
high_corr_df = pd.DataFrame(high_correlation_pairs, columns=['Varible1', 'Varible2', 'Correlation coefficient'])
high_corr_df.to_excel('high_correlation_pairs.xlsx', index=False)

#calculate nullity
nullity_matrix = final_dataset.isna()
nullity_correlation_matrix = nullity_matrix.corr(method='pearson')
nullity_correlation_matrix.rename(columns=rename_dict, inplace=True)
nullity_correlation_matrix.to_csv('value_nullity_name_75.csv',index = False)

#calculate 68 vars
drop_columns = ['V106','HV109','V190A','HV270','HV271','HV271A','HV105']
final_data = final_dataset.drop(columns = drop_columns)
correlation_matrix1 = final_data.corr(method='pearson')
correlation_matrix1.rename(columns=rename_dict, inplace=True)
correlation_matrix1.to_csv('value_correlation_name_68.csv',index = False)

nullity_correlation_matrix = nullity_matrix.corr(method='pearson')
nullity_correlation_matrix.rename(columns=rename_dict, inplace=True)
nullity_correlation_matrix.to_csv('value_nullity_name_68.csv',index = False)

columns_delect = ['M4','V169B','HV201A','V743B','V743D','V106','HV109','V190A','HV270','HV271','HV271A','HV105']
final_train_dataset = final_concat_keep.drop(columns = columns_delect)
final_train_dataset = final_train_dataset.drop(final_train_dataset.index[54351:54512])
final_train_dataset.to_csv('Final_data_68.csv',index = False)

print("finish")