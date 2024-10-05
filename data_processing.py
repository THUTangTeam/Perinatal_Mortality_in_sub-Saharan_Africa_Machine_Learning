import os
import pandas as pd
import logging
from fancyimpute import IterativeImputer
from options.train_options import BaseOptions
from sklearn.utils import resample


def cal_death(data):
    """
    Calculate the Perinatal Mortality According to DHS7 Guide
    :param data: a dataframe of one country
    :return: perinatal mortality of the country
    """
    data["V018"] = data["V018"].astype(int)
    data["VCAL$1"] = data["VCAL$1"].astype(str)
    stillbirths = data.apply(lambda row: "TPPPPPP" in row['VCAL$1'][row['V018']: row["V018"] + 60], axis=1).sum()

    data["B6"].fillna(0, inplace=True)
    filtered_df = data[(data["V008"] >= data["B3"]) & (data["B3"] >= data["V008"] - 59)]
    live_births = filtered_df.shape[0]
    early_neonatal_deaths = filtered_df[(filtered_df["B6"] >= 100) & (filtered_df["B6"] <= 106)].shape[0]

    return (stillbirths + early_neonatal_deaths) / (stillbirths + live_births) * 1000


if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # add option
    opt = BaseOptions().parse()

    # make sure the right path
    opt.datapath = 'results/Final_data_68.csv'  # make sure using the right path
    data_file = os.path.join(opt.dataroot, opt.datapath)

    # check the existance of file
    if not os.path.isfile(data_file):
        raise FileNotFoundError(f"The data file {data_file} does not exist. Please check the path and try again.")

    # read csv
    df = pd.read_csv(data_file, low_memory=False)
    dfs = [df]
    continuous_vars = set()

    # read continous vars file
    continuous_vars_file = os.path.join(opt.dataroot, opt.saveroot, "final_data_continuous_vars.txt")
    if not os.path.isfile(continuous_vars_file):
        raise FileNotFoundError(f"The continuous variables file {continuous_vars_file} does not exist. Please check the path and try again.")

    with open(continuous_vars_file, "r") as f:
        for line in f:
            continuous_vars.add(line.strip())

    try:
        pm = cal_death(df)
        logging.info(f"Perinatal Mortality of {data_file}: {pm} with {df.shape[0]} data")
    except KeyError:
        logging.info(f"No Perinatal Mortality for {data_file} with {df.shape[0]} data")

    # get common columns
    common_columns = set(df.columns)


    remove_vars = set(["VCAL$1", "VCAL$2", "B6", "Country_name"])  # exclude common vars
    common_columns = list(common_columns - remove_vars)  # turn to list
    dfs_common = [df[common_columns] for df in dfs]

    # concat data
    final_df = pd.concat(dfs_common, ignore_index=True)
    remain_vars = []  # exclude over missing value vars
    for var in final_df.columns:
        if final_df[var].isnull().sum() / final_df.shape[0] < 0.7:
            remain_vars.append(var)
    logging.info(f"{len(remain_vars)} variables out of {len(common_columns)} left")
    final_df = final_df[remain_vars]

    # find public continuous vars
    common_continuous_vars = list(continuous_vars & set(remain_vars))
    logging.info(f"find {len(common_continuous_vars)} continuous variables: {common_continuous_vars}")
    with open(os.path.join(opt.dataroot, opt.saveroot, "continuous_vars_output.txt"), "w") as f:
        f.write(str(common_continuous_vars))

    # Iterative Imputer
    imputer = IterativeImputer(max_iter=5, random_state=opt.randomstate)
    imputed_df = pd.DataFrame(imputer.fit_transform(final_df), columns=final_df.columns)
    imputed_df.to_csv(os.path.join(opt.dataroot, opt.saveroot, "imputed_data.csv"), index=False)
    

