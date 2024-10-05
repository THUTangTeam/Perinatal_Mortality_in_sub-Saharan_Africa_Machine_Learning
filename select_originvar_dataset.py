import pandas as pd
import os
import pyreadstat 
import re
pd.set_option('display.max_columns', None)

def list_directories(path):
    try:
        # get items in path
        items = os.listdir(path)
        
        # all directories
        directories = [item for item in items if (os.path.isdir(os.path.join(path, item))) and ('DHS' in item)]
        
        return directories
    except Exception as e:
        print(f"An error occurred: {e}")
        return []
path_to_check = 'Machine_learningdataset/'
directories = list_directories(path_to_check)


#vars in each file
BR_list = ["v001","v002","v003","bidx","b18","b19", "b3", "b5", "b6", "b7", "v008","v018","v005", "bord", "b0", "b11"]
HR_list = ["hv001","hv002",]
IR_list = ["v001","v002","v003",  "v157", "v158", "v159", "v171a", "v171b", 
            "hivclust", "hivnumb", "v743a", 
            "v743b", "v743d", "v731", "v042", "v455", "v456", "v170", "v169a", "v169b", 
            "v155", "v149", "mv149", "mv005", "v044", "v201", "v213", "v228", 
            "d118y", "d005", "v467b", "v467c", "v467d", "v467f", "v511", 
            "awfactt", "v212", "v481a", "v481b", "v481c", "v481d", "v481e", 
            "v481f", "v481g", "v481h", "v481x","vcal",
            "v013","v176","v130","v131","v502","v024","v025","v106","v190"]
PR_list = ["hv001","hv002","hvidx","hv102","hv103","hv104","hv109",
           "hv115","hv012", "hv005", "hv204", "hv237", 
            "hv237a", "hv237b", "hv237c", "hv237d", "hv237e", "hv237f", 
            "hv237g", "hv237h", "hv237j", "hv237k", "hv237x", "hv201", 
            "hv201a", "hv202", "hv205", "hv225", 
            "hv238a", "hv230a", "hv230b", "hv232", "hv232b", 
             "hv270", "hv271",  "hv105", 
            "hv111", "hv112", "hv113", "hv114", "hv219",]
KR_list = ["v001","v002","v003","midx","m14", "m13","m2a", "m2n", "midx", "m15", "m3a", "m3b","m3c","m3d","m3f", "m3g", "m3h", "m3i","m3j",
            "m3k", "m3l","m3m", "m3n", "m77", "m17", "m17a", "m61", "m18", "m19", 
            "m1", "m1a", "m1d", "m19a", "m78a", "m78b", "m78c", "m78d", 
            "m78e", "m4", "m34", "m55", "m18", "m19"]

#merge different files
def merge_data(BR,IR,KR,HR,PR):

    # BR file
    selected_columns = []
    lowercase_columns = []
    not_found_columns = []

    for item in BR_list:
        item = item.upper()
        pattern = re.compile(rf'^{item}($|$)?')
        found = False
        for col in BR.columns:
            if pattern.match(col.upper()):
                found = True
                if col[0].islower():
                    if col not in lowercase_columns:
                        lowercase_columns.append(col)
                else:
                    if col not in selected_columns:
                        selected_columns.append(col)
        if not found:
            not_found_columns.append(item)

    BR_final_columns = selected_columns + lowercase_columns
    BR_1 = BR[BR_final_columns]
    print("Not found columns:", not_found_columns)

    #IR file
    selected_columns = []
    lowercase_columns = []
    not_found_columns = []

    for item in IR_list:
        item = item.upper()
        pattern = re.compile(rf'^{item}($|$)?')
        
        found = False
        for col in IR.columns:
            if pattern.match(col.upper()):
                found = True
                if col[0].islower():
                    if col not in lowercase_columns:
                        lowercase_columns.append(col)
                else:
                    if col not in selected_columns:
                        selected_columns.append(col)
        if not found:
            not_found_columns.append(item)

    IR_final_columns = selected_columns + lowercase_columns
    IR_1 = IR[IR_final_columns]
    
    print("Not found columns:", not_found_columns)

    #KR file
    selected_columns = []
    lowercase_columns = []
    not_found_columns = []

    for item in KR_list:
        item = item.upper()
        pattern = re.compile(rf'^{item}($|$)?')
        found = False
        for col in KR.columns:
            if pattern.match(col.upper()):
                found = True
                if col[0].islower():
                    if col not in lowercase_columns:
                        lowercase_columns.append(col)
                else:
                    if col not in selected_columns:
                        selected_columns.append(col)
        if not found:
            not_found_columns.append(item)

    KR_final_columns = selected_columns + lowercase_columns
    KR_1 = KR[KR_final_columns]
    KR_1.rename(columns = {"MIDX":"BIDX"},inplace = True)
    print("Not found columns:", not_found_columns)
    
    #HR file
    selected_columns = []
    lowercase_columns = []
    not_found_columns = []

    for item in HR_list:
        item = item.upper()
        pattern = re.compile(rf'^{item}($|$)?')
        found = False
        for col in HR.columns:
            if pattern.match(col.upper()):
                found = True
                if col[0].islower():
                    if col not in lowercase_columns:
                        lowercase_columns.append(col)
                else:
                    if col not in selected_columns:
                        selected_columns.append(col)
        if not found:
            not_found_columns.append(item)

    HR_final_columns = selected_columns + lowercase_columns
    HR_1 = HR[HR_final_columns]
    HR_1.rename(columns = {"HV001":"V001"},inplace = True)
    HR_1.rename(columns = {"HV002":"V002"},inplace = True)
    
    print("Not found columns:", not_found_columns)
    
    #PR file
    selected_columns = []
    lowercase_columns = []
    not_found_columns = []

    for item in PR_list:
        item = item.upper()
        pattern = re.compile(rf'^{item}($|$)?')
        found = False
        for col in PR.columns:
            if pattern.match(col.upper()):
                found = True
                if col[0].islower():
                    if col not in lowercase_columns:
                        lowercase_columns.append(col)
                else:
                    if col not in selected_columns:
                        selected_columns.append(col)
        if not found:
            not_found_columns.append(item)

    PR_final_columns = selected_columns + lowercase_columns
    PR_1 = PR[PR_final_columns]
    PR_1.rename(columns = {"HV001":"V001"},inplace = True)
    PR_1.rename(columns = {"HV002":"V002"},inplace = True)
    PR_1.rename(columns = {"HVIDX":"V003"},inplace = True)
    
    print("Not found columns:", not_found_columns)
    BKR = pd.merge(BR_1,KR_1,on = ['V001','V002','V003','BIDX'])
    IKR = pd.merge(IR_1,BKR,on = ['V001','V002','V003'])
    PHR = pd.merge(PR_1,HR_1,on = ['V001','V002'])
    HPIKR = pd.merge(IKR,PHR,on = ['V001','V002','V003'])
    HPIKR['Country_name'] = KR_countryname
    HPIKR.to_csv('Final_dataset/'+PR_countryname + '.csv',index = False)

# get file's path
BR_filelist = []
IR_filelist = []
KR_filelist = []
HR_filelist = []
PR_filelist = []

for directory in directories:
    path_item = path_to_check +'/' + directory
    #print(path_item)
    sec_items = os.listdir(path_item)
    #print(sec_items)
    for sec_item in sec_items:
        if (os.path.isdir(os.path.join(path_item, sec_item))) and ('BR' in sec_item):
            BR_dir = sec_item
            path_sav = path_item + '/' + BR_dir
            BR_sav_items = os.listdir(path_sav)
            for BR_sav_item in BR_sav_items:
                #print('---')
                if (os.path.isfile(os.path.join(path_sav, BR_sav_item))) and (BR_sav_item.endswith(".SAV")):
                    BR_sav = BR_sav_item
                    BR_sav_path = path_sav + '/'+BR_sav
                    BR_filelist.append(BR_sav_path)
                    #print('have read BR')
    for sec_item in sec_items:
        if (os.path.isdir(os.path.join(path_item, sec_item))) and ('IR' in sec_item):
            IR_dir = sec_item
            path_sav = path_item + '/' + IR_dir
            IR_sav_items = os.listdir(path_sav)
            for IR_sav_item in IR_sav_items:
                #print('---')
                if (os.path.isfile(os.path.join(path_sav, IR_sav_item))) and (IR_sav_item.endswith(".SAV")):
                    IR_sav = IR_sav_item
                    IR_sav_path = path_sav + '/'+IR_sav
                    IR_filelist.append(IR_sav_path)
                    #print('have read IR')
    for sec_item in sec_items:
        if (os.path.isdir(os.path.join(path_item, sec_item))) and ('KR' in sec_item):
            KR_dir = sec_item
            path_sav = path_item + '/' + KR_dir
            KR_sav_items = os.listdir(path_sav)
            for KR_sav_item in KR_sav_items:
                #print('---')
                if (os.path.isfile(os.path.join(path_sav, KR_sav_item))) and (KR_sav_item.endswith(".SAV")):
                    KR_sav = KR_sav_item
                    KR_sav_path = path_sav + '/'+KR_sav
                    KR_filelist.append(KR_sav_path)
                    #print('have read KR')
    for sec_item in sec_items:
        if (os.path.isdir(os.path.join(path_item, sec_item))) and ('HR' in sec_item):
            HR_dir = sec_item
            path_sav = path_item + '/' + HR_dir
            HR_sav_items = os.listdir(path_sav)
            for HR_sav_item in HR_sav_items:
                #print('---')
                if (os.path.isfile(os.path.join(path_sav, HR_sav_item))) and (HR_sav_item.endswith(".SAV")):
                    HR_sav = HR_sav_item
                    HR_sav_path = path_sav + '/'+HR_sav
                    HR_filelist.append(HR_sav_path)
                    #print('have read KR')
    for sec_item in sec_items:
        if (os.path.isdir(os.path.join(path_item, sec_item))) and ('PR' in sec_item):
            PR_dir = sec_item
            path_sav = path_item + '/' + PR_dir
            PR_sav_items = os.listdir(path_sav)
            for PR_sav_item in PR_sav_items:
                #print('---')
                if (os.path.isfile(os.path.join(path_sav, PR_sav_item))) and (PR_sav_item.endswith(".SAV")):
                    PR_sav = PR_sav_item
                    PR_sav_path = path_sav + '/'+PR_sav
                    PR_filelist.append(PR_sav_path)
                    #print('have read KR')
                    
#merge each country                   
for BR_item in BR_filelist:
    BR_countryname = BR_item[-12:-10]
    print('first',BR_countryname)
    BR,meta_BR = pyreadstat.read_sav(BR_item)
    for IR_item in IR_filelist:
        IR_countryname = IR_item[-12:-10]
        if IR_countryname == BR_countryname:
            print("Match1",IR_countryname)
            IR,meta_IR = pyreadstat.read_sav(IR_item)
        else:
            continue
        for KR_item in KR_filelist:
            KR_countryname = KR_item[-12:-10]
            if KR_countryname == IR_countryname: 
                print('Match2',KR_countryname)
                KR,meta_KR = pyreadstat.read_sav(KR_item)
            else:
                continue
            for HR_item in HR_filelist:
                HR_countryname = HR_item[-12:-10]
                if HR_countryname == KR_countryname: 
                    print('Match3',HR_countryname)
                    HR,meta_HR = pyreadstat.read_sav(HR_item)
                else:
                    continue
                for PR_item in PR_filelist:
                    PR_countryname = PR_item[-12:-10]
                    if PR_countryname == HR_countryname: 
                        print('Match4',PR_countryname)
                        PR,meta_PR = pyreadstat.read_sav(PR_item)
                        merge_data(BR,IR,KR,HR,PR)
                        print('success merge')
                    else:
                        continue
                