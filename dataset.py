#!/usr/bin/python

# coding: utf-8

# In[1]:




# In[2]:

import preprocessing
from preprocessing import *


# In[88]:


#torch.tensor(np.array(a))


# In[89]:


data_directory="/data/down"
train_df = pd.read_csv("{}/Overall_Survival_1year.csv".format(data_directory))
#display(train_df)

random_state = 12


def ran_state(random_state):
    df_train, df_valid = sk_model_selection.train_test_split(
        train_df,
        test_size=0.2,
        random_state=random_state,
        stratify=train_df["OS 1 year"])
    # print(df_valid)
    dummies_train = pd.get_dummies(df_train["OS 1 year"])  # Classification
    products_train = dummies_train.columns
    y_train = dummies_train.values
    # print(y_train)
    dummies_valid = pd.get_dummies(df_valid["OS 1 year"])  # Classification
    products_valid = dummies_valid.columns
    y_valid = dummies_valid.values

    return df_train, df_valid, dummies_train, products_train, y_train, dummies_valid, products_valid, y_valid


df_train, df_valid, dummies_train, products_train, y_train, dummies_valid, products_valid, y_valid = ran_state(
    random_state)



# In[90]:


#df_train.head()#tail


# In[91]:


class Dataset(torch_data.Dataset):
    def __init__(self, paths, targets, split, norm_set_of_files, transforms=None):
        self.paths = paths
        self.targets = targets
        self.split = split
        self.norm_set_of_files = norm_set_of_files
        self.transforms=transforms
        
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, index):
        scan_id = self.paths[index]
        if self.targets is None:
            data = load_image_3d(scan_id, data_directory="/data/down", split=self.split, set_of_files=(["T13D_res111.nii.gz"]), norm_set_of_files=self.norm_set_of_files)
            data = np.array(data)
            #print(data.shape)
            if self.transforms:
                data = self.transforms(data)
            return {"X": torch.tensor(data).float(), "id": scan_id}

        else:
            data = load_image_3d(scan_id, data_directory="/data/down", split=self.split, set_of_files=(["T13D_res111.nii.gz"]), norm_set_of_files=self.norm_set_of_files)
            data = np.array(data)
            #print(data.shape)
            if self.transforms:
                data = self.transforms(data)
            y = torch.tensor(self.targets[index], dtype=torch.float32)
            return {"X": torch.tensor(data).float(), "y": y}
          

