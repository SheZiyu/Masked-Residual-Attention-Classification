#!/usr/bin/python
# coding: utf-8

# In[1]:

import dependencies
from dependencies import *


# In[5]:


# Transform .nrrd to .nii 
# Load images

def _load_raw_image(in_files, out_files):
    img = sitk.ReadImage(in_files)
    sitk.WriteImage(img, out_files)
    img = sitk.ReadImage(out_files)
    return img

#_load_raw_image("D:\\down\\Pcnls_1\\Pcnls_baseline\\Flair_res111.nrrd",
#               "D:\\down\\Pcnls_1\\Pcnls_baseline\\Flair_res111.nii.gz")
#set_of_files1=(os.path.join("D:\\down\\","Pcnls_1\\Pcnls_baseline\\",item) for item in ("Flair_res111.nrrd", "T1_res111.nrrd", "T13D_res111.nrrd", "T2_res111.nrrd"))
#set_of_files2=(os.path.join("D:\\down\\","Pcnls_1\\Pcnls_baseline\\",item) for item in ("Flair_res111.nii.gz", "T1_res111.nii.gz", "T13D_res111.nii.gz", "T2_res111.nii.gz"))
#for i, j in zip(set_of_files1, set_of_files2):
#    print(i, j)
#    _load_raw_image(i, j)


# In[6]:


# Transform .nrrd to .nii 
# Load images
#2 CHANNEL
def load_raw_image(scan_id, in_files, out_files):
    for image_file1, image_file2 in zip(in_files, out_files):
        in_file = "/data/down/{}/Pcnls_baseline/{}".format(scan_id, image_file1)
        img = sitk.ReadImage(in_file)
        out_file = "/data/down/{}/Pcnls_baseline/{}".format(scan_id, image_file2)
        sitk.WriteImage(img, out_file)
        #img = sitk.ReadImage(out_file)



# In[8]:


#inputs = {
       # "image": "D:\\down\\Pcnls_1\\Pcnls_baseline\\Flair_res111.nii.gz",
       # "label": "/content/maskT1.nrrd"
#    }
#T1 = LoadImaged(keys=["image"])(inputs)

#print(f"image.shape: {T1['image'].shape}")
#print(f"image.shape: {T1['image'].min()}")
#print(f"image.shape: {T1['image'].max()}")
#print(f"image.shape: {T1['image'].mean()}")


# In[9]:


def show_plt(array, title):
    rows, cols = 1, 3
    plt.figure(figsize=(10, 10), dpi=80)
    plt.figure(1)
    plt.title(title, {"fontsize": 12,
                      "color": "black"})

    ax1 = plt.subplot(rows, cols, 1)
    ax2 = plt.subplot(rows, cols, 2)
    ax3 = plt.subplot(rows, cols, 3)

    x = array.shape[0] // 2
    y = array.shape[1] // 2
    z = array.shape[2] // 2

    plt.sca(ax1)
    plt.imshow(array[x, :, :], cmap="gray")
    plt.sca(ax2)
    plt.imshow(array[:, y, :], cmap="gray")
    plt.sca(ax3)
    plt.imshow(array[:, :, z], cmap="gray")
    
    plt.savefig("/data/data_cotune_l2/test1.jpg")

    plt.show()


# In[10]:


#show_plt(T1["image"], "before resampling")


# In[11]:


#T2 = AddChanneld(keys=["image"])(copy.deepcopy(T1))
#T2 = Spacingd(keys=["image"], pixdim=(1,1,1))(T2)
#T2 = Resized(keys=["image"], spatial_size=(256,256,128))(T2)
#T2 = ScaleIntensityd(keys=["image"])(T2)
#print(f"image.shape: {T2['image'].shape}")
#print(f"image.shape: {T2['image'].min()}")
#print(f"image.shape: {T2['image'].max()}")
#print(f"image.shape: {T2['image'].mean()}")


# In[12]:


#show_plt(T2["image"][0], "after resampling")


# In[78]:


def get_foreground_from_set_of_files(scan_id, set_of_files, background_value=0, tolerance=0.00001):
    #set_of_files=["D:\\down\\Pcnls_1\\Pcnls_baseline\\Flair_res111.nii.gz",
    #              "D:\\down\\Pcnls_1\\Pcnls_baseline\\T1_res111.nii.gz",
    #              "D:\\down\\Pcnls_1\\Pcnls_baseline\\T13D_res111.nii.gz",
    #              "D:\\down\\Pcnls_1\\Pcnls_baseline\\T2_res111.nii.gz"]
    load_raw_image(scan_id, 
                   in_files=(["T13D_res111.nrrd"]),
                   out_files=set_of_files)
    foreground = np.zeros((256, 256, 128), dtype=np.int8)
    for i, image_file in enumerate(set_of_files):
        dict = {"image": "/data/down/{}/Pcnls_baseline/{}".format(scan_id, image_file)}
        #print(dict)
        image = LoadImaged(keys=["image"])(dict)
        image = AddChanneld(keys=["image"])(copy.deepcopy(image))
        image = Spacingd(keys=["image"], pixdim=(1,1,1))(image)
        image = Resized(keys=["image"], spatial_size=(256,256,128))(image)
        image = ScaleIntensityd(keys=["image"])(image)
        output_dir = "/data/down/{}/Pcnls_baseline/".format(scan_id)
        #output_dir = os.path.join("D:\\down\\","Pcnls_1\\Pcnls_baseline\\")
        SaveImaged(keys=["image"], output_dir=output_dir)(image)
        image = image["image"][0]
        is_foreground = np.logical_or(image<(background_value-tolerance),
                                      image>(background_value+tolerance))
        #if i == 0:
            #foreground = np.zeros(is_foreground.shape, dtype=np.int8)
        foreground[is_foreground] = 1
    
    return foreground


# In[79]:


set_of_files = (["T13D_res111.nii.gz"])
data_directory="/data/down"
train_df = pd.read_csv("{}/Overall_Survival_1year.csv".format(data_directory))
scan_ids = list(train_df["Patient"])
for scan_id in scan_ids:
    get_foreground_from_set_of_files(scan_id, set_of_files, background_value=0, tolerance=0.00001)
    
    


# In[82]:


def get_foreground_from_set_of_files_withoutsave(scan_id, set_of_files, background_value=0, tolerance=0.00001):
    #set_of_files=["D:\\down\\Pcnls_1\\Pcnls_baseline\\Flair_res111.nii.gz",
    #              "D:\\down\\Pcnls_1\\Pcnls_baseline\\T1_res111.nii.gz",
    #              "D:\\down\\Pcnls_1\\Pcnls_baseline\\T13D_res111.nii.gz",
    #              "D:\\down\\Pcnls_1\\Pcnls_baseline\\T2_res111.nii.gz"]
    load_raw_image(scan_id, 
                   in_files=(["T13D_res111.nrrd"]),
                   out_files=set_of_files)
    foreground = np.zeros((256, 256, 128), dtype=np.int8)
    for i, image_file in enumerate(set_of_files):
        dict = {"image": "/data/down/{}/Pcnls_baseline/{}".format(scan_id, image_file)}
        #print(dict)
        image = LoadImaged(keys=["image"])(dict)
        image = AddChanneld(keys=["image"])(copy.deepcopy(image))
        image = Spacingd(keys=["image"], pixdim=(1,1,1))(image)
        image = Resized(keys=["image"], spatial_size=(256,256,128))(image)
        image = ScaleIntensityd(keys=["image"])(image)
        image = image["image"][0]
        is_foreground = np.logical_or(image<(background_value-tolerance),
                                      image>(background_value+tolerance))
        #if i == 0:
            #foreground = np.zeros(is_foreground.shape, dtype=np.int8)
        foreground[is_foreground] = 1
    
    return foreground


# In[83]:


#set_of_files = ("Flair_res111.nii.gz", "T1_res111.nii.gz", "T13D_res111.nii.gz", "T2_res111.nii.gz")
#nii_foreground = get_foreground_from_set_of_files_withoutsave("Pcnls_1", set_of_files)
#show_plt(nii_foreground, 'nii_foreground')


# In[61]:


def get_multi_index(foreground, rtol=1e-8):
    
    infinity_norm = max(-foreground.min(), foreground.max())
    passes_threshold = np.logical_or(foreground<-rtol*infinity_norm,
                                     foreground>rtol*infinity_norm) 
    #print(passes_threshold)
    coords = np.array(np.where(passes_threshold))
    #print(coords)
   
    start = coords.min(axis=1)
    end = coords.max(axis=1) + 1
    
    start = np.maximum(start-1, 0)
    end = np.minimum(end+1, foreground.shape[:3])
    #print(start, end)
      
    slices = [slice(s, e) for s, e in zip(start, end)]
    return slices


# In[62]:


#crop = get_multi_index(nii_foreground)
#print(crop)
#crop_img_to("C:\\Users\\shezi\\T2_res111\\T2_res111_trans.nii.gz", crop, copy=True)


# In[84]:


def get_crop_images_list(scan_id, set_of_files, norm_set_of_files):
#def get_crop_images(set_of_files, norm_set_of_files):
    #set_of_files=["D:\\down\\Pcnls_1\\Pcnls_baseline\\Flair_res111.nii.gz",
    #              "D:\\down\\Pcnls_1\\Pcnls_baseline\\T1_res111.nii.gz",
    #              "D:\\down\\Pcnls_1\\Pcnls_baseline\\T13D_res111.nii.gz",
    #              "D:\\down\\Pcnls_1\\Pcnls_baseline\\T2_res111.nii.gz"]
    foreground = get_foreground_from_set_of_files_withoutsave(scan_id, set_of_files)
    crop = get_multi_index(foreground)
  
    crop_images = []
    for i, image_file in enumerate(norm_set_of_files):
        file = "/data/down/{}/Pcnls_baseline/{}".format(scan_id, image_file)
        crop_image = crop_img_to(file, crop, copy=True).get_fdata()
        #print(crop_image)
        resize_crop_image = skTrans.resize(crop_image, (224,224,224), order=1, preserve_range=True)
        norm_resize_crop_image = rescale_intensity(resize_crop_image)
        crop_images.append(norm_resize_crop_image)
    return crop_images


# In[85]:


#set_of_files = ("Flair_res111.nii.gz", "T1_res111.nii.gz", "T13D_res111.nii.gz", "T2_res111.nii.gz")
#norm_set_of_files = ("Flair_res111\\Flair_res111_trans.nii.gz", "T1_res111\\T1_res111_trans.nii.gz",
#                     "T13D_res111\\T13D_res111_trans.nii.gz", "T2_res111\\T2_res111_trans.nii.gz")
#nii_crop_images_list = get_crop_images_list("Pcnls_1", set_of_files, norm_set_of_files)


# In[86]:


#get_norm_resize_crop_image = nii_crop_images_list[0]
#print(get_norm_resize_crop_image.dtype)
#print(f"image.shape: {get_norm_resize_crop_image.shape}")
#print(f"image.shape: {get_norm_resize_crop_image.min()}")
#print(f"image.shape: {get_norm_resize_crop_image.max()}")
#print(f"image.shape: {get_norm_resize_crop_image.mean()}")
#show_plt(get_norm_resize_crop_image, "norm_resize_crop_image")


# In[108]:

#norm_set_of_files = ("Flair_res111/Flair_res111_trans.nii.gz", "T1_res111/T1_res111_trans.nii.gz",
#                     "T13D_res111/T13D_res111_trans.nii.gz", "T2_res111/T2_res111_trans.nii.gz")
norm_set_of_files = (["T13D_res111/T13D_res111_trans.nii.gz"])

def load_image_3d(scan_id="Pcnls_1", 
                  data_directory="/data/down", 
                  split="Pcnls_baseline",
                  set_of_files = (["T13D_res111.nii.gz"]),
                  norm_set_of_files=norm_set_of_files):
                  #= ("Flair_res111\\Flair_res111_trans.nii.gz", "T1_res111\\T1_res111_trans.nii.gz",
                                       #"T13D_res111\\T13D_res111_trans.nii.gz", "T2_res111\\T2_res111_trans.nii.gz")
                 
    images_list = get_crop_images_list(scan_id, set_of_files, norm_set_of_files)
    return images_list
    #middle = len(files) // 2 
    #num_imgs2 = num_imgs // 2
    #p1 = max(0, middle - num_imgs2)
    #p2 = min(len(files), middle + num_imgs2)
    #img3d = np.stack([load_dicom_image(f) for f in files[p1:p2]]).T
    
    #if img3d.shape[-1] < num_imgs:
    #    n_zero = np.zeros((img_size, img_size, num_imgs-img3d.shape[-1]))
    #    img3d = np.concatenate((img3d, n_zero), axis=-1)
     
    #if np.min(img3d) < np.max(img3d):
    #    img3d = img3d - np.min(img3d)
    #    img3d = img3d / np.max(img3d)
    
    #return np.expand_dims(img3d, 0)
   




# In[88]:


#torch.tensor(np.array(a))

