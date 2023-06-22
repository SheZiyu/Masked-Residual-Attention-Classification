#!/usr/bin/python
# coding: utf-8

 
import train_92
from train_92 import *

 

#class Model(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.net = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': 2}, in_channels=2)
#        n_features = self.net._fc.in_features
#        self.net._fc = nn.Linear(in_features=n_features, out_features=2, bias=True)
    
#    def forward(self, x):
#        out = self.net(x)
#        return out
    
if __name__ == "__main__":
    set_of_files = (["T13D_res111.nii.gz"])#, "T2_res111.nii.gz")
    norm_set_of_files = (["T13D_res111/T13D_res111_trans.nii.gz"])#, "T2_res111/T2_res111_trans.nii.gz")
            #"T1_res111/T1_res111_trans.nii.gz",
            #"T13D_res111/T13D_res111_trans.nii.gz",
            #"T2_res111/T2_res111_trans.nii.gz")
    modelfile1, modelfile2, modelfile3, modelfile4 = train_all_type(norm_set_of_files=norm_set_of_files, random_state=random_state, mri_type="T13D")#-T2")"all"
    print(modelfile1, modelfile2, modelfile3, modelfile4)
    #a = load_image_3d(scan_id="Pcnls_1")
    #medCAM(a, modelfile, mri_types = ("T13D", "T2")) #"T1", "T13D", "T2"))

    #modelfile = "/data/data_cotune_l2/T13D-T2-densenet_min_loss_model_classification3d_array.pth"  # train_all_type(df_train, df_valid, mri_type="T13D-T2")#"all"
    # print(modelfile)
    # a = load_image_3d(scan_id="Pcnls_1")
    # medCAM(a, modelfile, mri_types = ("T13D", "T2")) #"T1", "T13D", "T2"))
    #pred = predict(modelfile, products_test)
    
#ROC = 0.42857142857142855
#auc = 0.5
#precision = 0.3125
#recall = 0.35714285714285715
#fscore = 0.3333333333333333

    
#a = np.array(a)
#a = torch.tensor(a, dtype=torch.float32)
#print(a.shape)
#print(a[0].shape)
#print(a.squeeze(0).shape)
#show_plt(a[0], "Flair")
#b = a.unsqueeze(0)
#print(b.shape)

 

 
#a = load_image_3d(scan_id="Pcnls_1")
#print(a[0].shape)
#print(np.min(a), np.max(a), np.mean(a), np.median(a))
    


