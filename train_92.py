#!/usr/bin/python
# coding: utf-8

import residual_attention_network
from residual_attention_network import *
import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ResidualAttentionModel_92()#.cuda()
#torch.backends.cudnn.benchmark = True
model = torch.nn.DataParallel(model)
model.to(device)



def get_last_conv_name(model):
    layer_name = None
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv3d):
            layer_name = name
    return layer_name


print(get_last_conv_name(model))
norm_set_of_files = (["T13D_res111/T13D_res111_trans.nii.gz"])
random_state = 12

class Trainer:
    def __init__(
            self,
            model,
            device,
            criterion,
            optimizer,
            scheduler,
            random_state
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.random_state = random_state

        self.best_train_loss = np.inf
        self.best_valid_loss = np.inf
        self.best_valid_ROC = -np.inf
        self.best_valid_fscore = -np.inf
        self.n_patience_train_loss = 0
        self.n_patience_valid_loss = 0
        self.n_patience_valid_ROC = 0
        self.n_patience_valid_fscore = 0
        self.lastmodel_train_loss = None
        self.lastmodel_valid_loss = None
        self.lastmodel_valid_ROC = None
        self.lastmodel_valid_fscore = None
        
    def fit_train_loss(self, epochs, train_loader, valid_loader, save_path, patience):
        writer = SummaryWriter(log_dir="/data/data_cotune_l2/densenet_train_loss")
        start_time = time.time()
        for n_epoch in range(1, epochs + 1):
            self.info_message("EPOCH: {}", n_epoch)
            
            train_loss, train_time = self.train_epoch(train_loader)
            
            valid_loss, valid_auc, valid_ROC, valid_time, precision, recall, fscore, support = self.valid_epoch(valid_loader)

            self.info_message(
                "[Epoch Train: {}] loss: {:.4f}, time: {:.2f} s",
                n_epoch, train_loss, train_time
            )
            writer.add_scalar("Train_loss", train_loss, n_epoch)

        
            writer.add_scalar("Valid_precision", precision, n_epoch)
            writer.add_scalar("Valid_recall", recall, n_epoch)
            writer.add_scalar("Valid_fscore", fscore, n_epoch)
            self.info_message(
                "[Epoch Valid: {}] loss: {:.4f}, auc: {:.4f}, ROC: {:.4f}, precision: {:.4f}, recall: {:.4f}, fscore: {:.4f}, time: {:.2f} s ",
                n_epoch, valid_loss, valid_auc, valid_ROC, precision, recall, fscore, valid_time
            )
            writer.add_scalar("Valid_loss", valid_loss, n_epoch)
            writer.add_scalar("Valid_auc", valid_auc, n_epoch)
            writer.add_scalar("Valid_ROC", valid_ROC, n_epoch)

            if self.best_train_loss > train_loss:
                self.save_model_train_loss(n_epoch, save_path)
                self.info_message(
                    "Loss improved from {:.4f} to {:.4f}. Saved model to '{}'",
                    self.best_train_loss, train_loss, self.lastmodel_train_loss
                )
                self.best_train_loss = train_loss
                self.n_patience_train_loss = 0
            else:
                self.n_patience_train_loss += 1

            #if self.best_valid_ROC < valid_ROC:
            #    self.save_model(n_epoch, save_path)
            #    self.info_message(
            #        "ROC improved from {:.4f} to {:.4f}. Saved model to '{}'",
            #        self.best_valid_ROC, valid_ROC, self.lastmodel
            #    )
            #    self.best_valid_ROC = valid_ROC
            #    self.n_patience = 0
            #else:
            #    self.n_patience += 1
        
            if self.n_patience_train_loss >= patience:
                self.info_message("\nTrain loss didn't improve last {} epochs.", patience)
                break
        time_took = time.time() - start_time
        print("train completed, time took: {}.".format(hms_string(time_took)))
        writer.close()

    def fit_valid_loss(self, epochs, train_loader, valid_loader, save_path, patience):
        writer = SummaryWriter(log_dir="/data/data_cotune_l2/densenet_valid_loss")
        start_time = time.time()
        for n_epoch in range(1, epochs + 1):
            self.info_message("EPOCH: {}", n_epoch)

            train_loss, train_time = self.train_epoch(train_loader)

            valid_loss, valid_auc, valid_ROC, valid_time, precision, recall, fscore, support = self.valid_epoch(valid_loader)

            self.info_message(
                "[Epoch Train: {}] loss: {:.4f}, time: {:.2f} s",
                n_epoch, train_loss, train_time
            )
            writer.add_scalar("Train_loss", train_loss, n_epoch)

            writer.add_scalar("Valid_precision", precision, n_epoch)
            writer.add_scalar("Valid_recall", recall, n_epoch)
            writer.add_scalar("Valid_fscore", fscore, n_epoch)
            self.info_message(
                "[Epoch Valid: {}] loss: {:.4f}, auc: {:.4f}, ROC: {:.4f}, precision: {:.4f}, recall: {:.4f}, fscore: {:.4f}, time: {:.2f} s ",
                n_epoch, valid_loss, valid_auc, valid_ROC, precision, recall, fscore, valid_time
            )
            writer.add_scalar("Valid_loss", valid_loss, n_epoch)
            writer.add_scalar("Valid_auc", valid_auc, n_epoch)
            writer.add_scalar("Valid_ROC", valid_ROC, n_epoch)

            if self.best_valid_loss > valid_loss:
                self.save_model_valid_loss(n_epoch, save_path)
                self.info_message(
                    "Loss improved from {:.4f} to {:.4f}. Saved model to '{}'",
                    self.best_valid_loss, valid_loss, self.lastmodel_valid_loss
                )
                self.best_valid_loss = valid_loss
                self.n_patience_valid_loss = 0
            else:
                self.n_patience_valid_loss += 1

            # if self.best_valid_ROC < valid_ROC:
            #    self.save_model(n_epoch, save_path)
            #    self.info_message(
            #        "ROC improved from {:.4f} to {:.4f}. Saved model to '{}'",
            #        self.best_valid_ROC, valid_ROC, self.lastmodel
            #    )
            #    self.best_valid_ROC = valid_ROC
            #    self.n_patience = 0
            # else:
            #    self.n_patience += 1

            if self.n_patience_valid_loss >= patience:
                self.info_message("\nValid loss didn't improve last {} epochs.", patience)
                break
        time_took = time.time() - start_time
        print("train completed, time took: {}.".format(hms_string(time_took)))
        writer.close()

    def fit_valid_ROC(self, epochs, train_loader, valid_loader, save_path, patience):
        writer = SummaryWriter(log_dir="/data/data_cotune_l2/densenet_valid_ROC")
        start_time = time.time()
        for n_epoch in range(1, epochs + 1):
            self.info_message("EPOCH: {}", n_epoch)

            train_loss, train_time = self.train_epoch(train_loader)

            valid_loss, valid_auc, valid_ROC, valid_time, precision, recall, fscore, support = self.valid_epoch(valid_loader)

            self.info_message(
                "[Epoch Train: {}] loss: {:.4f}, time: {:.2f} s",
                n_epoch, train_loss, train_time
            )
            writer.add_scalar("Train_loss", train_loss, n_epoch)

            writer.add_scalar("Valid_precision", precision, n_epoch)
            writer.add_scalar("Valid_recall", recall, n_epoch)
            writer.add_scalar("Valid_fscore", fscore, n_epoch)

            self.info_message(
                "[Epoch Valid: {}] loss: {:.4f}, auc: {:.4f}, ROC: {:.4f}, precision: {:.4f}, recall: {:.4f}, fscore: {:.4f}, time: {:.2f} s ",
                n_epoch, valid_loss, valid_auc, valid_ROC, precision, recall, fscore, valid_time
            )
            writer.add_scalar("Valid_loss", valid_loss, n_epoch)
            writer.add_scalar("Valid_auc", valid_auc, n_epoch)
            writer.add_scalar("Valid_ROC", valid_ROC, n_epoch)

            if self.best_valid_ROC < valid_ROC:
                self.save_model_valid_ROC(n_epoch, save_path)
                self.info_message(
                    "ROC improved from {:.4f} to {:.4f}. Saved model to '{}'",
                    self.best_valid_ROC, valid_ROC, self.lastmodel_valid_ROC
                )
                self.best_valid_ROC = valid_ROC
                self.n_patience_valid_ROC = 0
            else:
                self.n_patience_valid_ROC += 1

            # if self.best_valid_ROC < valid_ROC:
            #    self.save_model(n_epoch, save_path)
            #    self.info_message(
            #        "ROC improved from {:.4f} to {:.4f}. Saved model to '{}'",
            #        self.best_valid_ROC, valid_ROC, self.lastmodel
            #    )
            #    self.best_valid_ROC = valid_ROC
            #    self.n_patience = 0
            # else:
            #    self.n_patience += 1

            if self.n_patience_valid_ROC >= patience:
                self.info_message("\nValid ROC didn't improve last {} epochs.", patience)
                break
        time_took = time.time() - start_time
        print("train completed, time took: {}.".format(hms_string(time_took)))
        writer.close()

    def fit_valid_fscore(self, epochs, train_loader, valid_loader, save_path, patience):
        writer = SummaryWriter(log_dir="/data/data_cotune_l2/densenet_valid_fscore")
        start_time = time.time()
        for n_epoch in range(1, epochs + 1):
            self.info_message("EPOCH: {}", n_epoch)

            train_loss, train_time = self.train_epoch(train_loader)

            valid_loss, valid_auc, valid_ROC, valid_time, precision, recall, fscore, support = self.valid_epoch(valid_loader)

            self.info_message(
                "[Epoch Train: {}] loss: {:.4f}, time: {:.2f} s",
                n_epoch, train_loss, train_time
            )
            writer.add_scalar("Train_loss", train_loss, n_epoch)

            writer.add_scalar("Valid_precision", precision, n_epoch)
            writer.add_scalar("Valid_recall", recall, n_epoch)
            writer.add_scalar("Valid_fscore", fscore, n_epoch)

            self.info_message(
                "[Epoch Valid: {}] loss: {:.4f}, auc: {:.4f}, ROC: {:.4f}, precision: {:.4f}, recall: {:.4f}, fscore: {:.4f}, time: {:.2f} s ",
                n_epoch, valid_loss, valid_auc, valid_ROC, precision, recall, fscore, valid_time
            )
            writer.add_scalar("Valid_loss", valid_loss, n_epoch)
            writer.add_scalar("Valid_auc", valid_auc, n_epoch)
            writer.add_scalar("Valid_ROC", valid_ROC, n_epoch)

            if self.best_valid_fscore < fscore:
                self.save_model_valid_fscore(n_epoch, save_path)
                self.info_message(
                    "Fscore improved from {:.4f} to {:.4f}. Saved model to '{}'",
                    self.best_valid_fscore, fscore, self.lastmodel_valid_fscore
                )
                self.best_valid_fscore = fscore
                self.n_patience_valid_fscore = 0
            else:
                self.n_patience_valid_fscore += 1

            # if self.best_valid_ROC < valid_ROC:
            #    self.save_model(n_epoch, save_path)
            #    self.info_message(
            #        "ROC improved from {:.4f} to {:.4f}. Saved model to '{}'",
            #        self.best_valid_ROC, valid_ROC, self.lastmodel
            #    )
            #    self.best_valid_ROC = valid_ROC
            #    self.n_patience = 0
            # else:
            #    self.n_patience += 1

            if self.n_patience_valid_fscore >= patience:
                self.info_message("\nValid fscore didn't improve last {} epochs.", patience)
                break
        time_took = time.time() - start_time
        print("train completed, time took: {}.".format(hms_string(time_took)))
        writer.close()

    def train_epoch(self, train_loader):
        self.model.train()
        t = time.time()
        sum_loss = 0

        for step, batch in enumerate(train_loader, 1):
            train_inputs = batch["X"].to(self.device)
            train_labels = batch["y"].to(self.device)
            ind = train_labels.argmax(dim=1)
            ind = torch.Tensor.cpu(ind)

            #weight_decay, dropout
            train_outputs = self.model(train_inputs)
            loss = - train_labels * self.criterion(train_outputs)
            loss = torch.mean(torch.sum(loss, dim=-1))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            sum_loss += loss.item()

            message = "Train Step {}/{}, train_loss: {:.4f}"
            self.info_message(message, step, len(train_loader), sum_loss/step, end="\r")
        self.scheduler.step()
        return sum_loss/len(train_loader), int(time.time()-t)

    
    def valid_epoch(self, valid_loader):
        self.model.eval()
        t = time.time()
        sum_loss = 0
        y_all = []
        outputs_all = []
        pred_all = []
        num_correct = 0.0
        metric_count = 0
        with torch.no_grad():
            for step, batch in enumerate(valid_loader, 1):
                val_inputs = batch["X"].to(self.device)
                val_labels = batch["y"].to(self.device)
                ind = val_labels.argmax(dim=1)
                ind = torch.Tensor.cpu(ind)

                val_outputs = self.model(val_inputs)
                loss = - val_labels * self.criterion(val_outputs)
                loss = torch.mean(torch.sum(loss, dim=-1))

                sum_loss += loss.item()

                value = torch.eq(torch.sigmoid(val_outputs).argmax(dim=1), val_labels.argmax(dim=1))
                metric_count += len(value)
                num_correct += value.sum().item()
        
                y_all.extend(batch["y"].argmax(dim=1).tolist())
                outputs_all.extend(torch.sigmoid(val_outputs).argmax(dim=1).tolist())
                pred_all.extend(torch.sigmoid(val_outputs).tolist())


                message = "Valid Step {}/{}, valid_loss: {:.4f}"
                self.info_message(message, step, len(valid_loader), sum_loss/step, end="\r")
        
            auc = num_correct / metric_count
            pred_all_new = pd.DataFrame(pred_all)
            print(pred_all_new)
            print(y_all)
            pred_all_new = pred_all_new.iloc[:, 1]
            pred_all_new = list(pred_all_new)
            # print(mean_new)
            ROC = roc_auc_score(y_all, pred_all_new)
            #metric_values.append(metric)
            #auc = roc_auc_score(y_all, torch.Tensor.cpu(outputs))
            #outputs_all = [1 if y > 0.5 else 0 for y in outputs_all]
            #auc_ = roc_auc_score(y_all, outputs_all)
            precision, recall, fscore, support = precision_recall_fscore_support(y_all, outputs_all, average='macro')
            
            return sum_loss/len(valid_loader), auc, ROC, int(time.time()-t), precision, recall, fscore, support
        
    
    def save_model_train_loss(self, n_epoch, save_path):
        self.lastmodel_train_loss = "/data/data_cotune_l2/{}-densenet_min_train_loss_model_classification3d_array-{}-92.pth".format(save_path, self.random_state)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_train_loss,#self.best_valid_ROC,
                "n_epoch": n_epoch,
            },
            self.lastmodel_train_loss,
        )

    def save_model_valid_loss(self, n_epoch, save_path):
        self.lastmodel_valid_loss = "/data/data_cotune_l2/{}-densenet_min_valid_loss_model_classification3d_array-{}-92.pth".format(save_path, self.random_state)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_loss,#self.best_valid_ROC,
                "n_epoch": n_epoch,
            },
            self.lastmodel_valid_loss,
        )

    def save_model_valid_ROC(self, n_epoch, save_path):
        self.lastmodel_valid_ROC = "/data/data_cotune_l2/{}-densenet_max_valid_ROC_model_classification3d_array-{}-92.pth".format(save_path, self.random_state)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_ROC,#self.best_valid_ROC,
                "n_epoch": n_epoch,
            },
            self.lastmodel_valid_ROC,
        )

    def save_model_valid_fscore(self, n_epoch, save_path):
        self.lastmodel_valid_fscore = "/data/data_cotune_l2/{}-densenet_max_valid_fscore_model_classification3d_array-{}-92.pth".format(save_path, self.random_state)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_valid_score": self.best_valid_fscore,#self.best_valid_ROC,
                "n_epoch": n_epoch,
            },
            self.lastmodel_valid_fscore,
        )


    
    @staticmethod
    def info_message(message, *args, end="\n"):
        print(message.format(*args), end=end)


 

#torch.cuda.empty_cache()
# = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loader(random_state):
    df_train, df_valid, dummies_train, products_train, y_train, dummies_valid, products_valid, y_valid = ran_state(
        random_state)
    train_transforms = Compose(
        [
            # AsChannelFirst(),

            Rand3DElastic(  # mode=("bilinear", "nearest"),
                prob=0.25,
                sigma_range=(5, 7),
                magnitude_range=(50, 150),
                spatial_size=(224, 224, 224),
                # translate_range=(2, 2, 2),
                # rotate_range=(np.pi/36, np.pi/36, np.pi),
                # scale_range=(0.15, 0.15, 0.15),
                padding_mode="zeros"),

            RandAffine(  # mode=("bilinear", "nearest"),
                prob=0.25,
                spatial_size=(224, 224, 224),
                translate_range=(0.5, 0.5, 0.5),
                rotate_range=(np.pi / 36, np.pi / 36, np.pi / 4),
                scale_range=(0.15, 0.15, 0.15),
                padding_mode="zeros"),
            # Orientation(axcodes="PLI"),
            # RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5),
            # RandSpatialCrop(roi_size=(96, 96, 96)),
            Resize(spatial_size=(224, 224, 224)),
            # RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
            # RandFlip(spatial_axis=0, prob=0.5),
            ScaleIntensity(),
            EnsureType(),
        ]
    )
    valid_transforms = Compose(
        [
            # AsChannelFirst(),
            Resize(spatial_size=(224, 224, 224)),
            # RandRotate(range_x=np.pi / 12, prob=0.5, keep_size=True),
            # RandFlip(spatial_axis=0, prob=0.5),
            ScaleIntensity(),
            EnsureType(),
        ]
    )

    train_data_retriever = Dataset(
        paths=df_train["Patient"].values,
        targets=y_train,
        norm_set_of_files=norm_set_of_files,
        split="Pcnls_baseline",
        transforms=train_transforms
    )
    valid_data_retriever = Dataset(
        paths=df_valid["Patient"].values,
        targets=y_valid,
        norm_set_of_files=norm_set_of_files,
        split="Pcnls_baseline",
        transforms=valid_transforms
    )

    train_loader = DataLoader(  # torch_data.DataLoader(
        train_data_retriever,
        batch_size=2,
        shuffle=True,
        num_workers=1,
        pin_memory=torch.cuda.is_available(),
        collate_fn=pad_list_data_collate
    )
    valid_loader = DataLoader(  # torch_data.DataLoader(
        valid_data_retriever,
        batch_size=2,
        shuffle=False,
        num_workers=1,
        pin_memory=torch.cuda.is_available(),
        collate_fn=pad_list_data_collate
    )

    return train_loader, valid_loader


def train_all_type(norm_set_of_files=norm_set_of_files, random_state=random_state, mri_type="all"):
    train_loader, valid_loader = loader(random_state)

    # print(relationship.shape)
    # model = Model()
    # model = monai.networks.nets.DenseNet121(spatial_dims=3, in_channels=2, out_channels=3)#.to(device)
    # model = monai.networks.nets.resnet.resnet50(n_input_channels=2, n_classes=2)#.to(device)
    # model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    # criterion = torch.nn.CrossEntropyLoss()#torch_functional.binary_cross_entropy_with_logits
    criterion = torch.nn.LogSoftmax(dim=-1)
    # trade_off = 1.3

    trainer = Trainer(
        model,
        device,
        criterion,
        optimizer,
        scheduler,
        random_state,
    )
    
    history_train_loss = trainer.fit_train_loss(
        250,
        train_loader,
        valid_loader,
        f"{mri_type}",
        50,
    )
    history_valid_loss = trainer.fit_valid_loss(
        250,
        train_loader,
        valid_loader,
        f"{mri_type}",
        50,
    )
    history_valid_ROC = trainer.fit_valid_ROC(
        250,
        train_loader,
        valid_loader,
        f"{mri_type}",
        50,
    )
    history_valid_fscore = trainer.fit_valid_fscore(
        250,
        train_loader,
        valid_loader,
        f"{mri_type}",
        50,
    )
    return trainer.lastmodel_train_loss, trainer.lastmodel_valid_loss, trainer.lastmodel_valid_ROC, trainer.lastmodel_valid_fscore
#print("done2")
#if __name__ == "__main__":
#modelfile = train_all_type(df_train, df_valid, "all")
#print(modelfile)

