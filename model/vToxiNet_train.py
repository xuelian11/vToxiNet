import argparse
import time
import numpy as np
import torch
from sklearn.metrics import auc, roc_curve, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import torch.utils.data as du
from vToxiNet_NN import *
from vToxiNet_NN_dropout import *
from vToxiNet_utils import *


def create_ptw_mask(ptw_direct_gene_map, num_gene):
    """
    :param ptw_direct_gene_map: dictionary storing {pathway:gene_set} map
    :param num_gene: the total number of genes
    :return: dictionary storing {ptw: mask_matrix}, where mask_matrix
            is a tensor with
                number of rows: the number of gene directly associated with a pathway
                number of columns: the total number of genes
                value: 1 if a gene is in gene_set, 0 if not
    """
    ptw_mask_map = {}
    for ptw, gene_set in ptw_direct_gene_map.items():
        mask = torch.zeros(len(gene_set), num_gene)
        for i, gene_id in enumerate(gene_set):
            mask[i, gene_id] = 1
        # mask_gpu = torch.autograd.Variable(mask.cuda(CUDA_ID))
        # term_mask_map[term] = mask_gpu
        # if do not use GPU and also the torch.autograd.Variable is deprecated
        ptw_mask_map[ptw] = mask

    return ptw_mask_map


def prob2binary(pred):
    """function that takes prediction probability tensor and return list of binary class prediction"""
    eval_pred = np.array(pred.data).flatten()

    eval_pred_label = []
    for ep in range(0, len(eval_pred)):
        if eval_pred[ep] <= 0.5:
            eval_pred_label.append(0)
        else:
            eval_pred_label.append(1)
    return eval_pred_label


def cal_ccr(label, preds_bi):
    tn, fp, fn, tp = confusion_matrix(label, preds_bi).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    ccr = (sensitivity + specificity) / 2
    return ccr


def train_model(ptw_size_map, ptw_direct_gene_map, dG, dG_gene, train, vali, num_frag, num_MIE, num_gene, batch_size,
                model_save_folder, train_epochs, learning_rate, num_hiddens, eps, decay, pam_ratio, drop_rate):

    best_model = 0
    max_corr = None
    patience = 0
    early_stop = False

    train_data = ChemDatasetCV(train)
    vali_data = ChemDatasetCV(vali)

    train_feature, train_label = train_data.x_data, train_data.y_data.reshape(train_data.n_samples, 1)
    vali_feature, vali_label = vali_data.x_data, vali_data.y_data.reshape(vali_data.n_samples, 1)

    print(drop_rate)
    # vAOP neural network
    if drop_rate > 0:
        model = AopDnnDropout(num_frag, num_MIE, num_gene, ptw_direct_gene_map, ptw_size_map, num_hiddens, dG, dG_gene, drop_rate)
    elif drop_rate == 0:
        model = AopDnn(num_frag, num_MIE, num_gene, ptw_direct_gene_map, ptw_size_map, num_hiddens, dG, dG_gene)
    else:
        raise Exception('define dropout rate, 0 means no dropout')

    # print(model._modules)
    print('the number of modules is %d' % len(model._modules))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, eps=eps, weight_decay=decay)
    ptw_mask_map = create_ptw_mask(model.ptw_direct_gene_map, num_gene)
    optimizer.zero_grad()

    for name, param in model.named_parameters():
        ptw_name = name.split('_')[0]
        # generate the initial weights for direct annotated genes for a pathway
        if "_direct_gene_layer.weight" in name:
            # element-wise vector multiplication with mask matrix
            param.data = torch.mul(param.data, ptw_mask_map[ptw_name]) * pam_ratio
        else:
            param.data = param.data * pam_ratio

    train_loader = du.DataLoader(du.TensorDataset(train_feature, train_label), batch_size=batch_size, shuffle=False)
    vali_loader = du.DataLoader(du.TensorDataset(vali_feature, vali_label), batch_size=batch_size, shuffle=False)

    for epoch in range(train_epochs):
        if early_stop is True:
            break
        model.train()
        train_predict = torch.zeros(0, 0)
        # aux_out_map_tt = {}
        for i, (input_data, labels) in enumerate(train_loader):
            input_data.requires_grad = True
            features = input_data
            # zero out the gradients
            optimizer.zero_grad()
            # forward
            aux_out_map, ptw_hidden_out_map, _ = model(features)
            # concatenate all the predictions
            if train_predict.size()[0] == 0:
                train_predict = aux_out_map['AO'].data
            else:
                train_predict = torch.cat([train_predict, aux_out_map['AO'].data], dim=0)

            # loss
            total_loss = 0
            for name, output in aux_out_map.items():
                loss = nn.BCELoss()
                # using smooth labels (0.05, 0.95) instead of (0, 1)
                smoothed_labels = labels * 0.9 + 0.05
                if name == 'AO':
                    total_loss += loss(output, smoothed_labels)
                else:
                    total_loss += 0.2 * loss(output, smoothed_labels)
            total_loss.backward()

            # make sure weights for not associated genes are zeros, using the mask matrix
            for name, param in model.named_parameters():
                if "_direct_gene_layer.weight" in name:
                    ptw_name = name.split('_')[0]
                    param.grad.data = torch.mul(param.grad.data, ptw_mask_map[ptw_name])
            optimizer.step()

        train_pred_label = prob2binary(train_predict)
        train_label1 = np.array(train_label).flatten().tolist()
        train_ccr = cal_ccr(train_label1, train_pred_label)
        fpr, tpr, _ = roc_curve(train_label1, np.array(train_predict.data).flatten())
        train_auc = auc(fpr, tpr)

        model.eval()
        with torch.no_grad():
            vali_predict = torch.zeros(0, 0)
            for i, (input_data, labels) in enumerate(vali_loader):
                features = input_data
                # get prediction
                aux_out_map_t, ptw_hidden_out_map, _ = model(features)
                # concatenate all the predictions
                if vali_predict.size()[0] == 0:
                    vali_predict = aux_out_map_t['AO'].data
                else:
                    vali_predict = torch.cat([vali_predict, aux_out_map_t['AO'].data], dim=0)
                # validation loss
                validation_loss = 0
                for name, output in aux_out_map_t.items():
                    loss = nn.BCELoss()
                    smoothed_labels = labels * 0.9 + 0.05
                    if name == 'AO':
                        validation_loss += loss(output, smoothed_labels)
                    else:
                        validation_loss += 0.2*loss(output, smoothed_labels)

            # vali_predict = torch.sigmoid(vali_predict)
            vali_pred_label = prob2binary(vali_predict)
            vali_label1 = np.array(vali_label).flatten().tolist()
            vali_ccr = cal_ccr(vali_label1, vali_pred_label)
            fpr, tpr, _ = roc_curve(vali_label1, np.array(vali_predict.data).flatten())
            vali_auc = auc(fpr, tpr)

            print("Epoch %d train_ccr %.3f vali_ccr %.3f vali_auc %.3f train_loss %.3f vali_loss %.3f" % (epoch,
                                                                                            train_ccr, vali_ccr,vali_auc,
                                                                                            total_loss,validation_loss))

        # using validation ccr for early stopping
        if max_corr is None:
            max_corr = 0
        if train_ccr >= 0.75 and vali_auc > 0.5 and vali_ccr > 0.5:
            if max_corr >= vali_ccr:
                print(max_corr, patience)
                patience += 1
                if patience >= 10:
                    early_stop = True
                    torch.save(model, model_save_folder + f'/model_epoch_{epoch}_final.pt')
            else:
                max_corr = vali_ccr
                best_model = epoch
                patience = 0

    # save stat_dict()
    torch.save(model.state_dict(), model_save_folder + '/model_final.pt')
    print("Best performance model is epoch %d" % best_model)


parser = argparse.ArgumentParser(description='Train vAOP DNN')
parser.add_argument('-nfrag', help='the number of chemical fragments', type=int, default=834)
parser.add_argument('-nMIE', help='the number of MIE assays', type=int, default=215)
parser.add_argument('-ngene', help='the number of genes associated with the pathways', type=int, default=280)
parser.add_argument('-recto', help='name of txt file contains the reactome pathway info', type=str)
parser.add_argument('-train_folder', help='name of the folder containing the train file', type=str)
parser.add_argument('-whole_train_file', help='name of csv file contains the whole train data', type=str)
parser.add_argument('-gene2id', help='name of the txt file contains ID:gene_symbol', type=str)
parser.add_argument('-epochs', help='training epochs', type=int, default=150)
parser.add_argument('-lr', help='learning rate', type=float)
parser.add_argument('-batchsize', help='batch size', type=int, default=32)
parser.add_argument('-modeldir', help='folder to save trained models', type=str)
parser.add_argument('-n_hiddens', help='the number of hidden nodes for each pathway', type=int, default=6)
parser.add_argument('-eps', help='eps in adam optimizer', type=float)
parser.add_argument('-decay', help='weight_decay in adam optimizer', type=float)
parser.add_argument('-pam_ratio', help='param ration in mask', type=float)
parser.add_argument('-drop_rate', help='dropout rate in fully connected layers', type=float)

arg = parser.parse_args()
torch.set_printoptions(precision=5) # 5 decimals

whole_train = pd.read_csv(arg.train_folder + '/' + arg.whole_train_file, index_col=0)
assert(whole_train.shape[1] == arg.nfrag + arg.nMIE + arg.ngene + 1)

X = whole_train.iloc[:, :-1].copy()
y = whole_train.iloc[:, -1].copy()
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
vali_predict_list = []
vali_label_list = []
train_index, vali_index = skf.split(X, y)
X_train, X_vali = X.iloc[train_index, :], X.iloc[vali_index, :]
y_train, y_vali = y.iloc[train_index], y.iloc[vali_index]
vali_label_list.append(y_vali)
train = pd.concat([X_train, y_train], axis=1)
vali = pd.concat([X_vali, y_vali], axis=1)
train = train.sample(frac=1)
save_name = arg.whole_train_file.rstrip('.csv')

gene2id_mapping = load_mapping(arg.gene2id)
# hierarchy with 5 curated pathway layers
dG, dG_gene, root, ptw_size_map, ptw_direct_gene_map = load_reactome_ori(arg.recto, gene2id_mapping)
num_hiddens = arg.n_hiddens
decay = arg.decay
pam_ratio = arg.pam_ratio
drop_rate = arg.drop_rate

train_model(ptw_size_map, ptw_direct_gene_map, dG, dG_gene, train, vali, arg.nfrag, arg.nMIE, arg.ngene,
        arg.batchsize, arg.modeldir, arg.epochs, arg.lr, num_hiddens, arg.eps, decay, pam_ratio, drop_rate)


