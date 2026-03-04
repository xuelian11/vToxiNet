import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from vToxiNet_NN import *
from sklearn.metrics import auc, roc_curve, accuracy_score, confusion_matrix
import torch.utils.data as du
import argparse


def prob2binary(pred):
    """function that takes prediction probability tensor and return list of binary class prediction"""
    eval_pred = np.array(pred.data).flatten()

    eval_pred_label = []
    for ep in range(0, len(eval_pred)):
        if eval_pred[ep] < 0.5:
            eval_pred_label.append(0)
        else:
            eval_pred_label.append(1)
    return eval_pred_label


def predict_vaop(pred_file, batch_size, model_folder, model_file):
    """
    :param pred_file:
    :param num_frag:
    :param num_MIE:
    :param num_gene:
    :param batch_size:
    :param model_file:
    :return:
    """
    loaded_model = torch.load(os.path.join(model_folder, model_file))
    # read the prediction file dataframe and use the index when saving predictions
    pred_df = pd.read_csv(pred_file, index_col=0)
    pred_data = ChemDatasetCV(pred_df)

    loaded_model.eval()

    pred_feature, pred_label = pred_data.x_data, pred_data.y_data.reshape(pred_data.n_samples, 1)
    test_loader = du.DataLoader(du.TensorDataset(pred_feature, pred_label), batch_size=batch_size, shuffle=False)

    # test
    test_predict = torch.zeros(0, 0)
    aux_out_map_tt = {}
    batch_num = 0
    for i, (input_data, labels) in enumerate(test_loader):
        features = input_data
        aux_out_map, ptw_hidden_out_map, _ = loaded_model(features)
        if test_predict.size()[0] == 0:
            test_predict = aux_out_map['AO'].data
        else:
            test_predict = torch.cat([test_predict, aux_out_map['AO'].data], dim=0)

        if len(aux_out_map_tt) == 0:
            for key, value in aux_out_map.items():
                aux_out_map_tt[key] = value.detach().numpy().flatten()
        else:
            for key, value in aux_out_map_tt.items():
                new_value = np.concatenate([value, aux_out_map[key].detach().numpy().flatten()], axis=0)
                aux_out_map_tt[key] = new_value

        batch_num += 1
    test_predict_bi = prob2binary(test_predict)
    pred_label1 = np.array(pred_label).flatten()
    
    tn, fp, fn, tp = confusion_matrix(pred_label1, test_predict_bi).ravel()
    sensitivity = tp/(tp+fn)
    specificity = tn/(tn+fp)
    ccr = (sensitivity+specificity)/2
    print(f'sensitivity: {sensitivity}, specificity: {specificity}, ccr: {ccr}')
    print(f'TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}')
    
    test_acc = accuracy_score(pred_label1, test_predict_bi)
    fpr, tpr, _ = roc_curve(pred_label1, np.array(test_predict.data).flatten())
    test_auc = auc(fpr, tpr)
    
    print('External validation accuracy %.3f' % test_acc)
    print('External validation auc %.3f' % test_auc)
    aux_out_map_tt = pd.DataFrame(aux_out_map_tt, index=pred_df.index)
    aux_out_map_tt['Hepatotoxicity'] = pred_df['Hepatotoxicity']
    pred_name = pred_file.split('/')[-1].strip('.csv')
    model_file2 = model_file.strip('.pt')
    aux_out_map_tt.to_csv(model_folder + f'/{model_file2}_{pred_name}.csv')


parser = argparse.ArgumentParser(description='predict vaop')
parser.add_argument('-pred_file', help='Dataset to be predicted', type=str)
parser.add_argument('-batchsize', help='Batchsize', type=int, default=32)
parser.add_argument('-model_folder', help='Model file folder', type=str)
parser.add_argument('-model_file', help='name of the model file', type=str)


opt = parser.parse_args()
torch.set_printoptions(precision=5)

predict_vaop(opt.pred_file, opt.batchsize, opt.model_folder, opt.model_file)

