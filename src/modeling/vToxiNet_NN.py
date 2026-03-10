import sys
import os
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset


class AopDnn(nn.Module):

    def __init__(self, num_frag, num_MIE, num_gene, ptw_direct_gene_map, ptw_size_map, num_hiddens, dG, dG_gene):
        super(AopDnn, self).__init__()
        self.num_frag = num_frag
        self.num_MIE = num_MIE
        self.num_gene = num_gene
        self.ptw_direct_gene_map = ptw_direct_gene_map
        self.num_hiddens = num_hiddens
        self.ptw_layer_list = []
        self.ptw_neighbor_map = {}
        self.ptw_dim_map = self.cal_ptw_dim(ptw_size_map)
        self.ptwGene_neighbor_map = {}

        # construct chemical/MIE/gene layers
        self.construct_chem_nn()
        # construct direct gene layers using pathway_direct_gene_map dictionary
        self.construct_direct_gene_layer(ptw_direct_gene_map)
        # construct later pathway layers with customized connection
        self.construct_pathway_nn(dG, dG_gene)

    def _init_weights(self, module):
        # initialize weights with values from normal distribution
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def construct_chem_nn(self):
        # add MIE layer
        self.add_module('MIE_layer', nn.Linear(self.num_frag, self.num_MIE))
        # add MIE layer
        self.add_module('gene_layer', nn.Linear(self.num_MIE, self.num_gene))

    def construct_direct_gene_layer(self, ptw_direct_gene_map):
        for ptw, gene_set in ptw_direct_gene_map.items():
            if len(gene_set) == 0:
                print('There are no direct gene associated for pathway ', ptw)
            else:
                self.add_module(ptw+'_direct_gene_layer', nn.Linear(self.num_gene, len(gene_set)))

    def cal_ptw_dim(self, ptw_size_map):
        """
        :param ptw_size_map: dictionary stores the pathway and corresponding number of genes associated (direct+child)
        :return: dictionary ptw_dim_map that stores the pathway and corresponding output size, should be same
                 for all the pathways, which is the num_hiddens, use 6
        """
        ptw_dim_map = {}
        for ptw in ptw_size_map.keys():
            num_output = int(self.num_hiddens)
            ptw_dim_map[ptw] = num_output
        return ptw_dim_map

    def construct_pathway_nn(self, dG, dG_gene):
        # ptw_neighbor_map stores the child pathways for each pathway
        for ptw in dG.nodes():
            self.ptw_neighbor_map[ptw] = []
            for child in dG.neighbors(ptw):
                self.ptw_neighbor_map[ptw].append(child)

        for ptw in dG_gene.nodes():
            self.ptwGene_neighbor_map[ptw] = []
            for child in dG_gene.neighbors(ptw):
                self.ptwGene_neighbor_map[ptw].append(child)

        # start from the bottom node (terminal_nodes), get the input/output size and add modules accordingly
        count = 1
        while True:
            leaves = [node for node in dG.nodes() if dG.out_degree(node) == 0]
            if len(leaves) == 0:
                break
            # store the pathways in the same layer in list
            self.ptw_layer_list.append(leaves)
            # add module for each pathway in the same layer
            for ptw in leaves:
                input_size = 0
                # if parent pathways, add input size from child pathways
                if count > 1:
                    for child in self.ptw_neighbor_map[ptw]:
                        input_size += self.ptw_dim_map[child]
                # if bottom pathways, add input size from direct associated genes
                if (count == 1) and (ptw in self.ptw_direct_gene_map):
                    input_size += len(self.ptw_direct_gene_map[ptw])
                # the number of hidden(output) nodes for each pathway
                ptw_hidden = self.ptw_dim_map[ptw]

                # add module for each pathway
                self.add_module(ptw+'_linear_layer', nn.Linear(input_size, ptw_hidden))
                self.add_module(ptw+'_BN_layer', nn.BatchNorm1d(ptw_hidden))
                # add auxiliary layers that will forward to one single node
                self.add_module(ptw+'_aux_linear_layer1', nn.Linear(ptw_hidden,1))
                self.add_module(ptw+'_aux_linear_layer2', nn.Linear(1,1))
            dG.remove_nodes_from(leaves)
            count += 1

    def split_data(self, x):
        layer_data = []
        # get the list of layer data
        start = 0
        end = 0
        for num in [self.num_frag, self.num_MIE, self.num_gene]:
            end += num
            idx = torch.LongTensor([range(start, end)])
            data = x[:, idx]
            data = torch.squeeze(data, 1)
            # print(data.size())
            layer_data.append(data)
            start = end
        return layer_data

    def forward(self, x):
        """x is the tensor contains chemical fragments + assay data + gene data"""
        layer_data = self.split_data(x)

        # define dicts that store output from gene layers, pathway hidden layers and auxiliary layers
        ptw_gene_out_map = {}
        # pathway hidden nodes output map, dimension [1,num_hidden]
        ptw_hidden_out_map = {}
        # auxiliary layer output map, dimension [1, 1]
        aux_out_map = {}
        # get wx+b of MIE layer
        mie_out = self._modules['MIE_layer'](layer_data[0])  # [n_frag, n_MIE]
        
        # element-wise multiplication of the MIE layer output and MIE data, which act as the 'activation' function
        ptw_hidden_out_map['MIE_out'] = torch.mul(mie_out, layer_data[1]) # n_MIE

        # gene layer
        gene_out = self._modules['gene_layer'](ptw_hidden_out_map['MIE_out']) # [n_MIE, n_gene]
        
        # element-wise multiplication of the gene layer output and gene data, which act as our 'activation' function
        ptw_hidden_out_map['gene_out'] = torch.mul(gene_out, torch.abs(layer_data[2]))

        # direct gene layer, take all the gene and only forwarding out genes related to the pathway
        for ptw, _ in self.ptw_direct_gene_map.items():
            ptw_gene_out_map[ptw] = self._modules[ptw+'_direct_gene_layer'](ptw_hidden_out_map['gene_out'])

        # iterate different pathway layers
        for i, layer in enumerate(self.ptw_layer_list):
            for ptw in layer:
                child_input_list = []
                # child nodes' output will become inputs of their parent node
                for child in self.ptw_neighbor_map[ptw]:
                    child_input_list.append(ptw_hidden_out_map[child])
                if ptw in self.ptw_direct_gene_map:
                    child_input_list.append(ptw_gene_out_map[ptw])
                # concatenate along column dimension to get a long input vector
                child_input = torch.concat(child_input_list, 1)
                ptw_hidden_out = self._modules[ptw+'_linear_layer'](child_input)
                # activation function
                Relu_out = torch.relu(ptw_hidden_out)
                ptw_hidden_out_map[ptw] = self._modules[ptw+'_BN_layer'](Relu_out)
                ptw_hidden_out_map[ptw] = Relu_out
                # auxiliary layer outputs
                aux_layer1_out = self._modules[ptw+'_aux_linear_layer1'](ptw_hidden_out_map[ptw])
                aux_out_map[ptw] = torch.sigmoid(self._modules[ptw+'_aux_linear_layer2'](aux_layer1_out))
                # aux_out_map[ptw] = self._modules[ptw + '_aux_linear_layer2'](aux_layer1_out)

        return aux_out_map, ptw_hidden_out_map, ptw_gene_out_map


class ChemDatasetCV(Dataset):

    def __init__(self, dataframe):
        xy = dataframe.to_numpy(dtype=np.float32)
        self.n_samples = xy.shape[0]

        # the last column is the class label, the rest are the features and assay data
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, -1])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples