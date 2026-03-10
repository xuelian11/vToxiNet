
import numpy as np
import pandas as pd 
import torch
from functools import reduce
from operator import concat
from vToxiNet_NN import *
import argparse


# This function computes a small increment as a smoothing factor (applied when denominator is 0)
# for a list of input tensors
def compute_increment_values(m_list, incre_prop=0.001):
	"""
	:param m_list: list of tensors
	:param incre_prop: small increment proportion of smallest absolute non-zero element
	"""
	# 1. Compute smoothing increment factor
	# iterate by tensor in the list 
	m_abs_min = torch.zeros(len(m_list)).double()
	for ml in range(0, len(m_list)):
		if len(m_list[ml]) > 0:
			# identify non-zero elements of current tensor
			m_list_n0 = m_list[ml][torch.nonzero(m_list[ml], as_tuple=True)]
			# if no non-zero element, move on to next one
			if len(m_list_n0) == 0:
				continue
			# if non-zero elements exist, compute the smallest absolute non-zero element
			else:
				m_abs_min[ml] = m_list_n0.abs().min() 
	# identify the smallest absolute non-zero element among all tensors, then compute increment   
	increment = m_abs_min[torch.nonzero(m_abs_min, as_tuple=True)].abs().min() * incre_prop

	return increment
    
    
# This function perform matrix multiplication while using smoothing factors to avoid 0 results.
def fill_in_zero_matmul(m1, m2, m1_incre, m2_incre, m1_sign, m2_sign):
	"""
	:param m1: tensor that represents matrix 1
	:param m2: tensor that represents matrix 2
	:param m1_incre: smoothing increment factor of matrix 1
	:param m2_incre: smoothing increment factor of matrix 2
	:param m1_sign: sign of adjusting matrix 1 with increment (1: +; -1: -)
	:param m2_sign: sign of adjusting matrix 2 with increment (1: +; -1: -)
	:return:
	"""
	# 1. Perform matrix multiplication
	mul = torch.matmul(m1, m2)
	if (mul == 0).sum() > 0:
		# identify row and column id in which the multiplication results are 0 
		mul0_id = torch.where(mul == 0)
		# adjust identified rows of matrix 1 by smoothing increment factor   
		m1_incres = torch.zeros(m1.shape[0], 1).double()
		m1_incres[torch.unique(mul0_id[0]), :] = m1_incre
		m1 = m1 + m1_incres * m1_sign
		# adjust identified columns of matrix 1 by smoothing increment factor  
		m2_incres = torch.zeros(1, m2.shape[1]).double()
		m2_incres[:, torch.unique(mul0_id[1])] = m2_incre
		m2 = m2 + m2_incres * m2_sign
		# perform matrix multiplication again
		mul = torch.matmul(m1, m2)

	return m1, m2, mul


# This function implements generic (combined LRP-γ and LRP-ϵ) rule for propagating relevance from hidden layer k to j
def gamma_epsilon_rule(gamma, epsilon, r_k, a_j, weight_jk, a_incre, w_incre):
	"""
	:param gamma: coefficient for positive weights in LRP-γ rule
	:param epsilon: adjusting factor of denominator in LRP-ϵ rule
	:param r_k: tensor that contains relevance of neurons in hidden layer k
	:param a_j: tensor that contains neuron output after activation by hidden layer j
	:param weight_jk: tensor that contains weight parameters between hidden layer k and j of learned DTox m
	:param a_incre: smoothing increment factor of a_j
	:param w_incre: smoothing increment factor of weight_jk
	:return:
	"""
	# 1. Clone tensors into double tensors in order to improve the precision of calculation
	r_k = r_k.clone().detach().double()
	a_j = a_j.clone().detach().double()
	weight_jk = weight_jk.clone().detach().double()

	# 2. Perform step 1 of calculation
	weight_jk0 = torch.zeros_like(weight_jk)
	pos_weight_jk = torch.where(weight_jk > 0, weight_jk, weight_jk0)
	weight_jk1 = weight_jk + gamma * pos_weight_jk

	# 3. Perform step 2 of calculation
	a_j_new, weight_jk1_new, weight_j_sum = fill_in_zero_matmul(a_j, weight_jk1.T, a_incre, w_incre, 1, 1)
	weight_j_sum1 = weight_j_sum + epsilon * torch.std(weight_j_sum.double())	

	# 4. Perform step 3 of calculation
	weight_j_rk = r_k/weight_j_sum1 

	# 5. Perform step 4 of calculation
	r_j = torch.matmul(weight_j_rk, weight_jk1_new.T) * a_j_new

	return r_j


# This function implements αβ rule for propagating relevance from hidden layer k to j
def alpha_beta_rule(alpha, beta, r_k, a_j, weight_jk, a_incre, w_incre):
	"""
	:param alpha: coefficient for positive weights in LRP-αβ rule
	:param beta: coefficient for negative weights in LRP-αβ rule
	:param r_k: tensor that contains relevance of neurons in hidden layer k
	:param a_j: tensor that contains neuron output after activation by hidden layer j
	:param weight_jk: tensor that contains weight parameters between hidden layer k and j of learned DTox model
	:param a_incre: smoothing increment factor of a_j
	:param w_incre: smoothing increment factor of weight_jk
	:return:
	"""
	# 1. Clone tensors into double tensors in order to improve the precision of calculation
	r_k = r_k.clone().detach().double()
	a_j = a_j.clone().detach().double()
	weight_jk = weight_jk.clone().detach().double() 

	# 2. Perform step 1 of calculation
	weight_jk0 = torch.zeros_like(weight_jk)
	pos_weight_jk = torch.where(weight_jk > 0, weight_jk, weight_jk0)
	neg_weight_jk = torch.where(weight_jk <= 0, weight_jk, weight_jk0)

	# 3. Perform step 2 of calculation
	a_j_pos, pos_weight_jk_new, pos_weight_j_sum = fill_in_zero_matmul(a_j, pos_weight_jk.T, a_incre, w_incre, 1, 1)
	a_j_neg, neg_weight_jk_new, neg_weight_j_sum = fill_in_zero_matmul(a_j, neg_weight_jk.T, a_incre, w_incre, 1, -1)

	# 4. Perform step 3 of calculation
	pos_weight_j_rk = r_k/pos_weight_j_sum 
	neg_weight_j_rk = r_k/neg_weight_j_sum 

	# 5. Perform step 4 of calculation
	r_j = alpha * torch.matmul(pos_weight_j_rk, pos_weight_jk_new.T) * a_j_pos - beta * torch.matmul(neg_weight_j_rk, neg_weight_jk_new.T) * a_j_neg

	return r_j


# This function implements special rule for propagating relevance from first hidden layer to input layer
def input_layer_rule(low, high, r_j, x_i, weight_ij, x_incre, w_incre):
	"""
	:param low: lower bound of input feature values
	:param high: upper bound of input feature values
	:param r_j: tensor that contains relevance of neurons in first hidden layer (layer j)
	:param x_i: tensor that contains feature value of input layer (layer i)
	:param weight_ij: tensor that contains weight parameters between input layer (i) and first hidden layer (j) of learned DTox model
	:param x_incre: smoothing increment factor of x_j
	:param w_incre: smoothing increment factor of weight_ij
	:return: relevance score calculated under specified rule
	"""
	# 1. Clone tensors into double tensors in order to improve the precision of calculation
	r_j = r_j.clone().detach().double()
	x_i = x_i.clone().detach().double()
	weight_ij = weight_ij.clone().detach().double() 

	# 2. Perform step 1 of calculation
	weight_ij0 = torch.zeros_like(weight_ij)
	pos_weight_ij = torch.where(weight_ij > 0, weight_ij, weight_ij0)
	neg_weight_ij = torch.where(weight_ij <= 0, weight_ij, weight_ij0)	

	# 3. Perform step 2 of calculation
	x_low = x_i * 0 + low 
	x_high = x_i * 0 + high 
	x_i_new, weight_ij_new, weight_i_sum = fill_in_zero_matmul(x_i, weight_ij.T, x_incre, w_incre, 1, 1)
	weight_i_sum = weight_i_sum - torch.matmul(x_low, pos_weight_ij.T) - torch.matmul(x_high, neg_weight_ij.T)	

	# 4. Perform step 3 of calculation
	weight_i_rj = r_j/(weight_i_sum + (weight_i_sum == 0).double() * 1e-6)

	# 5. Perform step 4 of calculation
	r_i = x_i_new * torch.matmul(weight_i_rj, weight_ij_new.T) - x_low * torch.matmul(weight_i_rj, pos_weight_ij) - x_high * torch.matmul(weight_i_rj, neg_weight_ij)

	return r_i


def load_mapping(mapping_file):
    """
    :param mapping_file: the filename of the csv containing the ID_number and gene_symbol in each line
    :return: a dict storing the {ID_number: gene_symbol} mapping
    """
    df = pd.read_csv(mapping_file)

    mapping = dict(zip(df['ID'], df['gene']))

    return mapping


# First step is to extract the layers from our model. Once we extracted all the useful modules from our network
# we will propagate the input data X through the network
def LRP_individual(loaded_model, ptw_id_map, id2gene_mapping, input_data, rule, rule_factor1, rule_factor2, hidden_size):
	"""
	:param loaded_model: trained model
	:param ptw_id_map: ptw:ID pair
	:param id2gene_mapping: a dict storing the {ID_number:gene_symbol} mapping
	:param input_data: input data contain chemical fragments, MIE profile and gene profile
	:param rule: which rule to use
	:param rule_factor1: first parameter for a specified rule
	:param rule_factor2: second parameter for a specified rule
	:param hidden_size: number of hidden neuron for each module
	:return: LRP relevance scores for each pathway module,
	after we get the neuron_relevance_df, we can get the R score for each bottom pathway, the previous layer are the
	ptw_direct_gene_layer for all the bottom pathways, whose weights are multiplied with the masked matrix to make sure
	that only the values of associated genes can be forwarded to the pathway layer. so each bottom pathway is actually
	connected to all the genes, but the weight values for not associated genes are 0.
	So, the bottom pathway --> gene --> MIE --> frag are fully connected
	"""
	# 1. perform forward propagation based on learned model
	aux_out_map, ptw_hidden_out_map, ptw_gene_out_map = loaded_model(input_data)
	layer_data = loaded_model.split_data(input_data)
	num_layers = len(loaded_model.ptw_layer_list)

	L = len(ptw_id_map)
	print(f'The number of ptw in ptw_id_map is {L}.')
	# iterate from last layer to first layer

	N_instance = input_data.shape[0]

	layer_input_list = [[] for _ in range(L + 2)]
	layer_output_list = [[] for _ in range(L + 2)]
	layer_weight_list = [[] for _ in range(L + 2)]
	layer_child_id = [[] for _ in range(L + 2)]
	layer_child_size = [[] for _ in range(L + 2)]

	# iterate each pathway, find its child pathway, compute layer_input a_j, layer_weight w_j_k,
	# layer_child_id and layer_child_size
	for pathways in loaded_model.ptw_layer_list:
		for ptw in pathways:
			# create lists to store 1) ptw input from output of child pathways and 2) child pathway sizes
			child_input_list = []
			child_sizes_list = []
			# get output from child pathways
			for child in loaded_model.ptw_neighbor_map[ptw]:
				child_sizes_list.append(hidden_size)
				child_input_list.append(ptw_hidden_out_map[child])
				layer_child_id[ptw_id_map[ptw]].append(ptw_id_map[child])
			if len(child_input_list) > 0:
				layer_input_list[ptw_id_map[ptw]] = torch.cat(child_input_list, 1)
			layer_child_size[ptw_id_map[ptw]] = child_sizes_list

			# get ptw weight
			module_name = f'{ptw}_linear_layer'
			layer_weight_list[ptw_id_map[ptw]] = loaded_model._modules[module_name].weight
			layer_output_list[ptw_id_map[ptw]] = ptw_hidden_out_map[ptw]
	# get the above information for the 'AO_aux_linear_layer1', shape = [hidden_size, 1]
	layer_input_list[L] = ptw_hidden_out_map['AO']
	layer_output_list[L] = loaded_model._modules['AO_aux_linear_layer1'](ptw_hidden_out_map['AO'])
	layer_weight_list[L] = loaded_model._modules['AO_aux_linear_layer1'].weight
	layer_child_id[L] = [1119]
	layer_child_size[L] = [hidden_size]

	# get the above information for the 'AO_aux_linear_layer2', shape = [1, 1]
	layer_input_list[L + 1] = layer_output_list[L]
	layer_output_list[L + 1] = aux_out_map['AO']
	layer_weight_list[L + 1] = loaded_model._modules['AO_aux_linear_layer2'].weight
	layer_child_id[L + 1] = [1120]
	layer_child_size[L + 1] = [1]

	# 2. perform backward propagation based on specified propagation rule
	# create list of lists for:
	# 1. current_relevance_list: known relevance scores for each neuron in current layer k
	# 2. neuron_relevance_list: calculated relevance score for each neuron in layer j
	# 3. module_relevance_list: combine layer j neurons that belong to the same module as the module relevance
	# total length for each list is L+1
	# iterate in reverse order and keep track of computed neuron/module relevance score
	current_relevance_list = [[] for _ in range(L + 2)]
	current_relevance_list[L + 1].append(aux_out_map['AO'])  # final output
	neuron_relevance_list = [[] for _ in range(L + 2)]
	module_relevance_list = [[] for _ in range(L + 2)]

	# compute smoothing increment factor for node value and weight tensors separately
	a_incre = compute_increment_values(layer_input_list)
	w_incre = compute_increment_values(layer_weight_list)

	for k in range(L+2)[::-1]:
		# for nodes that are connected to parent node, sum the relevance scores propagated from parent nodes
		# and take the sum as neuron relevance score
		if len(current_relevance_list[k]) > 0:
			neuron_relevance_list[k] = torch.stack(current_relevance_list[k], dim=0).sum(dim=0)
		module_relevance_list[k] = neuron_relevance_list[k].sum(dim=1).view(N_instance, 1)
		# for nodes in hidden layers, propagate neuron relevance score of current node backward to its child nodes
		# for nodes in the second hidden layer and beyond, propagate according to the specified rule
		if len(layer_child_id[k]) > 0:
			if rule == 'gamma-epsilon':
				kptw_child_relevance = gamma_epsilon_rule(rule_factor1, rule_factor2, neuron_relevance_list[k],
													layer_input_list[k], layer_weight_list[k], a_incre, w_incre)
			if rule == 'alpha-beta':
				kptw_child_relevance = alpha_beta_rule(rule_factor1, rule_factor2, neuron_relevance_list[k],
													layer_input_list[k], layer_weight_list[k], a_incre, w_incre)

			# obtain the lower and upper column index of each child node in the relevance score matrix
			kptw_child_len = len(layer_child_size[k])  # how many child nodes
			kptw_child_relevance_id = np.insert(layer_child_size[k], 0, 0).cumsum()  # [6,6,6] -> [0,6,12,18]

			# iterate by child nodes
			for i in range(kptw_child_len):
				# assign the propagated neuron relevance score to the current child node
				i_id = layer_child_id[k][i]
				current_relevance_list[i_id].append(
					kptw_child_relevance[:, kptw_child_relevance_id[i]:kptw_child_relevance_id[i+1]])

	# 3. concatenate propagated relevance scores of neurons
	neuron_relevance_name_list = [[] for _ in range(L+1)]
	for i in range(L+1):
		# name each neuron in each module by the format 'A_B':
		# A is the module ID, B is the order of the neuron in the module
		neuron_relevance_name_list[i] = [f'{str(i)}_{str(j)}' for j in range(neuron_relevance_list[i].shape[1])]
	# concatenate propagated relevance scores of all neurons, output a dataframe
	neuron_relevance_df = pd.DataFrame(torch.cat(neuron_relevance_list[:L+1], dim=1).detach().numpy())
	neuron_relevance_df.columns = reduce(concat, neuron_relevance_name_list)

	# concatenate propagated relevance scores of all modules, output a dataframe
	module_relevance_df = pd.DataFrame(torch.concat(module_relevance_list[:L+1], dim=1).detach().numpy())
	model_name = model_file.rstrip('.pt')
	pred_name = pred_file.split('/')[-1].rstrip('.csv')
	neuron_relevance_df.to_csv(model_dir+f'/{model_name}_{pred_name}_neuron_relevance.csv')
	module_relevance_df.to_csv(model_dir+f'/{model_name}_{pred_name}_module_relevance.csv')
	# save the ptw_id_map
	pd.DataFrame.from_dict(ptw_id_map, orient='index').to_csv(model_dir+'/genetrim_ptw_id_map.csv')

	# 4. based on relevance scores for the bottom layer modules, calculate relevance score for direct gene
	# get the hidden output of the gene layer, which is the input of the bottom pathway layer
	# for each of the bottom pathway layer, calculate the R score for gene layer,
	# and the final gene layer R score will be the sum
	# the gene layer and the bottom pathway layer are fully connected,
	# just some weights are zeroed out to map the specific gene-pathway relationship

	bottom_layers = loaded_model.ptw_layer_list[0]
	direct_gene_relevance_list = [[] for _ in range(len(bottom_layers))]
	gene_relevance_list = [[] for _ in range(len(bottom_layers))]

	for ptw in bottom_layers:
		ptw_id = ptw_id_map[ptw]
	# calculate the gene R score for each direct gene layer
		if rule == 'gamma-epsilon':
			direct_gene_relevance_list[ptw_id] = gamma_epsilon_rule(rule_factor1, rule_factor2, neuron_relevance_list[ptw_id],
															ptw_gene_out_map[ptw],
															loaded_model._modules[ptw+'_linear_layer'].weight,
															a_incre, w_incre)
			gene_relevance_list[ptw_id] = gamma_epsilon_rule(rule_factor1, rule_factor2, direct_gene_relevance_list[ptw_id],
															ptw_hidden_out_map['gene_out'],
															loaded_model._modules[ptw+'_direct_gene_layer'].weight,
															a_incre, w_incre)

		if rule == 'alpha-beta':
			direct_gene_relevance_list[ptw_id] = alpha_beta_rule(rule_factor1, rule_factor2, neuron_relevance_list[ptw_id],
															ptw_gene_out_map[ptw],
															loaded_model._modules[ptw+'_linear_layer'].weight,
															a_incre, w_incre)

			gene_relevance_list[ptw_id] = alpha_beta_rule(rule_factor1, rule_factor2, direct_gene_relevance_list[ptw_id],
															ptw_hidden_out_map['gene_out'],
															loaded_model._modules[ptw+'_direct_gene_layer'].weight,
															a_incre, w_incre)

	# from direct gene layers calculate full gene relevance for the gene layer

	# sum the gene R scores generated from different bottom pathways
	gene_relevance = torch.stack(gene_relevance_list, dim=0).sum(dim=0)
	# with gene relevance, we can calculate the MIE R score
	if rule == 'gamma-epsilon':
		MIE_relevance = gamma_epsilon_rule(rule_factor1, rule_factor2, gene_relevance, ptw_hidden_out_map['MIE_out'],
										loaded_model._modules['gene_layer'].weight, a_incre, w_incre)
	if rule == 'alpha-beta':
		MIE_relevance = alpha_beta_rule(rule_factor1, rule_factor2, gene_relevance, ptw_hidden_out_map['MIE_out'],
										loaded_model._modules['gene_layer'].weight, a_incre, w_incre)

	# calculate fragment R score based on MIE R score
	frag_relevance = input_layer_rule(0, 1, MIE_relevance, layer_data[0],
									loaded_model._modules['MIE_layer'].weight, a_incre, w_incre)
	# save gene, MIE, fragment relevance to dataframe
	gene_relevance_df = pd.DataFrame(gene_relevance.detach().numpy())
	gene_relevance_df.rename(columns=id2gene_mapping, inplace=True)
	MIE_relevance_df = pd.DataFrame(MIE_relevance.detach().numpy())
	frag_relevance_df = pd.DataFrame(frag_relevance.detach().numpy())
	gene_ptw_df = pd.concat([gene_relevance_df, module_relevance_df], axis=1)

	gene_relevance_df.to_csv(model_dir + f'/{model_name}_{pred_name}_gene_relevance.csv')
	MIE_relevance_df.to_csv(model_dir + f'/{model_name}_{pred_name}_MIE_relevance.csv')
	frag_relevance_df.to_csv(model_dir + f'/{model_name}_{pred_name}_frag_relevance.csv')
	gene_ptw_df.to_csv(model_dir + f'/{model_name}_{pred_name}_geneptw_relevance.csv')

	return neuron_relevance_df, module_relevance_df, gene_relevance_df, MIE_relevance_df, frag_relevance_df


parser = argparse.ArgumentParser(description='calculate relevance score')
parser.add_argument('-model_dir', help='folder contains the model file', type=str)
parser.add_argument('-model_file', help='name of the model file', type=str)
parser.add_argument('-pred_file', help='name of the pred_file for which we want to calculate R score', type=str)
parser.add_argument('-id2gene', help='file contains the ID:gene pairs', type=str)
parser.add_argument('-rule', help='which rule to use for calculating R scores', type=str, default='gamma-epsilon')
parser.add_argument('-factor1', help='rule factor 1', type=float, default=0)
parser.add_argument('-factor2', help='rule factor 2', type=float, default=1e-9)
parser.add_argument('-hidden_size', help='number of hidden neurons for each pathway module', type=int, default=6)

arg = parser.parse_args()
model_dir = arg.model_dir
model_file = arg.model_file
pred_file = arg.pred_file
rule = arg.rule
rule_factor1 = arg.factor1
rule_factor2 = arg.factor2
hidden_size = arg.hidden_size

id2gene_mapping = load_mapping(arg.id2gene)
# if models are trained on GPU, load model to cpu using map_location
loaded_model = torch.load(os.path.join(model_dir, model_file), map_location=lambda storage, location: storage)

ptw_id_map = {}
# store the id map for each pathway including AO
count = 0
for ly in loaded_model.ptw_layer_list:
	for ptw in ly:
		ptw_id_map[ptw] = count
		count = count + 1

pred_df = pd.read_csv(pred_file, index_col=0)
pred_data = ChemDatasetCV(pred_df)
pred_feature, pred_label = pred_data.x_data, pred_data.y_data.reshape(pred_data.n_samples, 1)

neuron_rele_df, module_rele_df, gene_rele_df, MIE_rele_df, frag_rele_df = LRP_individual(loaded_model, ptw_id_map,
																						id2gene_mapping,
																						pred_feature, rule,
																						rule_factor1, rule_factor2,
																						hidden_size)


