import sys
import networkx as nx
import networkx.algorithms.components.connected as nxacc
import networkx.algorithms.dag as nxadag
import pandas as pd


def load_mapping(mapping_file):
    """
    :param mapping_file: the filename of the csv containing the ID_number and gene_symbol in each line
    :return: a dict storing the {gene_symbol : ID_number} mapping
    """
    df = pd.read_csv(mapping_file)

    mapping = dict(zip(df['gene'], df['ID']))

    return mapping


def load_reactome_ori(file_name, gene2id_mapping):
    """
    :param file_name: the name of the csv file containing the reactome pathway relationship:
                    1) pathway, pathway, default - a pathway ID and its child or parent pathway
                    2) pathway, gene_symbol, gene - a pathway and its directly gene annotated gene
    :param gene2id_mapping: dictionary storing the {gene_symbol : ID_number} mapping
    :return:
            dG: directed graph that map the hierarchy structure of DILI reactome database
            dG_gene: directed graph that map the hierarchy structure of DILI reactome database including gene nodes
            leaves[0]: the root of the hierarchy, should have only one root
            ptw_size_map: for each pathway, record the size of directly annotated gene set
            ptw_direct_gene_map: for each pathway, store its direct annotated gene set
            ptw_child_dict: for each pathway, store its children pathway
    """
    dG = nx.DiGraph()
    dG_gene = nx.DiGraph()
    ptw_direct_gene_map = {}
    ptw_size_map = {}
    ptw_child_dict = {}

    file_handle = pd.read_csv(file_name, index_col=0)
    gene_set = set()
    for index, row in file_handle.iterrows():
        dG_gene.add_edge(row['Parent'], row['Child'])
        if row['Note'] == 'pathway':
            # add edge parent -> child
            dG.add_edge(row['Parent'], row['Child'])
        elif row['Note'] == 'gene':
            # pathway-gene relation
            if row['Child'] not in gene2id_mapping:
                print(row['Child'])
            if row['Parent'] not in ptw_direct_gene_map:
                ptw_direct_gene_map[row['Parent']] = set()
            ptw_direct_gene_map[row['Parent']].add(gene2id_mapping[row['Child']])
            gene_set.add(row['Child'])

    print('There are total', len(gene_set), 'genes')
    
    # generate ptw_size_map for the customized graph      
    for ptw in dG.nodes():
        # for pathways have annotated genes, get pathway related gene set size
        if ptw in ptw_direct_gene_map:
            ptw_gene_set = ptw_direct_gene_map[ptw]
            ptw_size_map[ptw] = len(ptw_gene_set)
        # for other pathways, get how many child pathways it has
        else:    
            # get the list of child pathways
            deslist = nxadag.descendants(dG, ptw)
            ptw_size_map[ptw] = len(deslist)
            
            
    # get the root nodes, from the Reactome
    leaves = [n for n in dG.nodes if dG.in_degree(n) == 0]

    uG = dG.to_undirected()
    connected_subG_list = list(nxacc.connected_components(uG))

    print('There are', len(leaves), 'roots:', leaves)
    print('There are', len(dG.nodes()), 'pathways')
    print('There are', len(connected_subG_list), 'connected componenets')

    return dG, dG_gene, leaves, ptw_size_map, ptw_direct_gene_map