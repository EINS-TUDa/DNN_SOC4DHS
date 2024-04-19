"""
Author: Andreas Bott
year: 2024
mail: andreas.bott@eins.tu-darmstadt.de or andi-bott@web.de

This module contains helper functions to parse the grid Json and to set up the different elements used in the main script
"""

import itertools
import json
import re

import networkx as nx
import numpy as np
import pandas as pd
import tensorflow as tf

import lib_dnn_soc.DNN_lib as NN
from lib_dnn_soc.steady_state_modelling.state_equations import StateEquations
from lib_dnn_soc.utility import ZeroTruncatedMultivariateNormal
from lib_dnn_soc.utility.utility import GridModel, Uniform


def _parse_gridfile(gridfile):
    """
    Parse grid file (json format) and extract the following information:
        pp_ind: dictionary mapping the PP names to their index in the covariance matrix
        act_edg_ind: dictionary mapping the DEM names to their index in the covariance matrix
        active_edge: dictionary mapping the edge names to their index in the covariance matrix
        h_key: name of the slack power plant
    """
    pp_ind = dict()
    dem_ind = dict()
    active_edge = dict()
    dem_ind_list = []
    with open(gridfile, 'r') as gf:
        for l in gf:
            if 'edge control type": "active' in l:
                if 'PP' in l:
                    ind = re.findall('PP\w+', l)[0]
                    pp_ind[ind] = int(ind[-1])
                if 'DEM' in l:
                    ind = re.findall('DEM\w+', l)[0]
                    dem_ind[ind] = int(ind[-1])
                    dem_ind_list.append(int(ind[-1]))
                # active_edge counts the position of each demand in the covariance matrix
                if len(pp_ind) == 0:
                    active_edge[ind] = int(ind[-1])
                else:
                    active_edge[ind] = int(ind[-1]) - 1
    h_key = list(pp_ind.keys())[0]
    active_edge = {key: active_edge[key] for key in active_edge.keys()}
    return pp_ind, dem_ind, active_edge, h_key


def _calculate_d_prior_dist(dem_ind, dem_mean, dem_std, d_prior_type, dem_cor_factor=None, custom_cor=None, **kwargs):
    """
    calculates the prior distribution over the uncertain heat powers
    :param dem_ind: dictionary mapping the DEM names to their index in the covariance matrix
    :param dem_mean: float or dict: mean power values
    :param dem_std: float or dict: standard deviation relative to the mean power
    :param dem_cor_factor: float: correlation strength between the different demands
    :param d_prior_type: string: type of prior distribution
    :param custom_cor: np.array: covariance matrix -> used if d_prior_type == 'custom'
    :return: d_prior_dist: distribution over uncertain heat powers
    """
    # convert dem_mean and dem_std to dictionaries if necessary:
    if type(dem_mean) is not dict:
        dem_mean = {k: dem_mean for k in dem_ind.keys()}
    if type(dem_std) is not dict:
        dem_std = {k: dem_std for k in dem_ind.keys()}

    if d_prior_type == 'ind':        # no dependency
        cor = np.eye(len(dem_ind))
    elif d_prior_type == 'dist':       # distance based correlation
        dist = np.array([[abs(i - j) for i in dem_ind.values()] for j in dem_ind.values()])
        cor = np.exp(-dem_cor_factor * dist / np.max(dist))
    elif d_prior_type == 'custom':
        cor = custom_cor
    else:
        raise ValueError(f'unknown d_prior_type: {d_prior_type}')

    cov = np.eye(len(dem_ind))
    for (i, dem_i), (j, dem_j) in itertools.product(enumerate(dem_ind.keys()), (enumerate(dem_ind.keys()))):
        cov[i, j] = cor[i, j] * dem_std[dem_i] * dem_std[dem_j] * dem_mean[dem_i] * dem_mean[dem_j]
    m = np.array([dem_mean[dem] for dem in dem_ind.keys()])

    return ZeroTruncatedMultivariateNormal(loc=tf.constant(m, dtype=tf.float64),
                                           scale_tril=tf.linalg.cholesky(cov),
                                           validate_args=True, name='prior_demand_distribution')


def _calculate_T_prior_dist(act_edg_ind, dem_T_min, dem_T_max, heat_ind=None, heat_T_min=None, heat_T_max=None,
                            include_heating=False, slack=None, **kwargs):
    """
    calculates the prior distribution over the uncertain temperatures
    :param act_edg_ind: dictionary mapping the DEM names to their index in the covariance matrix
    :param dem_T_min: float or dict: minimum temperature
    :param dem_T_max: float or dict: maximum temperature
    :return: T_prior_dist: distribution over uncertain temperatures
    """

    heat_ind = dict() if heat_ind is None else heat_ind
    heat_T_min = {k: heat_T_min for k in heat_ind.keys()} if type(heat_T_min) is not dict else heat_T_min
    heat_T_max = {k: heat_T_max for k in heat_ind.keys()} if type(heat_T_max) is not dict else heat_T_max
    dem_T_min = {k: dem_T_min for k in act_edg_ind.keys() if not k in heat_ind.keys()} if type(dem_T_min) is not dict else dem_T_min
    dem_T_max = {k: dem_T_max for k in act_edg_ind.keys()if not k in heat_ind.keys()} if type(dem_T_max) is not dict else dem_T_max

    n_edg = len(act_edg_ind) if not include_heating else len(act_edg_ind) + 1
    tmin = np.zeros(n_edg)
    tmax = np.zeros(n_edg)
    for i, key in enumerate(act_edg_ind.keys()):
        if key in dem_T_min.keys():
            tmin[i] = dem_T_min[key]
            tmax[i] = dem_T_max[key]
        elif key in heat_ind.keys():
            tmin[i] = heat_T_min[key]
            tmax[i] = heat_T_max[key]
        else:
            raise ValueError(f'unknown edge: {key}')
    if include_heating and slack is not None:
        values = list(slack.values())[0]
        tmin[-1] = values['T_min']
        tmax[-1] = values['T_max']

    return Uniform(low=tmin, high=tmax)


def _get_powers(active_edges, dem_ind, heat_ind, dem_mean, dem_T_min, dem_T_max, heat_powers, heat_T_min, heat_T_max,
                **kwargs):
    """
    parses the supply and demand parameters in a specific way requested by the SE object
    """
    # convert everything to dictionaries if necessary:
    dictlike = lambda x, whitelist, blacklist=dict(): {k: x for k in whitelist.keys() if not k in blacklist.keys()}
    dem_mean = dictlike(dem_mean, dem_ind) if type(dem_mean) is not dict else dem_mean
    dem_T_min = dictlike(dem_T_min, dem_ind) if type(dem_T_min) is not dict else dem_T_min
    dem_T_max = dictlike(dem_T_max, dem_ind) if type(dem_T_max) is not dict else dem_T_max
    if heat_powers == 'mean':
        heat_powers = -1 * sum(dem_mean[k] for k in dem_ind.keys()) / (len(active_edges) - len(dem_ind))
    heat_powers = dictlike(heat_powers, active_edges, blacklist=dem_ind) if type(heat_powers) is not dict else heat_powers
    heat_T_min = dictlike(heat_T_min, active_edges, blacklist=dem_ind) if type(heat_T_min) is not dict else heat_T_min
    heat_T_max = dictlike(heat_T_max, active_edges, blacklist=dem_ind) if type(heat_T_max) is not dict else heat_T_max

    demands = dict()
    heatings = dict()
    for key in active_edges.keys():
        if key in dem_ind.keys():
            demands[key] = {'Power': dem_mean[key], 'Temperature': (dem_T_min[key] + dem_T_max[key]) / 2,
                            'T_min': dem_T_min[key], 'T_max': dem_T_max[key]}
        elif key == heat_ind:
            heatings[key] = {'Power': 0, 'Temperature': (heat_T_min[key] + heat_T_max[key]) / 2,
                            'T_min': heat_T_min[key], 'T_max': heat_T_max[key]}
        else:
            demands[key] = {'Power': heat_powers[key], 'Temperature': (heat_T_min[key] + heat_T_max[key]) / 2,
                            'T_min': heat_T_min[key], 'T_max': heat_T_max[key]}

    return demands, heatings


def _setup_SE(grid_file, active_edges, dem_ind, heat_ind, Ta, fix_dp, **grid_specification):
    """
    setup the State-Equations object
    :param grid_file:  path to the grid file (json format)
    :param Ta:  ambient temperature
    :param fix_dp: pressure values for fixed pressure point (i.e. slack generator); dict or 'auto'
    :param active_edges: dict containing all active edges and the position in the power vector
    :param dem_ind: dict containing all demands and the position in the power vector
    :param heat_ind: index of the slack generator
    :param grid_specification: dict containing the grid specifications -> passed to _get_powers
    :return: SE object
    """
    def loadGridModel(jsonPaths):
        """
        Loads district heating network data from json-file to  a MeFlexWärme GridModel
        @author: Friedrich
        """
        # load json-file. Note: Path must be given as raw-string: r"<Path>"
        nd = json.load(open(jsonPaths, "r"))
        nd = [el for el in nd if 'type' in el.keys()]  # Filter for elements that have the field 'type'
        nodes = [(nd[i]['index'], nd[i]) for i in range(len(nd)) if nd[i]['type'] == 'Node']
        edges = [(nd[i]['from'], nd[i]['to'], nd[i]) for i in range(len(nd)) if nd[i]['type'] == 'Edge']

        grid = GridModel()
        grid.add_nodes_from(nodes)
        grid.add_edges_from(edges)
        return grid

    def find_all_cycles(graph_input):
        g = nx.DiGraph(graph_input)
        cycles = []
        while True:
            try:
                cycle = nx.algorithms.cycles.find_cycle(g, orientation='ignore')
                cycles.append(cycle)
            except nx.exception.NetworkXNoCycle:  # exception appears if no cycle is found
                break
            for u, v, direction in cycle:
                g.remove_edge(u, v)
        return cycles

    grid = loadGridModel(grid_file)

    # identify all cycles in the passive parts of the grid (looping pipes)
    sup_graph = nx.DiGraph(grid.subgraph((node for node, data in grid.nodes(data=True) if data['nw_section'] == 'Sup')))
    ret_graph = nx.DiGraph(grid.subgraph((node for node, data in grid.nodes(data=True) if data['nw_section'] == 'Ret')))

    cycles = []
    cycles.extend(find_all_cycles(sup_graph))
    cycles.extend(find_all_cycles(ret_graph))
    if not cycles == []:
        cycles = [[(grid.edges[(e[0], e[1])]['index'], e[2]) for e in c] for c in cycles]

    # pipe parameters:
    pipes_db = pd.read_excel('./grids/Pipe_Parameters.xlsx', index_col=0, engine='openpyxl')

    # store nodes and edges in list format for easier access in the SE object:
    nodes = []
    edges = []

    for node_idx in grid.nodes():
        nodes.append(grid.nodes[node_idx])

    for edge_idx in grid.edges():
        edge_vals = grid.edges[edge_idx]
        if edge_vals['edge control type'] == 'passive':
            edge_vals['temp_loss_coeff'] = pipes_db.loc[edge_vals['Isoplus-ID'], 'Norm-Wärme-übergangskoeffizent']
            edge_vals['diameter'] = pipes_db.loc[edge_vals['Isoplus-ID'], 'hydraulischer Durchmesser']
            if edge_vals['nw_section'] == 'sup':
                edge_vals['fd_nom'] = pipes_db.loc[edge_vals['Isoplus-ID'], 'fd_nom 110C']
            else:
                edge_vals['fd_nom'] = pipes_db.loc[edge_vals['Isoplus-ID'], 'fd_nom 65C']
            edge_vals['bend_factor'] = edge_vals.get('pressure loss factor[-]', 1)
        # edges contains all active and passive edges
        edges.append(edge_vals)

    # setup demands and heatings:
    demands, heatings = _get_powers(active_edges, dem_ind, heat_ind, **grid_specification)
    fix_dp = [{e['from']: 3.5, e['to']: 6.5} for e in edges if e['index'] == heat_ind][0] if not type (fix_dp) is dict else fix_dp
    # construct state equations object
    SE = StateEquations(edges=edges, nodes=nodes, demands=demands, heatings=heatings, fix_dp=fix_dp, Ta=Ta)
    SE.set_init_state()
    return SE, cycles, grid


def random_input_generator(how='load', start=0, file=None, dem_sel=None, d_prior_dist=None, dhs_sched=None, StateSimulator=None):
    """
        this function is a wrapper around the import_training_data function and yields the data in a generator
        alternatively it can be used to generate random data based on the prior distributions and the state simulator

        how: string, either "load" or "random" if load: the data is loaded from a file, if random: the data is generated
        how=='load' :parameter
            start: int optional, index of first sample to yield
            file: string, name of the file the data is read in from
            dem_sel: list of integers optional, indices of the demands to be selected
        how=='random' :parameter
            d_prior_dist: tfp.distributions object, prior distribution for the demand
            dhs_sched: dictionary, schedule for the heat grid
            StateSimulator: object, state simulator for the heat grid
    """

    if how == 'load':
        assert file is not None, 'if how == "load", "file" has to be specified'
        if dem_sel is None:
            d, t, s = NN.import_training_data(1, file, f_id0=0, skip=0)
            dem_sel = list(range(d.shape[1]))
        count = start
        while True:
            d, t, s = NN.import_training_data(1, file, f_id0=0, skip=count)
            yield tf.gather(d, dem_sel, axis=1), t, tf.transpose(s)
            count += 1
    elif how == 'random':
        assert d_prior_dist is not None and dhs_sched is not None and StateSimulator is not None, \
            'd_prior_dist, dhs_sched and StateSimulator have to be specified if how is "random"'
        while True:
            d = d_prior_dist.sample(1)
            t = tf.concat([dhs_sched['T_d'], dhs_sched['T_q']], axis=1)
            s = StateSimulator.get_state(**{**dhs_sched, 'd': d})
            yield d, t, s
    else:
        raise ValueError('how has to be either "load" or "random"')