import pickle
import os
import time
import random
import math
import networkx as nx
import gurobipy as gp
from gurobipy import GRB
from itertools import islice
import numpy as np

def calculate_link_weights(nodes, eps_nodes, links, p_uv):
    """计算链路权重"""
    temp_G = nx.Graph()
    temp_G.add_nodes_from(nodes)
    temp_G.add_edges_from(links)
    degrees = dict(temp_G.degree())
    
    weighted_links = []
    for (u, v) in links:
        prob = p_uv.get((u, v), 1e-6)
        log_p = math.log(prob)
        u_is_eps = u in eps_nodes
        v_is_eps = v in eps_nodes
        
        weight = float('inf')
        if u_is_eps and not v_is_eps:
            weight = -1 * degrees[u] * log_p
        elif not u_is_eps and v_is_eps:
            weight = -1 * degrees[v] * log_p
        elif u_is_eps and v_is_eps:
            weight = min(-1 * degrees[u] * log_p, -1 * degrees[v] * log_p)
        else:
            weight = 99999.0
        weighted_links.append((u, v, weight))
    return weighted_links

def get_k_shortest_paths(nodes, eps_nodes, links, p_uv, demands, K=3):
    """生成候选路径集合 P_i"""
    weighted_edges = calculate_link_weights(nodes, eps_nodes, links, p_uv)
    G = nx.Graph()
    G.add_weighted_edges_from(weighted_edges)
    candidate_paths = {}
    
    for d_id, (src, dst) in demands.items():
        if src not in G or dst not in G:
            candidate_paths[d_id] = []
            continue
        try:
            path_generator = nx.shortest_simple_paths(G, source=src, target=dst, weight='weight')
            k_paths_nodes = list(islice(path_generator, K))
            
            k_paths_edges = []
            for path in k_paths_nodes:
                edges_in_path = []
                for i in range(len(path) - 1):
                    u, v = path[i], path[i+1]
                    if (u, v) in links: 
                        if (u,v) in links: edges_in_path.append((u, v))
                        else: edges_in_path.append((v, u)) 
                    else:
                        edges_in_path.append((u, v)) 
                k_paths_edges.append(edges_in_path)
            candidate_paths[d_id] = k_paths_edges
        except nx.NetworkXNoPath:
            candidate_paths[d_id] = []
    return candidate_paths

def run_epsa(nodes, eps_nodes, links, demands, candidate_paths, p_uv, verbose=False):
    """求解 LP 松弛问题"""
    # 创建环境并关闭输出，避免刷屏
    try:
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()
        model = gp.Model("EPSA", env=env)
    except gp.GurobiError:
        # Fallback if env setup fails
        model = gp.Model("EPSA")
        model.setParam("OutputFlag", 0)
    
    T = model.addVar(vtype=GRB.CONTINUOUS, lb=0, name="T")
    e = {}
    for d_id in demands:
        num_paths = len(candidate_paths[d_id])
        for k in range(num_paths):
            e[d_id, k] = model.addVar(vtype=GRB.CONTINUOUS, lb=0, ub=1, name=f"e_{d_id}_{k}")

    t_vars = {}
    link_set = set()
    for l in links:
        u, v = l
        link_set.add((u,v))
        t_vars[u, v] = model.addVar(vtype=GRB.CONTINUOUS, lb=0)
        t_vars[v, u] = model.addVar(vtype=GRB.CONTINUOUS, lb=0)

    for d_id in demands:
        model.addConstr(gp.quicksum(e[d_id, k] for k in range(len(candidate_paths[d_id]))) == 1)

    m_exprs = {}
    for link in link_set:
        m_exprs[link] = 0
        m_exprs[(link[1], link[0])] = 0 

    for d_id in demands:
        for k, path in enumerate(candidate_paths[d_id]):
            for (u, v) in path:
                if (u, v) in m_exprs: m_exprs[u, v] += e[d_id, k]
                elif (v, u) in m_exprs: m_exprs[v, u] += e[d_id, k]

    for (u, v) in link_set:
        prob_uv = p_uv.get((u, v), 0)
        prob_vu = p_uv.get((v, u), 0)
        demand_load = m_exprs.get((u, v), 0)
        model.addConstr(t_vars[u, v] * prob_uv + t_vars[v, u] * prob_vu >= demand_load)

    for u in eps_nodes:
        adjacent_links = []
        for (x, y) in link_set:
            if x == u: adjacent_links.append((x, y))
            elif y == u: adjacent_links.append((y, x))
        model.addConstr(gp.quicksum(t_vars[curr[0], curr[1]] for curr in adjacent_links) <= T)

    non_eps_nodes = set(nodes) - set(eps_nodes)
    for u in non_eps_nodes:
        adjacent_links = []
        for (x, y) in link_set:
            if x == u: adjacent_links.append((x, y))
            elif y == u: adjacent_links.append((y, x))
        for (u_curr, v_curr) in adjacent_links:
            model.addConstr(t_vars[u_curr, v_curr] == 0)

    model.setObjective(T, GRB.MINIMIZE)
    model.optimize()

    selected_paths = {}
    if model.status == GRB.OPTIMAL:
        for d_id in demands:
            best_k = -1
            max_val = -1.0
            num_paths = len(candidate_paths[d_id])
            for k in range(num_paths):
                val = e[d_id, k].X
                if val > max_val:
                    max_val = val
                    best_k = k
            selected_paths[d_id] = best_k
        return selected_paths, T.X
    else:
        return None, None
