import networkx as nx
import numpy as np

def generate_tsp_instance(num_cities):
    G_normal = nx.complete_graph(num_cities)
    G_modified = nx.complete_graph(num_cities)
    pos = {i: (np.random.rand(), np.random.rand()) for i in range(num_cities)}
    
    for i in range(num_cities):
        G_normal.nodes[i]['pos'] = pos[i]
        G_modified.nodes[i]['pos'] = pos[i]

    for u, v in G_normal.edges():
        base_distance = np.linalg.norm(np.array(pos[u]) - np.array(pos[v]))
        G_normal.edges[u, v]['weight'] = base_distance
        # PERTURBATIONS ON COSTS: adding random noise
        distance_modifier = np.random.uniform(0.7, 1.3)  # Example: 15% variation
        G_modified.edges[u, v]['weight'] = base_distance * distance_modifier

    return G_normal, G_modified


def generate_heuristic_solution(tsp_instance):
    path = nx.approximation.greedy_tsp(tsp_instance, source=0)

    path = path[:-1] if path[-1] == path[0] else path
    cost = sum(tsp_instance[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))
    return path, cost

def load_heuristic(G):
    pos = nx.get_node_attributes(G, 'pos')
    cities = []

    for node_id, (x, y) in pos.items():
        lat = (x - 0.5) * 180
        lon = (y - 0.5) * 360
        cities.append([node_id, lat, lon])
    
    path = []
    total_cost = 0
    unvisited = list(G.nodes())
    current = unvisited.pop(0)
    path.append(current)

    while unvisited:
        next_node = min(unvisited, key=lambda node: G[current][node]['weight'])
        total_cost += G[current][next_node]['weight']
        current = next_node
        path.append(current)
        unvisited.remove(current)

    # closes tour
    total_cost += G[path[-1]][path[0]]['weight']

    return cities, total_cost

def save_tsp_instance(G, path):
    nx.write_weighted_edgelist(G, path)

def load_tsp_instance(path):
    return nx.read_weighted_edgelist(path, nodetype=int)
