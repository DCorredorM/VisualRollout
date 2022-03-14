import networkx as nx
import numpy as np

from copy import copy
import matplotlib.pyplot as plt

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


def tree_layout(graph: nx.DiGraph):
    
    root = list([n for n in graph.nodes if len(list(graph.predecessors(n))) == 0])[0]
    layers = [[(None, [root])]]  # [(papa, hijos)]
    
    def nodes_in_layer(layer):
        return set().union(*[l[1] for l in layer])
    
    def len_layer(layer):
        return len(set().union(*[l[1] for l in layer]))
    
    pos = dict()
    
    idx = 0
    while True:
        layer = []
        for node_group in layers[idx]:
            for parent in node_group[1]:
                layer.append((parent, list(graph.successors(parent))))
        
        if all([len(l[1]) == 0 for l in layer]):
            break
        
        layers.append(layer)
        idx += 1
    
    max_layer = max(layers, key=len_layer)
    
    max_layer_idx = layers.index(max_layer)
    x = 0
    for (parent, sons) in max_layer:
        for son in sons:
            pos[son] = (x, -max_layer_idx)
            x += 1
    
    for y in range(max_layer_idx + 1, len(layers)):
        layer = layers[y]
        for node in nodes_in_layer(layer):
            parents = list(graph.predecessors(node))
            x = sum(pos[p][0] for p in parents) / len(parents)
            pos[node] = (x, -y)
    
    for y in list(range(0, max_layer_idx))[::-1]:
        layer = layers[y]
        for node in nodes_in_layer(layer):
            parents = list(graph.successors(node))
            x = sum(pos[p][0] for p in parents) / len(parents)
            pos[node] = (x, -y)
        
    
    
    return pos


def nudge(pos, x_shift: float = 0.0, y_shift: float = 0.0):
    return {n: (x + x_shift, y + y_shift) for n, (x, y) in pos.items()}


class TSPGraph:
    def __init__(self, distance_matrix: np.ndarray):
        self.distance_matrix = distance_matrix
        self.n = len(distance_matrix)
        
        self.pos = {}
        
        self.cities = [letters[i] for i in range(self.n)]
        
        self.tree = nx.DiGraph()
        self.tree.add_node(self.cities[0])
        
        self.current_path = [self.cities[0]]
        self.current_cost = 0
        self.heuristic_runs = []
        
        self.total_reward = float('inf')
        
        self._temp_path = []
        self._temp_reward = float('inf')
        
        self.max_depth = 1
    
    def fill_sons(self, node, recursively=False):
        missing = set(self.cities).difference(set(list(node)))
        sons = []
        if missing:
            for m in missing:
                missing_m = copy(missing)
                missing_m.remove(m)
                i, j = self.cities.index(node[-1]), self.cities.index(m)
                w = self.distance_matrix[i, j]
                sons.append(node + m)
                self.tree.add_edge(node, node + m, w=w)
                if recursively:
                    self.fill_sons(node + m, recursively)
        
        else:
            m = node[0]
            i, j = self.cities.index(node[-1]), self.cities.index(m)
            w = self.distance_matrix[i, j]
            self.tree.add_edge(node, 'f' + m, w=w)
            sons.append('f' + m)
        
        if recursively is False:
            return sons
    
    def create_dp_tree(self):
        self.fill_sons(self.cities[0], True)
    
    def visualize_tree(self, full=False, explored_by_rollout=True, normal_nodes=None):
        missing_nodes = set(self.tree.nodes)
        
        heuristic_runs = self.heuristic_runs
        
        current_path = self.current_path
        
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.axis('off')
        if full is True:
            self.create_dp_tree()
            
            self.pos = tree_layout(self.tree)
            pos = self.pos
            
            nx.draw_networkx_nodes(self.tree, pos, node_color='white', edgecolors='black', ax=ax)
            nx.draw_networkx_labels(self.tree, pos, labels={n: n[-1] for n in self.tree.nodes}, font_size=10, ax=ax)
            
            nx.draw_networkx_edges(self.tree, pos)
            nx.draw_networkx_edge_labels(
                self.tree,
                pos,
                edge_labels={e: self.tree[e[0]][e[1]]['w'] for e in self.tree.edges},
                font_size=8,
                clip_on=False,
                rotate=False
            )
            missing_nodes = set()

        # self.pos = graphviz_layout(self.tree, prog='dot')
        self.pos = tree_layout(self.tree)
        pos = self.pos
        
        # draw nodes
        # draw heuristic_runs
        for path, value in heuristic_runs:
            missing_nodes.difference_update(set(path))
            self.plot_heuristic(path, value, explored_by_rollout, ax=ax, node_color='tab:blue', alpha=0.3)
        
        # current_path
        nx.draw_networkx_nodes(
            self.tree, pos, nodelist=current_path, node_color='tab:orange', edgecolors='black', ax=ax
        )
        missing_nodes.difference_update(set(current_path))
        x_, y_ = tuple(zip(*list(pos.values())))
        x_range = max(x_) - min(x_)
        y_range = max(y_) - min(y_)
        
        nx.draw_networkx_labels(self.tree, nudge(pos, (x_range + 0.008) / 100, -(y_range + 0.01) / 100),
                                labels={current_path[-1]: self.current_cost},
                                ax=ax,
                                horizontalalignment='left',
                                verticalalignment='top')
        nx.draw_networkx_labels(self.tree, pos, labels={n: n[-1] for n in current_path}, font_size=10, ax=ax)
        edgelist = list(zip(current_path[:-1], current_path[1:]))
        nx.draw_networkx_edges(self.tree, pos, edgelist=edgelist)
        nx.draw_networkx_edge_labels(
            self.tree,
            pos,
            edge_labels={e: self.tree[e[0]][e[1]]['w'] for e in edgelist},
            font_size=8,
            clip_on=False,
            rotate=False
        )
        
        if normal_nodes is None:
            normal_nodes = missing_nodes
        
        nx.draw_networkx_nodes(self.tree, pos, nodelist=normal_nodes, node_color='white', edgecolors='black', ax=ax)
        nx.draw_networkx_labels(self.tree, pos, labels={n: n[-1] for n in normal_nodes}, font_size=10, ax=ax)
        
        edgelist = sum(
            [
                [(i, n) for i in self.tree.predecessors(n)] +
                [(n, i) for i in self.tree.successors(n)]
                for n in normal_nodes],
            []
        )
        nx.draw_networkx_edges(self.tree, pos, edgelist=edgelist)
        nx.draw_networkx_edge_labels(
            self.tree,
            pos,
            edge_labels={e: self.tree[e[0]][e[1]]['w'] for e in edgelist},
            font_size=8,
            clip_on=False,
            rotate=False
        )
        plt.show()
    
    def run_heuristic(self, node):
        def walk(node, cost):
            missing = set(self.cities).difference(set(list(node)))
            if node[0] == 'f':
                return [], cost
            
            elif missing:
                i = self.cities.index(node[-1])
                m = min(missing, key=lambda x: self.distance_matrix[i, self.cities.index(x)])
                j = self.cities.index(m)
                
                w = self.distance_matrix[i, j]
                self.tree.add_edge(node, node + m, w=w)
                
                f_path, f_cost = walk(node + m, cost + w)
                
                return [node] + f_path, f_cost
            
            else:
                m = node[0]
                i, j = self.cities.index(node[-1]), self.cities.index(m)
                w = self.distance_matrix[i, j]
                self.tree.add_edge(node, 'f' + m, w=w)
                
                return [node, 'f' + m], cost + w
        
        sol = walk(node, 0)
        if sol[0]:
            self.heuristic_runs.append(sol)
        return sol
    
    def plot_heuristic(self, path, value, explored_by_rollout, **kwargs):
        pos = self.pos
        ax = kwargs.pop('ax', None)
        
        x_, y_ = tuple(zip(*list(pos.values())))
        x_range = max(x_) - min(x_)
        
        nx.draw_networkx_labels(self.tree, nudge(pos, -(x_range + 0.008) / 100), labels={path[0]: value}, ax=ax,
                                horizontalalignment='right',
                                verticalalignment='bottom')
        if explored_by_rollout:
            # paint only first node
            nx.draw_networkx_nodes(self.tree, pos, nodelist=[path[0]], ax=ax, **kwargs)
            nx.draw_networkx_labels(self.tree, pos, labels={n: n[-1] for n in [path[0]]}, font_size=10, ax=ax)
            # draw edges
            edgelist = []
        else:
            # paint whole path
            nx.draw_networkx_nodes(self.tree, pos, nodelist=path, ax=ax, **kwargs)
            nx.draw_networkx_labels(self.tree, pos, labels={n: n[-1] for n in path}, font_size=10, ax=ax)
            edgelist = list(zip(path[:-1], path[1:]))
        if path[0][:-1] in self.tree.nodes:
            edgelist += [(path[0][:-1], path[0])]
        
        nx.draw_networkx_edges(self.tree, pos, edgelist=edgelist)
        nx.draw_networkx_edge_labels(
            self.tree,
            pos,
            edge_labels={e: self.tree[e[0]][e[1]]['w'] for e in edgelist},
            font_size=8,
            clip_on=False,
            rotate=False
        )
    
    def add_node_path(self, node):
        self.current_path.append(node)
    
    def roll_out(self, state=None, lookahead=1):
        if state is None:
            state = self.current_path[-1]
        self.max_depth = lookahead
        self._temp_path = [state]
        self._temp_reward = float('inf')
        
        self.roll_out_search(self._temp_path, 0, 0)
        
        self.current_path.append(self._temp_path[1])
        self.current_cost += self.tree[state][self._temp_path[1]]['w']
    
    def __len__(self):
        return self.n
    
    def roll_out_search(self, rollout_path, rollout_reward, depth):
        if depth >= self.max_depth or rollout_path[-1][0] == 'f':
            self.update_temporary_search(rollout_path, rollout_reward)
        else:
            for son in self.fill_sons(rollout_path[-1]):
                new_path = rollout_path + [son]
                new_reward = rollout_reward + self.tree[rollout_path[-1]][son]['w']
                self.roll_out_search(rollout_path=new_path, rollout_reward=new_reward, depth=depth + 1)
    
    def update_temporary_search(self, temp_path, temp_reward):
        """
        Updates the online exploration incumbent path and reward.

        Parameters
        ----------
        temp_path:
            The current temporary path. A sequence of state and action paris that indicates the path used.
        temp_reward:
            Temporary number of colors used.
        """
        cur_state = temp_path[-1]
        _, heuristic_reward = self.run_heuristic(cur_state)
        
        if temp_reward + heuristic_reward < self._temp_reward:
            self._temp_reward = temp_reward + heuristic_reward
            self._temp_path = temp_path
        
        if self._temp_reward < self.total_reward:
            self.total_reward = self._temp_reward
