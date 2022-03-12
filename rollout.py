import networkx as nx
import numpy as np
from copy import copy

import time
from networkx.drawing.nx_pydot import graphviz_layout
import ipywidgets as widgets

import matplotlib.pyplot as plt

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


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
        self.pos = graphviz_layout(self.tree, prog='dot')
        
        pos = self.pos
        fig, ax = plt.subplots(figsize=(12, 7))
        ax.axis('off')
        if full is True:
            self.create_dp_tree()
            
            self.pos = graphviz_layout(self.tree, prog='dot')
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


class GUI:
    def __init__(self):
        self.c_number_of_nodes: widgets.IntSlider
        self.c_current_sons_b: widgets.Button
        self.c_run_heuristic_b: widgets.Button
        self.c_run_heuristic_dd: widgets.Dropdown
        self.c_run_heuristic: widgets.HBox
        self.c_run_step_rollout: widgets.Button
        self.c_walk_to_b: widgets.Button
        self.c_walk_to_dd: widgets.Dropdown
        self.c_walk_to: widgets.HBox
        self.c_plot_H_path: widgets.ToggleButton
        self.c_full_simulation: widgets.Button
        self.c_plot_whole_tree: widgets.Button
        self.out: widgets.Output
        
        self.build_widgets()
        
        self.graph = self._update_graph(self.c_number_of_nodes.value)
        
        self.link_widgets()
    
    # noinspection PyAttributeOutsideInit
    def build_widgets(self):
        self.c_number_of_nodes = widgets.IntSlider(
            value=5,
            min=3,
            max=len(letters),
            step=1,
            description='# Cities:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        
        self.c_current_sons_b = widgets.Button(
            description='Create Sons',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Create Sons',
            icon=''  # (FontAwesome names without the `fa-` prefix)
        )
        self.c_lookahead = widgets.IntSlider(
            value=1,
            min=1,
            max=3,
            step=1,
            description='Lookahead:',
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='d'
        )
        
        self.c_sons_lookahead = widgets.VBox([self.c_current_sons_b, self.c_lookahead])
        
        self.c_run_heuristic_b = widgets.Button(
            description='Run Heuristic',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Run Heuristic',
            icon=''  # (FontAwesome names without the `fa-` prefix)
        )
        
        self.c_run_heuristic_dd = widgets.Dropdown(
            options=['All', 'A'],
            value='All'
        )
        self.c_run_heuristic = widgets.HBox(children=[self.c_run_heuristic_b, self.c_run_heuristic_dd])
        
        self.c_run_step_rollout = widgets.Button(
            description='Run Step Rollout',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Run Step Rollout',
            icon=''  # (FontAwesome names without the `fa-` prefix)
        )
        
        self.c_walk_to_b = widgets.Button(
            description='Walk to node',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Update current path',
            icon=''  # (FontAwesome names without the `fa-` prefix)
        )
        
        self.c_walk_to_dd = widgets.Dropdown(options=[])
        
        self.c_walk_to = widgets.HBox(children=[self.c_walk_to_b, self.c_walk_to_dd])
        
        self.c_plot_H_path = widgets.ToggleButton(
            value=False,
            description='Plot Heuristics Path',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Plot Heuristics Path',
            icon='check'  # (FontAwesome names without the `fa-` prefix)
        )
        
        self.c_full_simulation = widgets.Button(
            description='Run Full Rollout Algorithm',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Run Full Rollout Algorithm',
            icon=''  # (FontAwesome names without the `fa-` prefix)
        )
        
        self.c_plot_whole_tree = widgets.Button(
            description='Plot Whole Tree',
            disabled=False,
            button_style='',  # 'success', 'info', 'warning', 'danger' or ''
            tooltip='Plots the whole DP tree',
            icon=''  # (FontAwesome names without the `fa-` prefix)
        )
        
        self.out = widgets.Output()
    
    def link_widgets(self):
        self.c_number_of_nodes.observe(self.on_number_of_nodes, names='value')
        
        self.c_current_sons_b.on_click(self.on_create_sons)
        
        self.c_run_heuristic_b.on_click(self.on_run_heuristics)
        
        self.c_run_step_rollout.on_click(self.on_run_step_rollout)
        
        self.c_walk_to_b.on_click(self.on_walk_to)
        
        self.c_plot_H_path.observe(lambda change: self.update_image())
        
        self.c_full_simulation.on_click(self.on_c_full_simulation)
        
        self.c_plot_whole_tree.on_click(self.on_plot_whole_tree)
    
    def reset_widgets(self):
        self.c_current_sons_b.disabled = False
        
        self.c_run_heuristic_b.disabled = False
        
        self.c_run_heuristic_dd.options = ['All', 'A']
        self.c_run_heuristic_dd.value = 'All'
        
        self.c_run_step_rollout.disabled = False
        
        self.c_walk_to_b.disabled = False
        
        self.c_walk_to_dd.options = []
        
        self.c_plot_H_path.value = False
        self.c_plot_H_path.disabled = False
        
        self.c_full_simulation.disabled = False
    
    @staticmethod
    def _update_graph(n):
        np.random.seed(754)
        distance_matrix = np.random.randint(2, 20, (n, n))
        
        return TSPGraph(distance_matrix)
    
    def display(self):
        out = widgets.VBox([
            widgets.HBox([
                widgets.VBox([
                    self.c_number_of_nodes, self.c_sons_lookahead, self.c_run_heuristic, self.c_run_step_rollout
                ]),
                widgets.VBox([
                    self.c_walk_to, self.c_plot_H_path, self.c_full_simulation, self.c_plot_whole_tree
                ])
            ]),
            self.out]
        )
        self.update_image()
        return out
    
    def update_image(self, full=False, normal_nodes=None):
        if self.graph.current_path[-1][0] == 'f':
            self.c_current_sons_b.disabled = True
            self.c_walk_to_b.disabled = True
            self.c_full_simulation.disabled = True
            self.c_run_step_rollout.disabled = True
        
        explored_by_rollout = self.c_plot_H_path.value
        
        self.update_heuristic_nodes()
        
        with self.out:
            self.out.clear_output(True)
            self.graph.visualize_tree(full=full, explored_by_rollout=explored_by_rollout, normal_nodes=normal_nodes)
    
    def on_number_of_nodes(self, b):
        self.graph = self._update_graph(b['new'])
        if b['new'] > 5:
            self.c_plot_whole_tree.disabled = True
        else:
            self.c_plot_whole_tree.disabled = False
        
        self.update_image()
        with self.out:
            print(f'New graph with {len(self.graph)} cities created.')
        
        self.reset_widgets()
    
    def on_create_sons(self, b):
        cur_node = self.graph.current_path[-1]
        l = int(self.c_lookahead.value)
        
        sons = [[cur_node]]
        
        for d in range(l):
            sons.append(sum([self.graph.fill_sons(n) for n in sons[d]], []))
        
        self.c_walk_to_dd.options = sons[1]
        self.c_walk_to_dd.value = sons[1][0]
        
        self.update_image(normal_nodes=sum(sons[1:], []))
    
    def update_heuristic_nodes(self):
        nodes = list(self.graph.tree.nodes)
        nodes = list(filter(lambda x: 'f' not in x, nodes))
        self.c_run_heuristic_dd.options = ['All'] + nodes
    
    def on_run_heuristics(self, b):
        node = self.c_run_heuristic_dd.value
        if node == 'All':
            nodes = list([n for n in self.graph.tree.nodes if len(list(self.graph.tree.successors(n))) == 0])
        else:
            nodes = [node]
        
        for n in nodes:
            if n[0] != 'f':
                self.graph.run_heuristic(n)
        
        self.update_image()
    
    def on_walk_to(self, b):
        node = self.c_walk_to_dd.value
        self.graph.current_path.append(node)
        self.update_image()
        if node[0] == 'f':
            self.c_walk_to_b.disabled = True
    
    def on_run_step_rollout(self, b):
        lookahead = self.c_lookahead.value
        self.graph.roll_out(lookahead=lookahead)
        self.update_image()
    
    def on_c_full_simulation(self, b):
        while True:
            self.on_run_step_rollout(None)
            time.sleep(0.1)
            if self.graph.current_path[-1][0] == 'f':
                break
    
    def on_plot_whole_tree(self, b):
        self.update_image(full=True)


def main():
    n = 4
    np.random.seed(754)
    distance_matrix = np.random.randint(2, 20, (n, n))
    
    g = TSPGraph(distance_matrix)
    
    l = 1
    
    # sons = [['A']]
    #
    # for d in range(l):
    #     sons.append(sum([g.fill_sons(n) for n in sons[d]], []))
    
    g.roll_out('A', l)
    
    g.visualize_tree()


if __name__ == "__main__":
    main()