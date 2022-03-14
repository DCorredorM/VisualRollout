import numpy as np

from rollout_graph import TSPGraph

import time
import ipywidgets as widgets

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
           'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']


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
        l = self.c_lookahead.value
        
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
