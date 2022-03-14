import numpy as np
from graph import TSPGraph


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
    
    # g.roll_out('A', l)
    
    g.visualize_tree(full=True)


if __name__ == "__main__":
    main()