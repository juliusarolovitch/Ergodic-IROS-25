import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix

class TSP:
    def __init__(self, coordinates, plot_results=False):
        """
        Initializes the TSP solver with a list of 2D coordinates.
        :param coordinates: List of (x, y) tuples representing nodes.
        :param plot_results: Boolean to enable plotting (default: False).
        """
        self.coordinates = np.array(coordinates)
        self.plot_results = plot_results
        self.graph = self.build_graph()

    def build_graph(self):
        """
        Constructs a fully connected graph with distances as edge weights.
        """
        G = nx.complete_graph(len(self.coordinates))
        dist_matrix = distance_matrix(self.coordinates, self.coordinates)
        
        for i in range(len(self.coordinates)):
            for j in range(len(self.coordinates)):
                if i != j:
                    G[i][j]['weight'] = dist_matrix[i, j]
        
        return G

    def solve_tsp(self):
        """
        Solves the TSP problem and extracts edges.
        :return: List of edges where each node connects to exactly two adjacent nodes.
        """
        tsp_path = nx.approximation.traveling_salesman_problem(self.graph, cycle=True)
        
        # Extract edges
        edges = [(tsp_path[i], tsp_path[i + 1]) for i in range(len(tsp_path) - 1)]
        edges.append((tsp_path[-1], tsp_path[0]))  # Complete the cycle
        
        if self.plot_results:
            self.plot_solution(edges)
        
        return edges

    def plot_solution(self, edges):
        """
        Plots the TSP solution with nodes and edges.
        """
        plt.figure(figsize=(8, 6))
        
        # Plot nodes
        plt.scatter(self.coordinates[:, 0], self.coordinates[:, 1], c='red', s=100, label="Nodes")
        
        # Plot edges
        for edge in edges:
            p1, p2 = self.coordinates[edge[0]], self.coordinates[edge[1]]
            plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'b-', linewidth=2)
        
        plt.title("TSP Solution")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()

# Example usage
if __name__ == "__main__":
    coordinates = [(8, 5), (80, 15), (42, 2), (64, 27), (90, 11), (13, 80)]
    
    tsp_solver = TSP(coordinates, plot_results=True)
    edges = tsp_solver.solve_tsp()
    
    print("TSP Edges (each node connects to two adjacent nodes):")
    for edge in edges:
        print(edge)
