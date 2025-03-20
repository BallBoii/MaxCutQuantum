import networkx as nx
from typing import List, Tuple, Dict, Union
from amplify import FixstarsClient, solve, VariableGenerator
from datetime import timedelta
import matplotlib.pyplot as plt
from qiskit_optimization.applications import Maxcut
import numpy as np

class Classical_Solver:
    def __init__(self, graph: nx.Graph):
        """
        Initialize the Classical_Solver with a weighted graph.

        Parameters
        ----------
        graph : nx.Graph
            A weighted graph. Edge weights should be stored under the 'weight'
            attribute (defaulting to 1.0 if not present).
        """
        self.graph = graph
        self.adjacency_matrix = nx.adjacency_matrix(graph).todense()
        self.nodes: List = list(graph.nodes())
        self.n: int = len(self.nodes)
        self.node_to_index: Dict = {node: i for i, node in enumerate(self.nodes)}

        # Prepare edge data from the graph.
        self.edges_data: List[Tuple] = list(graph.edges(data=True))
        if self.edges_data:
            self.edges: List[Tuple[int, int]] = [(u, v) for u, v, data in self.edges_data]
            self.edges_src: List[int] = [self.node_to_index[u] for u, v, data in self.edges_data]
            self.edges_tgt: List[int] = [self.node_to_index[v] for u, v, data in self.edges_data]
            self.edge_weights: List[float] = [data.get('weight', 1.0) for u, v, data in self.edges_data]
        else:
            self.edges, self.edges_src, self.edges_tgt, self.edge_weights = [], [], [], []
        
        # Attributes to store solution results.
        self.max_cut_value: Union[float, None] = None
        self.max_cut_partitions: List[List[int]] = []
        self.distribution: Dict[float, int] = {}
        self.bitstring: List[int] = None
        self.result = None  # This will hold the raw result from Fixstars

    def setClient(self, FIXSTAR_TOKEN: str):
        """
        Set up the Fixstars client using the provided token.

        Parameters
        ----------
        FIXSTAR_TOKEN : str
            The token used to authenticate with Fixstars.
        """
        self.client = FixstarsClient()
        self.client.token = FIXSTAR_TOKEN
        # Set additional parameters, e.g., timeout and output count.
        self.client.parameters.timeout = timedelta(milliseconds=1000)
        self.client.parameters.outputs.num_outputs = 0

    def Module(self):
        """
        Generate the quadratic model for max-cut using Qiskit's Maxcut application.

        Returns
        -------
        VariableGenerator.matrix
            A matrix (model) suitable for solving via Fixstars.
        """
        max_cut = Maxcut(self.adjacency_matrix)
        qp = max_cut.to_quadratic_program()
        Quadratic = qp.objective.quadratic.to_array()
        Linear = qp.objective.linear.to_array()

        gen = VariableGenerator()
        m = gen.matrix("Binary", self.n)
        m.quadratic = -Quadratic  # Flip sign to convert to maximization problem
        m.linear = -Linear
        return m

    def Classical_solve(self):
        """
        Solve the max-cut problem using Fixstars solver.
        """
        m = self.Module()
        # Call Fixstars solver with the generated model.
        self.result = solve(m, self.client)
        self.bitstring = list(self.result.client_result.spins[0])
        self.max_cut_value = self._evaluate_cut(self.bitstring)

    def get_result(self):
        return self.result

    def _evaluate_cut(self, partition: List[int]) -> float:
        """
        Compute the cut value for a given partition.

        Parameters
        ----------
        partition : List[int]
            A list of 0s and 1s representing the assignment of each node
            (based on the order in self.nodes).

        Returns
        -------
        float
            The cut value, i.e. the sum of weights of edges crossing the partition.
        """
        cut_value = 0.0
        for u, v, data in self.graph.edges(data=True):
            i = self.node_to_index[u]
            j = self.node_to_index[v]
            # If the vertices are in different partitions, add the weight.
            if partition[i] != partition[j]:
                weight = data.get('weight', 1.0)
                cut_value += weight
        return cut_value

    def get_max_cut_value(self) -> float:
        """
        Returns the maximum cut value found.

        Returns
        -------
        float
            The maximum cut value (if computed from self.result).
        """
        # Implement extraction logic from self.result if needed.
        return self.max_cut_value

    def get_bitstring(self) -> List[int]:
        return self.bitstring

    def get_distribution(self) -> Dict[float, int]:
        """
        Returns the distribution of cut values over all partitions.

        Returns
        -------
        Dict[float, int]
            A dictionary mapping each cut value to the count of partitions achieving it.
        """
        # Implement extraction logic from self.result if needed.
        return self.distribution

    def get_partition_sets(self, partition: List[int]) -> Tuple[set, set]:
        """
        Convert a partition (as a bitstring) into two sets of nodes.

        Parameters
        ----------
        partition : List[int]
            A list of 0s and 1s representing the assignment of each node.

        Returns
        -------
        Tuple[set, set]
            A tuple (set1, set2) where set1 contains the nodes assigned 0 and
            set2 contains the nodes assigned 1.
        """
        set1 = {self.nodes[i] for i in range(self.n) if partition[i] == 0}
        set2 = {self.nodes[i] for i in range(self.n) if partition[i] == 1}
        return set1, set2
