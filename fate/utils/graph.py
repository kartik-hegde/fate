"""
    This file provides an object to handle graphs.
"""

import pprint
from collections import defaultdict
import math

class Node:
    """
        Defines each node of the graph.
    """
    def __init__(self, name, parents, children, payload):
        self.parents = parents
        self.children = children
        self.name = name
        self.payload = payload
        self.dependency_count = 0

class Graph(object):
    """
        Graph data structure.
        We initialize with a parents.
    """

    def __init__(self):
        self.graph = [Node(0, [], [], None),]

    def add_edges(self, parents, children, payload):
        """
            Add the connections.
        """
        # We will return the length of graph, that should serve as the ID.
        assigned_id = len(self.graph)
        # Add the Node
        self.graph.append(Node(assigned_id, parents, children, payload))
        # Add the edge to each parent
        for parent in parents:
            if assigned_id not in self.graph[parent].children:
                self.graph[parent].children.append(assigned_id)

        return assigned_id

    def delete_edge(self, node1, node2):
        """
            Delete edge between node1 and node2
        """
        raise NotImplementedError

    def delete_node(self, node):
        """
            Remove all references to node
        """
        raise NotImplementedError

    def is_connected(self, node1, node2):
        """
            Check if node1 directly connected to node2
        """
        raise NotImplementedError

    def printGraph(self):
        """
            Visualize the graph.
        """
        for node in self.graph:
            print("Node {0} has parents {1} and {2} children and payload {3}"\
                .format(node.name, node.parents, node.children, node.payload))
        return None

    def getGraphSize(self, opcode_width, max_graph_degree, max_graph_nodes):
        """
            Get the size of graph storage.
        """
        metadata_size = 0
        opcode_size = 0
        opcode_graph_size = 0
        for node in self.graph:
            # Each Node is <opcode><num_parents><num_children><parents
            opcode_size += opcode_width
            metadata_size += (len(node.parents) + len(node.children))*int(math.log(max_graph_nodes,2))
            opcode_graph_size += 2*int(math.log(max_graph_degree,2))

        # Convert to Bytes
        metadata_size, opcode_size, opcode_graph_size = metadata_size>>3, opcode_size>>3, opcode_graph_size>>3

        print("Graph needs {0} Bytes for Opcodes, {1} Bytes for parent+children information (part of opcode) and {2} Bytes for graph metadata.".\
            format(opcode_size, opcode_graph_size, metadata_size))

        return metadata_size + opcode_size + opcode_graph_size

    def getAverageDegree(self):
        """
            Get the average degree of each Node
        """
        indegree = []
        outdegree = []
        for node in self.graph:
            indegree.append(len(node.parents))
            outdegree.append(len(node.children))
        avg_indegree = sum(indegree)/float(len(indegree))
        avg_outdegree = sum(outdegree)/float(len(outdegree))

        print("The Graph has an average in-degree of {0} and out-degree of {1}.".format(avg_indegree, avg_outdegree))

        return avg_indegree, avg_outdegree

