import csv
import itertools

import numpy as np
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree
from scipy.special import logsumexp

# Debugging flag for get_tree function
debugging_get_tree = True


class BinaryCLT:
    def __init__(self, data, root=None, alpha=0.01):
        """

        """
        self.data = data
        self.root = root
        self.alpha = alpha

        self.tree = None

    def get_tree(self):
        """
        Returns the list of predecessors of the learned structure:
        - If Xj is the parent of Xi then tree[i] = j
        - If Xi is the root of the tree then tree[i] = -1

        Returns:
            list: A list where the i-th element is the parent of variable i, or -1 if i is the root
        """
        n_vars = self.data.shape[1] # Number of variables

        # Compute mutual information between all pairs of variables (matrix of size n_vars x n_vars
        # Create a fully connected graph with mutual information as edge weights
        mutual_info = np.zeros((n_vars, n_vars))

        # Calculate marginal probabilities with Laplace smoothing
        counts_0 = 2 * self.alpha + np.sum(self.data == 0, axis=0) # Add alpha to both counts
        counts_1 = 2 * self.alpha + np.sum(self.data == 1, axis=0)
        total = 4 * self.alpha + self.data.shape[0] # number of samples; add alpha for each combination

        p_x = np.stack([counts_0, counts_1]) / total # Probability of each variable being 0 or 1

        # Calculate joint probabilities with Laplace smoothing
        for i in range(n_vars):
            for j in range(i+1, n_vars):
                if i == j:  # Skip diagonal entries to prevent self-loops
                    continue
                # Joint counts with Laplace smoothing
                counts_00 = self.alpha + np.sum((self.data[:, i] == 0) & (self.data[:, j] == 0))
                counts_01 = self.alpha + np.sum((self.data[:, i] == 0) & (self.data[:, j] == 1))
                counts_10 = self.alpha + np.sum((self.data[:, i] == 1) & (self.data[:, j] == 0))
                counts_11 = self.alpha + np.sum((self.data[:, i] == 1) & (self.data[:, j] == 1))

                joint_p = np.array([[counts_00, counts_01], [counts_10, counts_11]]) / total

                # Calculate mutual information: I(X,Y) = sum_x,y p(x,y) * log(p(x,y)/(p(x)p(y))
                mi = 0
                for xi in range(2):
                    for xj in range(2):
                        if joint_p[xi, xj] > 0:
                            mi += joint_p[xi, xj] * np.log(joint_p[xi, xj] / (p_x[xi, i] * p_x[xj, j]))

                mutual_info[i, j] = mi
                mutual_info[j, i] = mi  # I(X,Y) = I(Y,X), mutual information is symmetric

        if debugging_get_tree:
            print("Mutual Information Matrix:\n", np.round(mutual_info, 3))

        # Add a small epsilon to avoid zero weights in the MST !!!NOT SURE IF REQUIRED FOR FINAL DELIVERY!!!
        epsilon = 1e-6
        for i in range(n_vars):
            for j in range(n_vars):
                if i != j:
                    mutual_info[i, j] += epsilon


        # Create a maximum spanning tree using minimum_spanning_tree
        # For maximum spanning tree using minimum_spanning_tree, the weights need to be negated
        mst = minimum_spanning_tree(-mutual_info)

        if debugging_get_tree:
            print("MST Edges (row, col, weight):")

            mst_coo = mst.tocoo()
            for i, j, v in zip(mst_coo.row, mst_coo.col, mst_coo.data):
                print(f"{i} -- {j}  (weight: {-v:.4f})")

        # Initialize the tree with -1 (no parent)
        tree = [-1] * n_vars

        # Convert the MST to a directed tree by selecting a root
        # If root is None, choose a random root
        root = self.root if self.root is not None else np.random.randint(0, n_vars)

        # Use breadth-first search to direct edges away from the root
        # Convert the spanning tree as Compressed Sparsed Row (CSR) matrix for the breadth_first_order function
        csr_mst = mst.tocsr()

        # Use breadth-first-order to direct the tree
        order, predecessors = breadth_first_order(csr_mst, i_start=root, directed=False, return_predecessors=True)

        if debugging_get_tree:
            print("BFS Order:", order)
            print("Predecessors:", predecessors)


        # Set parents based on predecessors from BFS
        for i in range(n_vars):
            if i != root:
                # If the predecessor is valid, set it as the parent
                if predecessors[i] >= 0 and predecessors[i] < n_vars:
                    tree[i] = predecessors[i]

        tree = [int(x) for x in tree]
        self.tree = tree
        return tree


    def get_log_params(self):
        """
        Computes the log-conditional probability tables for each node given its parent, using Laplace smoothing.

        Returns:
            np.ndarray: An D x 2 x 2 array where log_params[i, j, k] = log(P(X_i = k | X_parent = j))
        """
        if self.tree is None:
            raise ValueError("Tree has not been computed. Call get_tree() first.")

        n_vars = self.data.shape[1]
        log_params = np.zeros((n_vars, 2, 2))  # Shape: D x 2 x 2

        for i in range(n_vars):
            parent = self.tree[i]

            if parent == -1:
                # Root node: use marginal probabilities (Laplace smoothed)
                counts_0 = self.alpha + np.sum(self.data[:, i] == 0)
                counts_1 = self.alpha + np.sum(self.data[:, i] == 1)
                total = counts_0 + counts_1

                p_0 = counts_0 / total
                p_1 = counts_1 / total

                log_params[i, :, 0] = np.log(p_0)
                log_params[i, :, 1] = np.log(p_1)
            else:
                # Conditional counts
                for parent_val in [0, 1]:
                    for child_val in [0, 1]:
                        count = self.alpha + np.sum((self.data[:, parent] == parent_val) & (self.data[:, i] == child_val))

                        total = 2 * self.alpha + np.sum(self.data[:, parent] == parent_val)
                        log_params[i, parent_val, child_val] = np.log(count / total)

        return log_params

    def log_prob(self, x, exhaustive=False):
        """

        """

        pass

    def sample(self, nsamples):
        """

        """
        if self.tree is None:
            raise ValueError("Tree has not been computed. Call get_tree() first.")

        params = np.exp(self.get_log_params())
        n_vars = self.data.shape[1]
        samples = np.zeros((nsamples, n_vars), dtype=int)

        samples[:, self.root] = np.random.choice([0, 1], size=nsamples, p=params[self.root, 0, :])

        next_nodes = [i for i, parent in enumerate(self.tree) if parent == self.root]

        while next_nodes:
            # Get the next node to sample
            current_node = next_nodes.pop(0)

            # Sample the current node given its parent
            parent_val = samples[:, self.tree[current_node]]
            samples[:, current_node] = np.random.binomial(n=1, p=params[current_node, parent_val, 1])

            # Find the children of the current node
            children = [i for i, parent in enumerate(self.tree) if parent == current_node]

            # Add children to the list of nodes to sample
            next_nodes.extend(children)

        return samples


def test_get_tree_simple():
    # X0 and X1 are highly correlated
    # X2 and X3 are highly correlated
    # X0 and X2 are somewhat correlated
    data = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1]
    ])

    # Create a BinaryCLT with a fixed root (X0)
    clt = BinaryCLT(data, root=0)
    tree = clt.get_tree()

    print("Tree with root at X0:", tree)

    # The expected tree structure [-1, 0, 0, 2]

    # Different root (X2)
    clt2 = BinaryCLT(data, root=2)
    tree2 = clt2.get_tree()

    print("Tree with root at X2:", tree2)
    # Expected structure [2, 0, -1, 2]


    print(np.exp(clt2.get_log_params()))

    samples = clt2.sample(10_000)
    print(np.sum(samples, axis=0) / len(samples))


    print(np.sum(samples[samples[:, 0] == 0][:, 1]) / len(samples[samples[:, 0] == 1]))

if debugging_get_tree:
    test_get_tree_simple()
