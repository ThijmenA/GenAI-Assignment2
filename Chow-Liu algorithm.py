from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse.csgraph import breadth_first_order
from scipy.special import logsumexp
import numpy as np
import itertools
import csv

# Debugging flag for get_tree function
debugging_get_tree = False


class BinaryCLT:
    def __init__(self, data, root=None, alpha=0.01):
        """

        """
        self.data = data
        self.root = root
        self.alpha = alpha

        pass

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
        return tree

    def get_log_params(self):
        """

        """

        pass
    
    def log_prob(self, x, exhaustive=False):
        """

        """

        pass

    def sample(self, nsamples):
        """

        """

        pass





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

if debugging_get_tree:
    test_get_tree_simple()