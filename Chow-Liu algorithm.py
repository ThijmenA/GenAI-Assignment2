import csv
import itertools

import numpy as np
from scipy.sparse.csgraph import breadth_first_order, minimum_spanning_tree
from scipy.special import logsumexp

# Debugging flag for get_tree function
debugging_get_tree = False


class BinaryCLT:
    def __init__(self, data, root=None, alpha=0.01):
        """

        """
        self.data = data
        self.root = root
        self.alpha = alpha

        self.tree = None
        self.children = None
        self.postorder = None
        self._log_params = None

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
        self.children = [[] for _ in range(n_vars)]
        for c, p in enumerate(self.tree):
            if p != -1:
                self.children[p].append(c)
        self.postorder = order[::-1]
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
        Row-wise log p(y)  —  x may contain NaN for unobserved variables.
        """
        if self.tree is None:
            raise ValueError("Tree has not been computed. Call get_tree() first.")

        # --- ensure children / postorder exist (robust if user called get_tree() before)
        if self.children is None or self.postorder is None:
            n = len(self.tree)
            self.children = [[] for _ in range(n)]
            for c, p in enumerate(self.tree):
                if p != -1:
                    self.children[p].append(c)
            order, _ = breadth_first_order(
                np.ones((n, n)), i_start=self.root, directed=False, return_predecessors=False
            )
            self.postorder = order[::-1]

        x = np.asarray(x, dtype=float)
        n_rows, d = x.shape
        out = np.zeros(n_rows)
        log_cpt = self.get_log_params()

        for r in range(n_rows):
            row = x[r]

            # ---------- fully observed fast path
            if not np.isnan(row).any():
                logp = log_cpt[self.root, 0, int(row[self.root])]
                for v in range(d):
                    p = self.tree[v]
                    if p != -1:
                        logp += log_cpt[v, int(row[p]), int(row[v])]
                out[r] = logp
                continue

            # ---------- exhaustive enumeration
            if exhaustive:
                missing = np.where(np.isnan(row))[0]
                terms = []
                for ass in itertools.product((0, 1), repeat=len(missing)):
                    filled = row.copy()
                    filled[missing] = ass
                    terms.append(self.log_prob(filled.reshape(1, -1))[0])
                out[r] = logsumexp(terms)
                continue

            # ---------- variable elimination (message passing on tree)
            msg = np.zeros((d, 2))                # msg[v, k] = log m_{v→parent}(k)

            for v in self.postorder:
                # evidence for node v (length-2 array)
                if np.isnan(row[v]):
                    ev = np.zeros(2)
                else:
                    k_obs = int(row[v])
                    ev = np.array([0.0, -np.inf]) if k_obs == 0 else np.array([-np.inf, 0.0])

                # pre-compute contribution from children for each state k of v
                child_sum = np.zeros(2)
                for c in self.children[v]:
                    child_sum += msg[c]           # msg[c,k] already conditioned on k

                if v == self.root:
                    # final log p(y) = log ∑_k P(root=k)·evidence·child-msgs
                    root_terms = log_cpt[self.root, 0] + ev + child_sum
                    out[r] = logsumexp(root_terms)
                else:
                    parent_vals = (0, 1)
                    for pv in parent_vals:
                        # for each parent state pv compute message m_{v→p}(pv)
                        terms = log_cpt[v, pv] + ev + child_sum  # vector over k∈{0,1}
                        msg[v, pv] = logsumexp(terms)


        return out



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

def clt_nltcs_demo(
    ds_path="Density-Estimation-Datasets/datasets/nltcs",
    alpha=0.01,
    root=0,
    nan_ratio=0.05,
):
    """
    Train on NLTCS, then print:
      • parent array
      • train / test average LL
      • 5 %-NaN query runtime + consistency (exhaustive vs VE)
      • 1000-sample average LL
    """
    # --- load data -------------------------------------------------
    train = np.genfromtxt(f"{ds_path}/nltcs.train.data", delimiter=",")
    test  = np.genfromtxt(f"{ds_path}/nltcs.test.data",  delimiter=",")

    # --- train CLT -------------------------------------------------
    clt = BinaryCLT(train, root=root, alpha=alpha)
    clt.get_tree()

    print("Parents:", clt.tree)
    print("AvgLL train:", clt.log_prob(train).mean())
    print("AvgLL test :", clt.log_prob(test ).mean())

    # --- 5 % NaN query set ----------------------------------------
    q = test.copy()
    idx = np.random.choice(q.size, int(nan_ratio * q.size), replace=False)
    q.flat[idx] = np.nan

    import time
    t0 = time.time(); lp_ex = clt.log_prob(q, exhaustive=True ); te = time.time() - t0
    t0 = time.time(); lp_ve = clt.log_prob(q, exhaustive=False); tv = time.time() - t0

    print(f"NaN-queries  enumeration {te:.3f}s   VE {tv:.3f}s")
    print("Max Δ", np.max(np.abs(lp_ex - lp_ve)))
    print("All-close", np.allclose(lp_ex, lp_ve))

    # --- sampling --------------------------------------------------
    samp = clt.sample(1000)
    print("AvgLL sample:", clt.log_prob(samp).mean())


if debugging_get_tree:
    test_get_tree_simple()

if __name__ == "__main__":
    clt_nltcs_demo()
