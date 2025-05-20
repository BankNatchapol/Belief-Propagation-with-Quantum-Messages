import numpy as np
import networkx as nx


class LinearCode(object):

    def __init__(self, G, H):
        self.G = np.array(G)
        self.H = np.array(H)
        self.n = G.shape[1]
        self.k = G.shape[0]

    def get_codewords(self):
        def add_cw(i):
            if i == self.k - 1:
                return [np.zeros(self.n), self.G[i]]
            ret = add_cw(i + 1)
            return ret + [(self.G[i] + c) % 2 for c in ret]

        return add_cw(0)

    def get_factor_graph(self):
        G = nx.Graph()
        for i in range(self.n):
            G.add_node("x" + str(i), type="variable")
            G.add_node("y" + str(i), type="output")
            G.add_edge("x" + str(i), "y" + str(i))
        for i in range(self.n - self.k):
            G.add_node("c" + str(i), type="check")
            for j in range(self.n):
                if self.H[i, j] > 0.5:
                    G.add_edge("c" + str(i), "x" + str(j))
        return G

    def get_computation_graph(self, root, height, cloner=None):
        fg = self.get_factor_graph()
        varnodes = [n for n in fg.nodes() if fg.nodes[n]["type"] == "variable"]
        G = nx.DiGraph()
        # keep track how many times each variable has been added in our digraph
        occurances = {v: 0 for v in varnodes}
        # keep track how many check nodes we have added in our digraph
        num_check_nodes = 0

        max_depth = 2 * height + 1

        def handle_node(node, prev, depth):
            nonlocal num_check_nodes
            if depth == max_depth:
                return None
            if fg.nodes[node]["type"] == "output":
                return None
            elif fg.nodes[node]["type"] == "variable":
                node_new = node + "_" + str(occurances[node])
                G.add_node(node_new, type="variable")
                occurances[node] += 1
            elif fg.nodes[node]["type"] == "check":
                node_new = "c" + str(num_check_nodes)
                G.add_node(node_new, type="check")
                num_check_nodes += 1

            descendants = [x for x in list(fg.neighbors(node)) if x != prev]
            for d in descendants:
                d_new = handle_node(d, node, depth + 1)
                if d_new is not None:
                    G.add_edge(node_new, d_new)

            if fg.nodes[node]["type"] == "variable":
                onode = node_new.replace("x", "y")
                G.add_node(onode, type="output")
                G.add_edge(node_new, onode)
            return node_new

        new_root = handle_node(root, None, 0)
        return G, occurances, new_root


def _binary_nullspace(H):
    H = np.array(H, dtype=int) % 2
    m, n = H.shape
    A = H.copy()
    pivots = []
    row = 0
    for col in range(n):
        pivot_row = None
        for r in range(row, m):
            if A[r, col] == 1:
                pivot_row = r
                break
        if pivot_row is not None:
            pivots.append(col)
            if pivot_row != row:
                A[[row, pivot_row]] = A[[pivot_row, row]]
            for r in range(m):
                if r != row and A[r, col] == 1:
                    A[r] ^= A[row]
            row += 1
    free_cols = [c for c in range(n) if c not in pivots]
    basis = []
    for col in free_cols:
        vec = np.zeros(n, dtype=int)
        vec[col] = 1
        for r, pcol in enumerate(pivots):
            if A[r, col] == 1:
                vec[pcol] = 1
        basis.append(vec)
    return np.array(basis, dtype=int)


def generator_from_parity_check(H):
    """Return a generator matrix for the code defined by parity check matrix ``H``."""
    return _binary_nullspace(H)


def solve_syndrome(H, s):
    """Find one vector ``x`` with ``H @ x = s (mod 2)``."""
    H = np.array(H, dtype=int) % 2
    s = np.array(s, dtype=int) % 2
    m, n = H.shape
    A = H.copy()
    b = s.copy()
    pivots = []
    row = 0
    for col in range(n):
        pivot_row = None
        for r in range(row, m):
            if A[r, col] == 1:
                pivot_row = r
                break
        if pivot_row is not None:
            pivots.append(col)
            if pivot_row != row:
                A[[row, pivot_row]] = A[[pivot_row, row]]
                b[[row, pivot_row]] = b[[pivot_row, row]]
            for r in range(m):
                if r != row and A[r, col] == 1:
                    A[r] ^= A[row]
                    b[r] ^= b[row]
            row += 1
    x = np.zeros(n, dtype=int)
    for r in range(row - 1, -1, -1):
        col = pivots[r]
        val = b[r]
        for c in range(col + 1, n):
            if A[r, c] == 1:
                val ^= x[c]
        x[col] = val
    return x
