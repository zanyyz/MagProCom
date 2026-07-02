import os
import time
import math
import random
from datetime import datetime

import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

import pandas as pd
import tkinter as tk
from tkinter import filedialog


# ══════════════════════════════════════════════════════════
#  Global font: Arial for all figure text
# ══════════════════════════════════════════════════════════
matplotlib.rcParams.update({
    "font.family"      : "Arial",
    "axes.titlesize"   : 11,
    "axes.labelsize"   : 9,
    "xtick.labelsize"  : 8,
    "ytick.labelsize"  : 8,
    "legend.fontsize"  : 8,
    "figure.titlesize" : 14,
})


# ══════════════════════════════════════════════════════════
#  Constants
# ══════════════════════════════════════════════════════════
BOARD_SIZE    = 9
BLACK         =  1
WHITE         = -1
EMPTY         =  0
KOMI          =  6.5        # Chinese rules komi
PASS_MOVE     = (-1, -1)    # Pass marker

# MCTS parameters
MCTS_ITER     = 300         # MCTS iterations per move
UCB_C         = 1.4         # UCB1 exploration constant
ROLLOUT_DEPTH = 40          # Rollout depth

# Self-supervised readout parameter
SS_LAG        = 5           # Temporal prediction lag (adjust to sampling rate)



# ══════════════════════════════════════════════════════════
#  Leaky Echo State Network
# ══════════════════════════════════════════════════════════
class LeakyESN:
    """
    Leaky integrator Echo State Network.
    The readout layer is implemented externally with linear Ridge regression.
    This class only drives reservoir states.
    """

    def __init__(
        self,
        input_dim,
        reservoir_size  = 800,
        spectral_radius = 0.92,
        sparsity        = 0.08,
        leak_rate       = 0.25,
        input_scale     = 0.08,
        seed            = 42,
    ):
        np.random.seed(seed)
        self.N    = reservoir_size
        self.leak = leak_rate

        self.Win = np.random.randn(self.N, input_dim) * input_scale

        W    = np.random.randn(self.N, self.N)
        mask = np.random.rand(self.N, self.N) < sparsity
        W   *= mask

        eig      = np.max(np.abs(np.linalg.eigvals(W)))
        self.W   = W / eig * spectral_radius
        self.state = np.zeros((self.N, 1))

    def reset(self):
        self.state[:] = 0.0

    def step(self, u):
        u          = u.reshape(-1, 1)
        pre        = self.Win @ u + self.W @ self.state
        new_state  = np.tanh(pre)
        self.state = (1 - self.leak) * self.state + self.leak * new_state
        return self.state.flatten()


# ══════════════════════════════════════════════════════════
#  Go Board — complete Chinese-rules implementation
# ══════════════════════════════════════════════════════════
class GoBoard:
    """
    9x9 Go, Chinese rules.
    Implements: captures, ko (Zobrist), suicide prohibition, territory scoring.
    """

    def __init__(self, size: int = BOARD_SIZE):
        self.size         = size
        self.board        = np.zeros((size, size), dtype=np.int8)
        self.ko           = None
        self.captures     = {BLACK: 0, WHITE: 0}
        self.move_history : list[tuple] = []
        self._zobrist_init()
        self.zobrist_hash = 0
        self.position_set = set()

    def _zobrist_init(self):
        rng = np.random.default_rng(0)
        self.zobrist_table = {
            BLACK: rng.integers(0, 2**63, (self.size, self.size), dtype=np.int64),
            WHITE: rng.integers(0, 2**63, (self.size, self.size), dtype=np.int64),
        }

    def _hash_update(self, r, c, color):
        self.zobrist_hash ^= int(self.zobrist_table[color][r, c])

    def neighbors(self, r, c):
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            rr, cc = r+dr, c+dc
            if 0 <= rr < self.size and 0 <= cc < self.size:
                yield rr, cc

    def _group_and_liberties(self, r, c):
        color   = self.board[r, c]
        visited = set()
        libs    = set()
        stack   = [(r, c)]
        while stack:
            cr, cc = stack.pop()
            if (cr, cc) in visited:
                continue
            visited.add((cr, cc))
            for nr, nc in self.neighbors(cr, cc):
                if self.board[nr, nc] == EMPTY:
                    libs.add((nr, nc))
                elif self.board[nr, nc] == color and (nr, nc) not in visited:
                    stack.append((nr, nc))
        return visited, libs

    def _remove_group(self, group, color):
        for r, c in group:
            self.board[r, c] = EMPTY
            self._hash_update(r, c, color)
        self.captures[-color] += len(group)

    def place(self, r, c, color):
        """Place a stone. Returns True if legal, False otherwise."""
        if r == -1:                          # pass move
            self.move_history.append(PASS_MOVE)
            self.ko = None
            return True

        if not (0 <= r < self.size and 0 <= c < self.size):
            return False
        if self.board[r, c] != EMPTY:
            return False
        if (r, c) == self.ko:
            return False

        self.board[r, c] = color
        self._hash_update(r, c, color)

        # Capture opponent groups with no liberties
        captured = []
        for nr, nc in self.neighbors(r, c):
            if self.board[nr, nc] == -color:
                grp, libs = self._group_and_liberties(nr, nc)
                if len(libs) == 0:
                    captured.append(grp)

        for grp in captured:
            self._remove_group(grp, -color)

        # Suicide check
        _, my_libs = self._group_and_liberties(r, c)
        if len(my_libs) == 0:
            self.board[r, c] = EMPTY
            self._hash_update(r, c, color)
            for grp in captured:
                for cr, cc in grp:
                    self.board[cr, cc] = -color
                    self._hash_update(cr, cc, -color)
                self.captures[-color] -= len(grp)
            return False

        # Ko detection
        total_captured = sum(len(g) for g in captured)
        self.ko = list(list(captured[0]))[0] if total_captured == 1 else None

        # Superko / repetition check
        h = self.zobrist_hash
        if h in self.position_set:
            self.board[r, c] = EMPTY
            self._hash_update(r, c, color)
            for grp in captured:
                for cr, cc in grp:
                    self.board[cr, cc] = -color
                    self._hash_update(cr, cc, -color)
                self.captures[-color] -= len(grp)
            self.ko = None
            return False

        self.position_set.add(h)
        self.move_history.append((r, c))
        return True

    def legal_moves(self, color):
        return [
            (r, c)
            for r in range(self.size)
            for c in range(self.size)
            if self.board[r, c] == EMPTY and (r, c) != self.ko
        ]

    def clone(self):
        b               = GoBoard.__new__(GoBoard)
        b.size          = self.size
        b.board         = self.board.copy()
        b.ko            = self.ko
        b.captures      = dict(self.captures)
        b.move_history  = list(self.move_history)
        b.zobrist_table = self.zobrist_table   # shared read-only
        b.zobrist_hash  = self.zobrist_hash
        b.position_set  = set(self.position_set)
        return b

    def score_chinese(self):
        """
        Returns (black_score, white_score) normalized to [0,1].
        Uses flood-fill territory assignment + komi.
        """
        visited   = np.zeros((self.size, self.size), dtype=bool)
        territory = np.zeros((self.size, self.size), dtype=np.int8)

        for r in range(self.size):
            for c in range(self.size):
                if not visited[r, c] and self.board[r, c] == EMPTY:
                    stack  = [(r, c)]
                    region = []
                    border = set()
                    while stack:
                        cr, cc = stack.pop()
                        if visited[cr, cc]:
                            continue
                        visited[cr, cc] = True
                        region.append((cr, cc))
                        for nr, nc in self.neighbors(cr, cc):
                            if self.board[nr, nc] == EMPTY and not visited[nr, nc]:
                                stack.append((nr, nc))
                            elif self.board[nr, nc] != EMPTY:
                                border.add(self.board[nr, nc])
                    owner = EMPTY
                    if len(border) == 1:
                        owner = border.pop()
                    for cr, cc in region:
                        territory[cr, cc] = owner

        black           = int(np.sum(self.board == BLACK) + np.sum(territory == BLACK))
        white           = int(np.sum(self.board == WHITE) + np.sum(territory == WHITE))
        white_with_komi = white + KOMI
        total           = black + white_with_komi
        return black / total, white_with_komi / total

    def print_board(self):
        sym = {BLACK: '●', WHITE: '○', EMPTY: '·'}
        for r in range(self.size):
            print(' '.join(sym[int(self.board[r, c])] for c in range(self.size)))
        print()


# ══════════════════════════════════════════════════════════
#  MCTS Node
# ══════════════════════════════════════════════════════════
class MCTSNode:
    __slots__ = ('move','parent','children','wins','visits','untried','board','color')

    def __init__(self, board: GoBoard, color: int, move=None, parent=None):
        self.board    = board
        self.color    = color
        self.move     = move
        self.parent   = parent
        self.children : list['MCTSNode'] = []
        self.wins     = 0.0
        self.visits   = 0
        legal = board.legal_moves(color)
        random.shuffle(legal)
        self.untried  = legal

    def ucb1(self, c=UCB_C):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits
                + c * math.sqrt(math.log(self.parent.visits) / self.visits))

    def best_child(self):
        return max(self.children, key=lambda n: n.ucb1())

    def is_terminal(self):
        h = self.board.move_history
        return len(h) >= 2 and h[-1] == PASS_MOVE and h[-2] == PASS_MOVE


# ══════════════════════════════════════════════════════════
#  White AI — UCB1-MCTS with heuristic rollout
# ══════════════════════════════════════════════════════════
class WhiteAI:
    """
    UCB1-MCTS for White.
    Rollout heuristic priority: capture > save liberty > avoid true eye
                                > star point > connect own stones > noise
    """

    def __init__(self, iterations: int = MCTS_ITER):
        self.iterations = iterations

    def choose(self, board: GoBoard) -> tuple:
        legal = board.legal_moves(WHITE)
        if not legal:
            return PASS_MOVE
        if len(legal) <= 3:
            return self._heuristic_best(board, WHITE, legal)

        root = MCTSNode(board.clone(), WHITE)
        for _ in range(self.iterations):
            node  = self._select(root)
            score = self._rollout(node.board, node.color)
            self._backprop(node, score)

        if not root.children:
            return self._heuristic_best(board, WHITE, legal)
        return max(root.children, key=lambda n: n.visits).move

    def _select(self, node):
        while not node.is_terminal():
            if node.untried:
                return self._expand(node)
            node = node.best_child()
        return node

    def _expand(self, node):
        move        = node.untried.pop()
        child_board = node.board.clone()
        child_board.place(*move, node.color)
        child = MCTSNode(child_board, -node.color, move=move, parent=node)
        node.children.append(child)
        return child

    def _rollout(self, board: GoBoard, color: int) -> float:
        b      = board.clone()
        cur    = color
        passes = 0
        for _ in range(ROLLOUT_DEPTH):
            legal = b.legal_moves(cur)
            if not legal:
                b.place(-1, -1, cur)
                passes += 1
            else:
                b.place(*self._heuristic_best(b, cur, legal), cur)
                passes = 0
            cur = -cur
            if passes >= 2:
                break
        _, ws = b.score_chinese()
        return ws   # White score including komi

    def _backprop(self, node, score):
        while node is not None:
            node.visits += 1
            node.wins   += score if node.color == WHITE else (1.0 - score)
            node = node.parent

    def _heuristic_best(self, board: GoBoard, color: int, legal: list) -> tuple:
        if not legal:
            return PASS_MOVE
        stars  = {(2,2),(2,4),(2,6),(4,2),(4,4),(4,6),(6,2),(6,4),(6,6)}
        center = np.array([board.size // 2, board.size // 2])
        opp    = -color
        best_s, best_m = -1e9, legal[0]

        for (r, c) in legal:
            s = 0.0
            # Capture threat: opponent group with 1 liberty
            for nr, nc in board.neighbors(r, c):
                if board.board[nr, nc] == opp:
                    _, libs = board._group_and_liberties(nr, nc)
                    if len(libs) == 1:
                        s += 50.0
            # Save own group: extend liberty
            for nr, nc in board.neighbors(r, c):
                if board.board[nr, nc] == color:
                    _, libs = board._group_and_liberties(nr, nc)
                    if len(libs) == 1:
                        s += 30.0
            # Avoid filling true eye
            if self._is_true_eye(board, r, c, color):
                s -= 200.0
            # Star point / center preference
            if (r, c) in stars:
                s += 8.0
            s += (4.0 - np.linalg.norm(np.array([r, c]) - center)) * 1.5
            # Connect own stones
            for nr, nc in board.neighbors(r, c):
                if board.board[nr, nc] == color:
                    s += 3.0
            # Small noise to break ties
            s += random.gauss(0, 0.5)
            if s > best_s:
                best_s, best_m = s, (r, c)
        return best_m

    @staticmethod
    def _is_true_eye(board: GoBoard, r, c, color) -> bool:
        for nr, nc in board.neighbors(r, c):
            if board.board[nr, nc] != color:
                return False
        diags = [(r-1,c-1),(r-1,c+1),(r+1,c-1),(r+1,c+1)]
        good  = sum(
            1 for dr, dc in diags
            if not (0 <= dr < board.size and 0 <= dc < board.size)
            or board.board[dr, dc] == color
        )
        return good >= 3


# ══════════════════════════════════════════════════════════
#  Neural Go System (Self-Supervised Readout)
# ══════════════════════════════════════════════════════════
class NeuralGoSystem:
    """
    Pipeline
    --------
    1. Load .mat  → StandardScaler → tanh compress → PCA(95%) → StandardScaler
    2. LeakyESN drive → reservoir state sequence  (T, N)
    3. Self-supervised linear readout (Scheme A):
         Stage A  Ridge: x(t) → x(t+lag)              [linear]
         Stage B  Ridge: cluster anchors → board space [linear]
    4. Participation Ratio peaks → neural event detection
    5. Per-event: Black places via decoded probability | White responds (MCTS)
    6. Terminal: double-pass or 80 steps
    7. Export Excel + Analysis figure (3-panel, white bg) + Board figure (standalone)
    """

    def __init__(self, path: str, mcts_iter: int = MCTS_ITER, lag: int = SS_LAG):
        self.path     = path
        self.save_dir = os.path.dirname(path) or "."
        self.lag      = lag
        self.ai       = WhiteAI(iterations=mcts_iter)
        self.env      = GoBoard()

        print("[1/4] Loading and preprocessing neural data ...")
        self._load_data()
        print("[2/4] Driving reservoir ...")
        self._drive_reservoir()
        print("[3/4] Fitting self-supervised linear readout ...")
        self._fit_self_supervised_readout()
        print("[4/4] Ready. Starting game.\n")

    # ── Data loading ──────────────────────────────────────
    def _load_data(self):
        mat = sio.loadmat(self.path)
        key = [k for k in mat if not k.startswith("__")][0]
        X   = mat[key]
        X   = X.T if X.shape[0] < X.shape[1] else X
        X   = StandardScaler().fit_transform(X)
        X   = np.tanh(X)                              # compress outliers
        pca = PCA(n_components=0.95, random_state=42)
        X   = pca.fit_transform(X)
        self.X = StandardScaler().fit_transform(X)

    # ── Reservoir drive ───────────────────────────────────
    def _drive_reservoir(self):
        self.esn = LeakyESN(self.X.shape[1])
        states   = [self.esn.step(self.X[t]) for t in range(len(self.X))]
        self.states = np.array(states)                # (T, N)

    # ── Self-supervised linear readout ───────────────────
    def _fit_self_supervised_readout(self):
        """
        STAGE A — Temporal self-supervised prediction
        -----------------------------------------------
        Goal  : learn f(x_t) ≈ x_{t+lag}  via Ridge linear regression.
        Why   : instead of arbitrary clustering labels, the readout learns
                the "forward manifold direction" of neural dynamics, which
                carries genuine functional information about the trajectory
                of reservoir states.
        Constraint: Ridge = W_A · x + b_A  (purely linear, no nonlinearity).

        STAGE B — State space → board space linear projection
        -------------------------------------------------------
        Goal  : linearly project reservoir states to 81-dim board logits.
        Method: fit MiniBatchKMeans cluster centers as anchor points, then
                learn Ridge W_B such that W_B · anchor_k ≈ one-hot(k).
                Clustering is used only to generate spatial anchors in
                state space, NOT as readout targets fed from data.
        Constraint: Ridge = W_B · x + b_B  (purely linear).

        The softmax in _decode_action is normalization only, not a feature
        transform, so the full pipeline remains strictly linear.
        """
        S   = self.states
        lag = self.lag

        # Stage A: linear temporal prediction
        self.readout_a = Ridge(alpha=1.0, fit_intercept=True)
        self.readout_a.fit(S[:-lag], S[lag:])         # x(t) → x(t+lag)

        # Stage B: cluster anchors → board linear projection
        km = MiniBatchKMeans(
            n_clusters   = 81,
            batch_size   = 512,
            random_state = 42,
            n_init       = 5,
        )
        km.fit(S)
        anchor_states  = km.cluster_centers_           # (81, N)
        anchor_targets = np.eye(81, dtype=np.float32)  # (81, 81)

        self.readout_b = Ridge(alpha=1.0, fit_intercept=True)
        self.readout_b.fit(anchor_states, anchor_targets)

    # ── Participation Ratio ───────────────────────────────
    def _participation_ratio(self):
        window = 20
        prs    = []
        for i in range(len(self.states) - window):
            seg     = self.states[i : i + window]
            _, s, _ = np.linalg.svd(seg, full_matrices=False)
            s2      = s ** 2
            pr      = (np.sum(s2) ** 2) / (np.sum(s2 ** 2) + 1e-9)
            prs.append(pr)
        return gaussian_filter1d(np.array(prs), sigma=2)

    # ── Decode action (strictly linear pipeline) ──────────
    def _decode_action(self, state: np.ndarray) -> np.ndarray:
        """
        Stage A (linear): predict future reservoir state.
        Stage B (linear): project to board logits.
        Softmax: normalization only (not a feature transform).
        """
        future = self.readout_a.predict(state.reshape(1, -1))[0]     # linear
        logits = self.readout_b.predict(future.reshape(1, -1))[0]    # linear
        probs  = softmax(logits / 0.15)                              # normalize
        return probs.reshape(BOARD_SIZE, BOARD_SIZE)

    # ── Main game loop ────────────────────────────────────
    def run(self):
        pr        = self._participation_ratio()
        events, _ = find_peaks(
            -pr,
            distance   = 35,
            prominence = np.std(pr) * 0.3,
        )

        metrics   = []
        occupancy = np.zeros((BOARD_SIZE, BOARD_SIZE))
        max_steps = min(80, len(events))

        print(f"Detected {len(events)} neural events → up to {max_steps} game steps.\n")

        for i, e in enumerate(events[:max_steps]):

            # Black (neural signal) move
            prob       = self._decode_action(self.states[e])
            flat_order = np.argsort(prob.flatten())[::-1]
            placed     = False
            for idx in flat_order:
                r, c = divmod(int(idx), BOARD_SIZE)
                if self.env.place(r, c, BLACK):
                    occupancy[r, c] += 1
                    placed = True
                    break
            if not placed:
                self.env.place(-1, -1, BLACK)

            # White (MCTS) move
            t0    = time.time()
            wmove = self.ai.choose(self.env)
            self.env.place(*wmove, WHITE)
            ai_ms = (time.time() - t0) * 1000

            # Record metrics
            bs, ws = self.env.score_chinese()
            metrics.append({
                "step"        : i + 1,
                "event_t"     : int(e),
                "black_score" : round(bs, 4),
                "white_score" : round(ws, 4),
                "pr"          : round(float(pr[e]), 4),
                "ai_ms"       : round(ai_ms, 1),
                "black_caps"  : self.env.captures[BLACK],
                "white_caps"  : self.env.captures[WHITE],
            })

            leader = "Black" if bs > ws else "White"
            print(
                f"\rStep {i+1:3d} | Black={bs:.3f}  White={ws:.3f}"
                f" | Leading={leader} | AI={ai_ms:.0f}ms",
                end="", flush=True,
            )

            # Double-pass termination
            h = self.env.move_history
            if len(h) >= 2 and h[-1] == PASS_MOVE and h[-2] == PASS_MOVE:
                print("\n\nDouble pass — game over.")
                break

        print("\n")
        bs_f, ws_f = self.env.score_chinese()
        print("=" * 48)
        print(f"  Final | Black={bs_f:.4f}  White={ws_f:.4f}  (komi={KOMI})")
        print(f"  >>> {'Black wins!' if bs_f > ws_f else 'White wins!'} <<<")
        print("=" * 48 + "\n")

        self.env.print_board()
        self._export(metrics)
        self._plot_analysis(pr, events[:max_steps], metrics)
        self._plot_board(occupancy)

    # ── Excel export ──────────────────────────────────────
    def _export(self, metrics: list):
        df   = pd.DataFrame(metrics)
        name = "NeuralGo_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".xlsx"
        path = os.path.join(self.save_dir, name)
        df.to_excel(path, index=False)
        print(f"[Saved] {path}")

    # ── Pearson r helper ──────────────────────────────────
    @staticmethod
    def _safe_pearson(x, y):
        """Pearson r, returns 0 if either vector is constant."""
        if np.std(x) < 1e-9 or np.std(y) < 1e-9:
            return 0.0, 1.0
        r, p = pearsonr(x, y)
        return (0.0, 1.0) if np.isnan(r) else (r, p)



    # ══════════════════════════════════════════════════════
    #  Figure 1: Analysis (2-row, 6-panel, white background)
    # ══════════════════════════════════════════════════════
    def _plot_analysis(self, pr, events, metrics):
        """
        Row 1 (original 3 panels):
          P1 — Reservoir Participation Ratio + event markers
          P2 — PCA manifold coloured by time
          P3 — Score dynamics

        Row 2 (new panels):
          P4 — Pre-clustering PCA manifold  (coloured by time, no cluster info)
          P5 — Post-clustering PCA manifold (coloured by KMeans cluster label)
                + Pearson r between PC1/PC2 and Black score annotated
        """
        # ── Pre-compute clustering and Pearson r ──────────
        N_CLUSTERS = 8
        proj2 = PCA(n_components=2, random_state=42).fit_transform(self.states)

        km8 = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
        cluster_labels = km8.fit_predict(self.states)

        # Align PR curve time-axis with event steps for correlation
        if len(metrics) > 1:
            event_ts  = [m["event_t"] for m in metrics]
            bs_seq    = np.array([m["black_score"] for m in metrics])
            ws_seq    = np.array([m["white_score"] for m in metrics])
            adv_seq   = bs_seq - ws_seq

            valid = [t for t in event_ts if t < len(proj2)]
            if len(valid) >= 3:
                pc1_at_events = proj2[valid, 0]
                pc2_at_events = proj2[valid, 1]
                adv_at_events = adv_seq[:len(valid)]
                r1, p1 = self._safe_pearson(pc1_at_events, adv_at_events)
                r2, p2 = self._safe_pearson(pc2_at_events, adv_at_events)
            else:
                r1, p1, r2, p2 = 0.0, 1.0, 0.0, 1.0
        else:
            r1, p1, r2, p2 = 0.0, 1.0, 0.0, 1.0

      
        # ── Layout ───────────────────────────────────────
        fig, axes = plt.subplots(
            2, 3,
            figsize=(21, 12),
            facecolor="white",
        )
        fig.suptitle(
            "NeuralGoSystem — Neural Reservoir Analysis",
            fontsize=14, y=1.01, color="#1a1a1a",
        )

        def _style(ax):
            ax.set_facecolor("white")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#cccccc")
            ax.spines["bottom"].set_color("#cccccc")
            ax.tick_params(colors="#444444")

        # ── Panel 1: Participation Ratio ──────────────────
        ax1 = axes[0, 0]
        _style(ax1)
        ax1.plot(pr, color="#1565C0", lw=1.3, label="PR", zorder=2)
        if len(events) > 0:
            ax1.scatter(
                events, pr[events],
                c="#E53935", s=28, zorder=5,
                edgecolors="#B71C1C", linewidths=0.5,
                label="Neural events",
            )
        ax1.set_xlabel("Time step")
        ax1.set_ylabel("Participation Ratio")
        ax1.set_title("Reservoir Participation Ratio")
        ax1.legend(frameon=True, facecolor="white", edgecolor="#dddddd")

        # ── Panel 2: PCA Manifold (time-coloured) ─────────
        ax2 = axes[0, 1]
        _style(ax2)
        sc2 = ax2.scatter(
            proj2[:, 0], proj2[:, 1],
            c=np.arange(len(proj2)), cmap="Blues",
            s=4, alpha=0.65, zorder=2,
        )
        if len(events) > 0:
            ax2.scatter(
                proj2[events, 0], proj2[events, 1],
                c="#E53935", s=35, zorder=5,
                edgecolors="#B71C1C", linewidths=0.5,
                label="Events",
            )
        cb2 = fig.colorbar(sc2, ax=ax2, fraction=0.04, pad=0.02)
        cb2.set_label("Time step"); cb2.ax.tick_params(labelsize=7)
        ax2.set_xlabel("PC 1"); ax2.set_ylabel("PC 2")
        ax2.set_title("Reservoir State Manifold (PCA 2D)")
        ax2.legend(frameon=True, facecolor="white", edgecolor="#dddddd")

        # ── Panel 3: Score dynamics ───────────────────────
        ax3 = axes[0, 2]
        _style(ax3)
        steps = [m["step"] for m in metrics]
        bs_   = [m["black_score"] for m in metrics]
        ws_   = [m["white_score"] for m in metrics]
        ax3.plot(steps, bs_, color="#1a1a1a", lw=1.6, label="Black (neural)")
        ax3.plot(steps, ws_, color="#888888", lw=1.6, ls="--", label="White (MCTS)")
        ax3.axhline(0.5, color="#cccccc", lw=0.9, ls=":", zorder=1)
        ax3.fill_between(steps, bs_, ws_,
            where=[b > w for b, w in zip(bs_, ws_)],
            alpha=0.12, color="#1a1a1a", label="Black advantage")
        ax3.fill_between(steps, bs_, ws_,
            where=[w >= b for b, w in zip(bs_, ws_)],
            alpha=0.12, color="#888888", label="White advantage")
        ax3.set_ylim(0, 1)
        ax3.set_xlabel("Game step")
        ax3.set_ylabel("Score (Chinese rules + komi)")
        ax3.set_title("Score Dynamics")
        ax3.legend(frameon=True, facecolor="white", edgecolor="#dddddd")

        # ── Panel 4: Pre-clustering manifold ─────────────
        ax4 = axes[1, 0]
        _style(ax4)
        sc4 = ax4.scatter(
            proj2[:, 0], proj2[:, 1],
            c=np.arange(len(proj2)), cmap="viridis",
            s=4, alpha=0.5, zorder=2,
        )
        cb4 = fig.colorbar(sc4, ax=ax4, fraction=0.04, pad=0.02)
        cb4.set_label("Time step"); cb4.ax.tick_params(labelsize=7)
        ax4.set_xlabel("PC 1"); ax4.set_ylabel("PC 2")
        ax4.set_title("Pre-clustering Manifold\n(coloured by time)")

        # ── Panel 5: Post-clustering manifold + Pearson r ─
        ax5 = axes[1, 1]
        _style(ax5)
        cmap_c = plt.get_cmap("tab10", N_CLUSTERS)
        sc5 = ax5.scatter(
            proj2[:, 0], proj2[:, 1],
            c=cluster_labels, cmap=cmap_c,
            vmin=0, vmax=N_CLUSTERS - 1,
            s=4, alpha=0.6, zorder=2,
        )
        cb5 = fig.colorbar(sc5, ax=ax5, fraction=0.04, pad=0.02,
                           ticks=range(N_CLUSTERS))
        cb5.set_label("Cluster"); cb5.ax.tick_params(labelsize=7)
        # Mark cluster centres in 2-D projection
        centres_2d = PCA(n_components=2, random_state=42).fit(
            self.states).transform(km8.cluster_centers_)
        ax5.scatter(centres_2d[:, 0], centres_2d[:, 1],
                    marker="X", s=80, c="white",
                    edgecolors="#333333", linewidths=0.8, zorder=6,
                    label="Cluster centre")
        ax5.set_xlabel("PC 1"); ax5.set_ylabel("PC 2")
        ax5.set_title(
            f"Post-clustering Manifold  (K={N_CLUSTERS})\n"
            f"Pearson r  PC1↔adv={r1:+.3f} (p={p1:.3f})   "
            f"PC2↔adv={r2:+.3f} (p={p2:.3f})",
            fontsize=9,
        )
        ax5.legend(frameon=True, facecolor="white", edgecolor="#dddddd",
                   fontsize=7)

    # ══════════════════════════════════════════════════════
    #  Figure 2: Final Board (standalone, high-res, Arial)
    # ══════════════════════════════════════════════════════
    def _plot_board(self, occupancy: np.ndarray):
        """
        Standalone final board figure.
        Features:
          - Wooden board background with outer border and star points
          - Black move frequency heatmap overlay (red tint, variable alpha)
          - Stone rendering with subtle highlight circles
          - Column (A-J, skipping I) and row (1-9) coordinate labels
          - Final score and winner annotation
          - Legend (Arial)
        Saved as Board_HHMMSS.png at 300 dpi.
        """
        n      = BOARD_SIZE
        bs_f, ws_f = self.env.score_chinese()
        winner = "Black wins" if bs_f > ws_f else "White wins"

        fig, ax = plt.subplots(figsize=(8, 8), facecolor="#F5EDD6")
        ax.set_facecolor("#C8A96E")
        ax.set_aspect("equal")

        # Grid lines
        for i in range(n):
            ax.plot([0, n-1], [i, i], color="#5a3e1b", lw=0.9, alpha=0.7, zorder=1)
            ax.plot([i, i], [0, n-1], color="#5a3e1b", lw=0.9, alpha=0.7, zorder=1)

        # Bold outer border
        for xs, ys in [([0,n-1],[0,0]), ([0,n-1],[n-1,n-1]),
                       ([0,0],[0,n-1]), ([n-1,n-1],[0,n-1])]:
            ax.plot(xs, ys, color="#3a2510", lw=2.2, zorder=2)

        # Star points (hoshi)
        stars = [(2,2),(2,4),(2,6),(4,2),(4,4),(4,6),(6,2),(6,4),(6,6)]
        for sr, sc in stars:
            ax.add_patch(Circle((sc, n-1-sr), 0.13, color="#3a2510", zorder=3))

        # Frequency heatmap (Black moves)
        if occupancy.max() > 0:
            for r in range(n):
                for c in range(n):
                    if occupancy[r, c] > 0:
                        alpha = float(occupancy[r, c] / occupancy.max()) * 0.55
                        ax.add_patch(Circle(
                            (c, n-1-r), 0.44,
                            facecolor="#D32F2F", alpha=alpha, zorder=4,
                        ))

        # Stones with highlight
        for r in range(n):
            for c in range(n):
                v = int(self.env.board[r, c])
                if v == BLACK:
                    ax.add_patch(Circle(
                        (c, n-1-r), 0.44,
                        facecolor="#1a1a1a", edgecolor="#000000",
                        lw=0.8, zorder=6,
                    ))
                    # Specular highlight
                    ax.add_patch(Circle(
                        (c - 0.11, n-1-r + 0.11), 0.13,
                        facecolor="#666666", alpha=0.5, zorder=7,
                    ))
                elif v == WHITE:
                    ax.add_patch(Circle(
                        (c, n-1-r), 0.44,
                        facecolor="#f8f8f8", edgecolor="#333333",
                        lw=1.0, zorder=6,
                    ))
                    ax.add_patch(Circle(
                        (c - 0.11, n-1-r + 0.11), 0.13,
                        facecolor="#ffffff", alpha=0.7, zorder=7,
                    ))

        # Coordinate labels (Arial)
        col_labels = list("ABCDEFGHJ")    # standard Go notation (no I)
        for i in range(n):
            ax.text(i, -0.68, col_labels[i],
                    ha="center", va="center",
                    fontsize=9, color="#3a2510")
            ax.text(-0.68, i, str(i + 1),
                    ha="center", va="center",
                    fontsize=9, color="#3a2510")

        # Title (Arial via rcParams)
        ax.set_title(
            f"Final Board  —  {winner}\n"
            f"Black {bs_f:.3f}   White {ws_f:.3f}   (komi {KOMI})",
            fontsize=13, color="#1a1a1a", pad=14,
        )

        # Legend
        legend_elements = [
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='#1a1a1a', markersize=11,
                   label='Black (neural signal)'),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='#f8f8f8', markeredgecolor='#333333',
                   markersize=11, label='White (MCTS AI)'),
            Line2D([0],[0], marker='o', color='w',
                   markerfacecolor='#D32F2F', alpha=0.6,
                   markersize=11, label='Black move frequency'),
        ]
        ax.legend(
            handles       = legend_elements,
            loc           = "upper right",
            bbox_to_anchor= (1.01, 1.01),
            frameon       = True,
            facecolor     = "#F5EDD6",
            edgecolor     = "#ccbbaa",
            fontsize      = 8,
        )

        ax.set_xlim(-0.95, n - 0.05)
        ax.set_ylim(-0.95, n - 0.05)
        ax.axis("off")

        plt.tight_layout()

        ts   = datetime.now().strftime("%H%M%S")
        path = os.path.join(self.save_dir, f"Board_{ts}.png")
        plt.savefig(path, dpi=300, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), edgecolor="none")
        print(f"[Saved] {path}")
        plt.close("all")


# ══════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    f = filedialog.askopenfilename(
        title     = "Select neural data file (.mat)",
        filetypes = [("MATLAB files", "*.mat"), ("All files", "*.*")],
    )
    if f:
        system = NeuralGoSystem(f, mcts_iter=MCTS_ITER, lag=SS_LAG)
        system.run()
    else:
        print("No file selected. Exiting.")