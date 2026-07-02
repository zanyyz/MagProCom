import numpy as np
import matplotlib
matplotlib.use('Agg')
import scipy.io as sio
from sklearn.linear_model import Ridge, RidgeClassifierCV, RidgeCV, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from scipy.stats import pearsonr, sem
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from collections import deque
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import tkinter as tk
from tkinter import filedialog
from scipy.signal import find_peaks, savgol_filter
from scipy.stats import entropy
import os
import h5py
import sys 
import csv
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from sklearn.metrics import calinski_harabasz_score
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


_CLUSTER_COLORS = [
    '#6e6aaa',  
    '#5e8f92', 
    '#7ab0c8',  
    '#8585bc', 
    '#6e9e95',  
    '#78a8cc',  
    '#9898c8', 
    '#8ab4b0',  
    '#4a88b0', 
    '#7c78b8',  
]
CLUSTER_CMAP = ListedColormap(_CLUSTER_COLORS)


_MANIFOLD_COLORS = ['#cbc8e8', '#9898c8', '#8585bc', '#6e6aaa',
                    '#7ab0c8', '#78a8cc', '#4a88b0', '#2d6090']
MANIFOLD_CMAP = LinearSegmentedColormap.from_list('muted_purple_blue', _MANIFOLD_COLORS)


_HEAT_COLORS = ['#c0d8d5', '#8ab4b0', '#6e9e95', '#5e8f92', '#4a88b0', '#2d6090', '#1a4a72']
HEAT_CMAP = LinearSegmentedColormap.from_list('muted_teal_blue', _HEAT_COLORS)


_C_PURPLE    = '#6e6aaa'
_C_BLUE      = '#4a88b0'
_C_GREEN     = '#5e8f92'
_C_LAVENDER  = '#9898c8'
_C_TEAL      = '#6e9e95'
_C_SOFTBLUE  = '#7ab0c8'
_C_MIDPURPLE = '#8585bc'
_C_GRAY      = '#9898a8'   

# ==========================================
#1. Go
# ==========================================
class GoEnvironment:
    def __init__(self):
        self.board_size = 7
        self.num_points = 49 
        self.white_stones = [(0,1), (0,2), (0,3), (1,1), (1,3), (2,1), (2,2)]
        self.black_stones = [(0,5), (1,4), (2,3), (3,3), (3,2), (3,1)] 
        self.target_moves = [(0, 4), (1, 2), (1, 0)]
        self.target_indices = [r * 7 + c for r, c in self.target_moves]
        self.frames_per_move = 20 

    def get_target_matrix(self, t_frames):
        Y_Go = np.zeros((t_frames, self.num_points))
        for i in range(t_frames):
            move_idx = (i // self.frames_per_move) % 3 
            target_pos = self.target_indices[move_idx]
            Y_Go[i, target_pos] = 1 
        return Y_Go

    def plot_go_heatmap(self, move_idx, predicted_probs, ax):
        for i in range(self.board_size):
            ax.plot([0, self.board_size-1], [i, i], 'k-', linewidth=1)
            ax.plot([i, i], [0, self.board_size-1], 'k-', linewidth=1)
            
        for r, c in self.white_stones:
            ax.add_patch(Circle((c, r), 0.4, facecolor='white', edgecolor='black', zorder=3))
        for r, c in self.black_stones:
            ax.add_patch(Circle((c, r), 0.4, facecolor='black', edgecolor='black', zorder=3))
            
        heatmap_2d = predicted_probs.reshape((self.board_size, self.board_size))
        im = ax.imshow(heatmap_2d, cmap='Reds', alpha=0.7, 
                       extent=[-0.5, self.board_size-0.5, self.board_size-0.5, -0.5], zorder=2)
        
        target_r, target_c = self.target_moves[move_idx]
        ax.plot(target_c, target_r, 'g*', markersize=15, zorder=4, label='Target')
        
        pred_idx = np.argmax(predicted_probs)
        pred_r, pred_c = pred_idx // 7, pred_idx % 7
        ax.plot(pred_c, pred_r, 'bo', markersize=12, fillstyle='none', markeredgewidth=2, zorder=5, label='Predicted')
        
        ax.set_title(f"Move {move_idx + 1}")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_xlim(-0.5, self.board_size-0.5)
        ax.set_ylim(self.board_size-0.5, -0.5) 


# ==========================================
# 1.1 GO
# ==========================================
class GoGameDatabaseEnvironment:
    def __init__(self, target_sequence=None):
        if target_sequence is None:
            self.move_sequence = [25, 9, 7]
        else:
            self.move_sequence = target_sequence
        
    def get_targets_for_milestones(self, n_milestones):
        return [self.move_sequence[i % len(self.move_sequence)] for i in range(n_milestones)]

# ==========================================
# 2. maze
# ==========================================
class MazeEnvironment:
    def __init__(self, width=15, height=15):
        self.width = width // 2 * 2 + 1
        self.height = height // 2 * 2 + 1
        self.maze = np.ones((self.height, self.width), dtype=int) 
        self.path = []
        self.start_pos = (1, 1)
        self.end_pos = (self.height - 2, self.width - 2)
        self.frames_per_step = 1 
        self.frames_per_run = 150 

    def generate_maze(self):
        self.maze[self.start_pos] = 0
        walls = [(self.start_pos[0], self.start_pos[1] + 1), (self.start_pos[0] + 1, self.start_pos[1])]
        while walls:
            wall_idx = np.random.randint(0, len(walls))
            r, c = walls.pop(wall_idx)
            cells = []
            if r > 1 and self.maze[r-1, c] == 0: cells.append((r-1, c, r+1, c))
            if r < self.height-2 and self.maze[r+1, c] == 0: cells.append((r+1, c, r-1, c))
            if c > 1 and self.maze[r, c-1] == 0: cells.append((r, c-1, r, c+1))
            if c < self.width-2 and self.maze[r, c+1] == 0: cells.append((r, c+1, r, c-1))
            if len(cells) == 1:
                self.maze[r, c] = 0
                next_r, next_c = cells[0][2], cells[0][3]
                self.maze[next_r, next_c] = 0
                if next_r > 1 and self.maze[next_r-1, next_c] == 1: walls.append((next_r-1, next_c))
                if next_r < self.height-2 and self.maze[next_r+1, next_c] == 1: walls.append((next_r+1, next_c))
                if next_c > 1 and self.maze[next_r, next_c-1] == 1: walls.append((next_r, next_c-1))
                if next_c < self.width-2 and self.maze[next_r, next_c+1] == 1: walls.append((next_r, next_c+1))
        self.maze[self.end_pos] = 0

    def solve_maze(self):
        queue = deque([(self.start_pos, [self.start_pos])])
        visited = set([self.start_pos])
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)] 
        while queue:
            (r, c), current_path = queue.popleft()
            if (r, c) == self.end_pos:
                self.path = current_path
                return self.path
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < self.height and 0 <= nc < self.width:
                    if self.maze[nr, nc] == 0 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), current_path + [(nr, nc)]))
        return []

    def get_target_trajectory(self, t_frames):
        if not self.path: self.solve_maze()
        num_steps = len(self.path) - 1
        if num_steps <= 0: return np.zeros((t_frames, 2))
        
        if self.frames_per_run < num_steps: self.frames_per_run = num_steps

        discrete_vx, discrete_vy = [], []
        for i in range(num_steps):
            r1, c1 = self.path[i]
            r2, c2 = self.path[i+1]
            discrete_vx.append(c2 - c1) 
            discrete_vy.append(r2 - r1) 
            
        self.frames_per_step = self.frames_per_run // num_steps
        single_run_Y = np.zeros((self.frames_per_run, 2))
        
        for i in range(num_steps):
            start_idx = i * self.frames_per_step
            end_idx = (i + 1) * self.frames_per_step if i < num_steps - 1 else self.frames_per_run
            single_run_Y[start_idx:end_idx, 0] = discrete_vx[i]
            single_run_Y[start_idx:end_idx, 1] = discrete_vy[i]
            
        single_run_Y[:, 0] = gaussian_filter1d(single_run_Y[:, 0], sigma=self.frames_per_step/4)
        single_run_Y[:, 1] = gaussian_filter1d(single_run_Y[:, 1], sigma=self.frames_per_step/4)
        
        runs = int(np.ceil(t_frames / self.frames_per_run))
        tiled_Y = np.tile(single_run_Y, (runs, 1))
        Y_Maze = tiled_Y[:t_frames, :] 
        
        return Y_Maze

    def simulate_physics(self, predicted_velocities):
        dt = 1.0 / self.frames_per_step
        T = predicted_velocities.shape[0]
        
        trials_x = []
        trials_y = []
        
        curr_x, curr_y = float(self.start_pos[1]), float(self.start_pos[0])
        curr_trial_x = [curr_x]
        curr_trial_y = [curr_y]
        
        for t in range(T):
            if t > 0 and t % self.frames_per_run == 0:
                trials_x.append(np.array(curr_trial_x))
                trials_y.append(np.array(curr_trial_y))
                curr_x, curr_y = float(self.start_pos[1]), float(self.start_pos[0])
                curr_trial_x = [curr_x]
                curr_trial_y = [curr_y]

            vx, vy = predicted_velocities[t, 0], predicted_velocities[t, 1]
            next_x, next_y = curr_x + vx * dt, curr_y + vy * dt
            grid_x, grid_y = int(round(next_x)), int(round(next_y))
            grid_x = max(0, min(self.width - 1, grid_x))
            grid_y = max(0, min(self.height - 1, grid_y))
            
            if self.maze[grid_y, grid_x] == 1: 
                if self.maze[int(round(curr_y)), grid_x] == 0: curr_x = next_x
                elif self.maze[grid_y, int(round(curr_x))] == 0: curr_y = next_y
            else:
                curr_x, curr_y = next_x, next_y
                
            curr_trial_x.append(curr_x)
            curr_trial_y.append(curr_y)
            
        trials_x.append(np.array(curr_trial_x))
        trials_y.append(np.array(curr_trial_y))
        
        return trials_x, trials_y

# ==========================================
# 3.BioReservoirDecoder
# ==========================================
class BioReservoirDecoder:
    def __init__(self, mat_file_path, data_key='norm_df', transpose=False, delay_steps=20):

        self.mat_file_path = mat_file_path
        self.delay_steps = delay_steps
        
        print(f"\n loading: {os.path.basename(mat_file_path)} ...")
        print(f"{self.delay_steps}")

        try:
            try:
                mat_data = sio.loadmat(mat_file_path)
                if data_key not in mat_data:
                    available_keys = [k for k in mat_data.keys() if not k.startswith('__')]
                    raise KeyError(f"cannot find '{data_key}'！The variables contained in this file are: {available_keys}")
                raw_X = mat_data[data_key]
                print("The reading was successful using scipy ")
            except NotImplementedError:
                print(" MATLAB v7.3 ，it is automatically switching to h5py reading...")
                with h5py.File(mat_file_path, 'r') as f:
                    if data_key not in f.keys():
                        available_keys = list(f.keys())
                        raise KeyError(f"cannot find '{data_key}'！The variables contained in this file are: {available_keys}")
                    raw_X = np.array(f[data_key])
                print(" The reading was successful using h5py ")
        except Exception as e:
            print(f"\n Loading completely failed! Detailed reasons: {e}")
            print(">>> Automatically switch to random simulation data demonstration...")
            raw_X = np.random.rand(28, 2000) 
        

        print(f"original data dimension: {raw_X.shape[0]} × {raw_X.shape[1]}")
        
        if transpose:
            print("Data being transposed: Time point × cell → Cell × time point...")
            raw_X = raw_X.T
        else:
            print("ok")
        

        self.N_cells, self.T_frames = raw_X.shape
        print(f"Confirm the data format: {self.N_cells} cells × {self.T_frames} times")
        

        self.X_raw = raw_X.T  
        scaler = StandardScaler()
        self.X_raw = scaler.fit_transform(self.X_raw)  
        self.X_raw = np.tanh(self.X_raw)  
        
        pca = PCA(n_components=0.95) 
        self.X_raw = pca.fit_transform(self.X_raw)
        
        self.T_frames, self.N_cells_pc = self.X_raw.shape
        print(f"PCA: {self.T_frames} times × {self.N_cells_pc} cells(PC)")
        

        self.maze_env = MazeEnvironment(width=15, height=15)
        self.maze_env.generate_maze()
        self.go_env = GoEnvironment()
        
        self.X = self._prepare_features()
        self.Y_Go, self.Y_Maze = self._prepare_targets()

    def _prepare_features(self, use_pca=False, use_delay=True, use_tanh=False):
        X = self.X_raw.copy()
        if use_tanh: X = np.tanh(X)
        if use_pca:
            pca = PCA(n_components=0.95)
            X = pca.fit_transform(X)
        
        if not use_delay:
            return X
            
        T, N = X.shape
        X_delay = np.zeros((T, N * (self.delay_steps + 1)))
        for d in range(self.delay_steps + 1):
            if d == 0: X_delay[:, d*N:(d+1)*N] = X
            else:
                X_delay[d:, d*N:(d+1)*N] = X[:-d]
                X_delay[:d, d*N:(d+1)*N] = X[0]
        return X_delay

    def _prepare_targets(self, use_smoothing=True):
        Y_Go = self.go_env.get_target_matrix(t_frames=self.T_frames)
        if use_smoothing:
            for i in range(self.T_frames):
                board = Y_Go[i].reshape((7, 7))
                Y_Go[i] = gaussian_filter(board, sigma=0.5).flatten()
                if np.sum(Y_Go[i]) > 0: Y_Go[i] /= np.sum(Y_Go[i])
        return Y_Go, self.maze_env.get_target_trajectory(t_frames=self.T_frames)

    def compute_cluster_entropy(self, cluster_labels):
        counts = np.bincount(cluster_labels)
        probs = counts / np.sum(counts)
        H_cluster = entropy(probs, base=2)
        return H_cluster

    def compute_temporal_entropy(self, X_pca, bins=30):
        pc1 = X_pca[:, 0]
        hist, _ = np.histogram(pc1, bins=bins)
        probs = hist / np.sum(hist)
        H_temp = entropy(probs + 1e-12, base=2)
        return H_temp

    def compute_trajectory_entropy(self, X_2d, bins=20):
        dx = np.diff(X_2d[:, 0])
        dy = np.diff(X_2d[:, 1])
        angles = np.arctan2(dy, dx)
        hist, _ = np.histogram(angles, bins=bins, range=(-np.pi, np.pi))
        probs = hist / np.sum(hist)
        H_traj = entropy(probs + 1e-12, base=2)
        return H_traj

    @staticmethod
    def _safe_pearson(x, y):
        if np.std(x) < 1e-5 or np.std(y) < 1e-5:
            return 0.0
        r, _ = pearsonr(x, y)
        return r if not np.isnan(r) else 0.0

    # ==========================================
    # 4. Neural dynamics flow field analysis
    # ==========================================
    def compute_flow_field(self, X_2d):

        dX = np.diff(X_2d, axis=0)
        X = X_2d[:-1]
        return X, dX

    def plot_flow_field(self, X_2d, stride=10, cluster_labels=None, task_labels=None, save_fig=True):

        X, dX = self.compute_flow_field(X_2d)
        has_color = (cluster_labels is not None or task_labels is not None)
        
        fig, axes = plt.subplots(1, 2 if has_color else 1, figsize=(12, 5))
        
        if has_color:
            ax1, ax2 = axes
        else:
            ax1 = axes
            
        ax1.quiver(
            X[::stride, 0],
            X[::stride, 1],
            dX[::stride, 0],
            dX[::stride, 1],
            angles='xy',
            scale_units='xy',
            scale=1.5,
            alpha=0.7,
            width=0.003,
            color=_C_BLUE
        )
        ax1.scatter(X_2d[:, 0], X_2d[:, 1], s=1, alpha=0.3, color=_C_LAVENDER)
        ax1.set_title("Neural Flow Field (Raw)", fontsize=12)
        ax1.set_xlabel("PC1 / t-SNE 1")
        ax1.set_ylabel("PC2 / t-SNE 2")
        
        if cluster_labels is not None:
            scatter = ax2.scatter(X_2d[:, 0], X_2d[:, 1], 
                                 c=cluster_labels, cmap=CLUSTER_CMAP, 
                                 s=12.5, alpha=0.6)
            ax2.quiver(
                X[::stride, 0],
                X[::stride, 1],
                dX[::stride, 0],
                dX[::stride, 1],
                angles='xy',
                scale_units='xy',
                scale=1.5,
                alpha=0.7,
                width=0.003,
                color=_C_PURPLE
            )
            ax2.set_title("Neural Flow Field (Colored by Cluster)", fontsize=12)
            plt.colorbar(scatter, ax=ax2)
        elif task_labels is not None:
            scatter = ax2.scatter(X_2d[:, 0], X_2d[:, 1], 
                                 c=task_labels, cmap=MANIFOLD_CMAP, 
                                 s=12.5, alpha=0.6)
            ax2.quiver(
                X[::stride, 0],
                X[::stride, 1],
                dX[::stride, 0],
                dX[::stride, 1],
                angles='xy',
                scale_units='xy',
                scale=1.5,
                alpha=0.7,
                width=0.003,
                color=_C_PURPLE
            )
            ax2.set_title("Neural Flow Field (Colored by Task)", fontsize=12)
            plt.colorbar(scatter, ax=ax2)
        
        plt.tight_layout()
        
        base_dir = os.path.dirname(self.mat_file_path)
        flow_path = os.path.join(base_dir, "Neural_Flow_Field.png")
        plt.savefig(flow_path, dpi=300, bbox_inches='tight')
        print(f"{flow_path}")
        plt.close('all')
        return X, dX

    # ==========================================
    # 5. Cluster state transition matrix
    # ==========================================
    def compute_transition_matrix(self, cluster_labels):

        K = np.max(cluster_labels) + 1
        T_counts = np.zeros((K, K))
        
        for t in range(len(cluster_labels) - 1):
            i = cluster_labels[t]
            j = cluster_labels[t + 1]
            T_counts[i, j] += 1
        
        T = T_counts / (T_counts.sum(axis=1, keepdims=True) + 1e-12)
        return T, K

    def plot_transition_matrix(self, T, K, title="Cluster State Transition Matrix", save_fig=True):

        fig, ax = plt.subplots(figsize=(10, 8))
        
        im = ax.imshow(T, cmap=HEAT_CMAP, interpolation='nearest', vmin=0, vmax=1)
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Transition Probability', fontsize=12)
        
        for i in range(K):
            for j in range(K):
                if T[i, j] > 0.05:
                    ax.text(j, i, f'{T[i, j]:.2f}',
                           ha="center", va="center", color="white", fontsize=8)
        
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Next State (t+1)", fontsize=12)
        ax.set_ylabel("Current State (t)", fontsize=12)
        ax.set_xticks(np.arange(K))
        ax.set_yticks(np.arange(K))
        ax.set_xticklabels([f'C{i}' for i in range(K)])
        ax.set_yticklabels([f'C{i}' for i in range(K)])
        
        plt.tight_layout()
        
        base_dir = os.path.dirname(self.mat_file_path)
        trans_path = os.path.join(base_dir, "Transition_Matrix.png")
        plt.savefig(trans_path, dpi=300, bbox_inches='tight')
        print(f"{trans_path}")
        plt.close('all')
        
        T_flat = T.flatten()
        T_entropy = entropy(T_flat[T_flat > 0] + 1e-12, base=2)
        print(f"\n Transition matrix statistics:")
        print(f" H(T): {T_entropy:.3f} bits")
        return T_entropy

    def compute_stationary_distribution(self, T):

        K = T.shape[0]
        eigvals, eigvecs = np.linalg.eig(T.T)
        stationary = np.real(eigvecs[:, np.isclose(eigvals, 1)])
        if stationary.size > 0:
            stationary = stationary.flatten()
            stationary = stationary / np.sum(stationary)
            return stationary
        else:
            return np.ones(K) / K

    # ==========================================
    # 6
    # ==========================================
    def build_evidence_chain(self, cluster_labels, X_pca, X_2d):

        print("\n" + "="*60)
        print("="*60)
        
        evidence = {}
        
        print("\n 1: Information compression effect")
        print("-" * 40)
        
        H_raw = self.compute_temporal_entropy(self.X_raw, bins=30)
        print(f"  H_raw: {H_raw:.4f} bits")
        
        H_pca = self.compute_temporal_entropy(X_pca, bins=30)
        compression_ratio_pca = (H_raw - H_pca) / H_raw * 100
        print(f"  PCA  H_pca: {H_pca:.4f} bits ({compression_ratio_pca:.1f}%)")
        
        H_cluster = self.compute_cluster_entropy(cluster_labels)
        compression_ratio_cluster = (H_raw - H_cluster) / H_raw * 100
        print(f"  Cluster  H_cluster: {H_cluster:.4f} bits ( {compression_ratio_cluster:.1f}%)")
        
        evidence['H_raw'] = H_raw
        evidence['H_pca'] = H_pca
        evidence['H_cluster'] = H_cluster
        evidence['compression_pca'] = compression_ratio_pca
        evidence['compression_cluster'] = compression_ratio_cluster
        
        print("\n 2. Dynamic stability")
        print("-" * 40)
        
        T_matrix, K = self.compute_transition_matrix(cluster_labels)
        H_transition = entropy(T_matrix.flatten()[T_matrix.flatten() > 0] + 1e-12, base=2)
        print(f"   H(transition): {H_transition:.4f} bits")
        
        self_transition_strength = np.mean(np.diag(T_matrix))
        print(f"  (stay probability): {self_transition_strength:.3f}")
        
        evidence['H_transition'] = H_transition
        evidence['self_transition'] = self_transition_strength
        evidence['T_matrix'] = T_matrix
        
        print("\n 3")
        print("-" * 40)
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X_raw, np.argmax(self.Y_Go, axis=1), test_size=0.3, random_state=42
        )
        model_raw = RidgeClassifierCV(alphas=np.logspace(-1, 4, 10))
        model_raw.fit(X_train, y_train)
        acc_raw = model_raw.score(X_test, y_test)
        print(f"  {acc_raw*100:.2f}%")
        
        X_train_pca, X_test_pca, _, _ = train_test_split(
            X_pca, np.argmax(self.Y_Go, axis=1), test_size=0.3, random_state=42
        )
        model_pca = RidgeClassifierCV(alphas=np.logspace(-1, 4, 10))
        model_pca.fit(X_train_pca, y_train)
        acc_pca = model_pca.score(X_test_pca, y_test)
        print(f"  PCA Spatial prediction accuracy: {acc_pca*100:.2f}% (improve {(acc_pca-acc_raw)*100:.2f}%)")
        
        encoder = OneHotEncoder(sparse_output=False)
        cluster_features = encoder.fit_transform(cluster_labels.reshape(-1, 1))
        
        X_train_cluster, X_test_cluster, _, _ = train_test_split(
            cluster_features, np.argmax(self.Y_Go, axis=1), test_size=0.3, random_state=42
        )
        model_cluster = RidgeClassifierCV(alphas=np.logspace(-1, 4, 10))
        model_cluster.fit(X_train_cluster, y_train)
        acc_cluster = model_cluster.score(X_test_cluster, y_test)
        print(f" {acc_cluster*100:.2f}% (improve: {(acc_cluster-acc_raw)*100:.2f}%)")
        
        evidence['acc_raw'] = acc_raw
        evidence['acc_pca'] = acc_pca
        evidence['acc_cluster'] = acc_cluster
        
        evidence['num_collapse_points'] = 0

        print("\n" + "="*60)
        print("Evidence")
        print("="*60)
        print(f"Core results:")
        print(f"  1. Information compression: Original → Cluster entropy has been reduced {(1 - H_cluster/H_raw)*100:.1f}%")
        print(f"  2. Prediction improvement: Cluster vs. original accuracy improvement{(acc_cluster-acc_raw)*100:.1f}%")
        print(f"  3. Dynamic structuring: Transfer entropy {H_transition:.3f} bits， {self_transition_strength:.3f}")

        return evidence

    def plot_evidence_summary(self, evidence, save_fig=True):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        ax1 = axes[0, 0]
        categories = ['Raw', 'PCA', 'Cluster']
        entropies = [evidence['H_raw'], evidence['H_pca'], evidence['H_cluster']]
        bars = ax1.bar(categories, entropies, color=[_C_GRAY, _C_BLUE, _C_PURPLE])
        ax1.set_ylabel('Entropy (bits)', fontsize=12)
        ax1.set_title('Information Compression Effect', fontsize=12)
        for bar, h in zip(bars, entropies):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=10)
        
        ax2 = axes[0, 1]
        accuracies = [evidence['acc_raw']*100, evidence['acc_pca']*100, evidence['acc_cluster']*100]
        bars = ax2.bar(categories, accuracies, color=[_C_GRAY, _C_BLUE, _C_PURPLE])
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Prediction Performance', fontsize=12)
        ax2.axhline(y=evidence['acc_cluster']*100, color=_C_MIDPURPLE, linestyle='--', alpha=0.5)
        for bar, acc in zip(bars, accuracies):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=10)
        
        ax3 = axes[1, 0]
        T_matrix = evidence.get('T_matrix')
        if T_matrix is not None:
            im = ax3.imshow(T_matrix, cmap=HEAT_CMAP, vmin=0, vmax=1)
            ax3.set_title(f'Transition Matrix (H_trans={evidence["H_transition"]:.3f})', fontsize=12)
            ax3.set_xlabel('Next State')
            ax3.set_ylabel('Current State')
            plt.colorbar(im, ax=ax3)
        
        ax4 = axes[1, 1]
        improvements = [
            (evidence['compression_cluster'], 'Compression'),
            ((evidence['acc_cluster'] - evidence['acc_raw'])*100, 'Prediction'),
            ((1 - evidence['H_transition']/evidence['H_raw'])*100, 'Structure')
        ]
        names = [imp[1] for imp in improvements]
        values = [imp[0] for imp in improvements]
        bars = ax4.bar(names, values, color=[_C_PURPLE, _C_BLUE, _C_GREEN])
        ax4.set_ylabel('Improvement (%)', fontsize=12)
        ax4.set_title('Evidence Summary', fontsize=12)
        for bar, val in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=10)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.suptitle("Evidence Chain: Clustering → Information Compression → Predictability", 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        base_dir = os.path.dirname(self.mat_file_path)
        evidence_path = os.path.join(base_dir, "Evidence_Chain_Summary.png")
        plt.savefig(evidence_path, dpi=300, bbox_inches='tight')
        print(f"{evidence_path}")
        plt.close('all')

    def plot_representation_analysis(self):

        print("\n Raw manifold + Heatmap...")

        X = self.X_raw.copy()
        
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        go_labels = np.argmax(self.Y_Go, axis=1)
        
        trial_len = self.go_env.frames_per_move * 3
        T = self.X_raw.shape[0]
        num_trials = T // trial_len
        
        X_trim = X[:num_trials * trial_len]
        X_trials = X_trim.reshape(num_trials, trial_len, -1)
        X_mean = np.mean(X_trials, axis=0)
        
        fig = plt.figure(figsize=(15, 10))
        
        ax1 = plt.subplot(2, 2, 1)
        sc1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=np.arange(len(X_pca)),
                         cmap=MANIFOLD_CMAP, s=5, alpha=0.6)
        ax1.set_title("Raw Neural Manifold (PCA)")
        plt.colorbar(sc1, ax=ax1)
        
        ax2 = plt.subplot(2, 2, 2)
        sc2 = ax2.scatter(X_tsne[:, 0], X_tsne[:, 1], c=go_labels,
                         cmap=CLUSTER_CMAP, s=2, alpha=0.8)
        ax2.set_title("Neural Manifold (t-SNE, Task Colored)")
        plt.colorbar(sc2, ax=ax2)
        
        ax3 = plt.subplot(2, 2, 3)
        sc3 = ax3.scatter(X_pca[:, 0], X_pca[:, 1], c=go_labels,
                         cmap=CLUSTER_CMAP, s=2, alpha=0.8)
        ax3.set_title("Task Structure in Neural Space")
        
        ax4 = plt.subplot(2, 2, 4)
        im = ax4.imshow(X_mean.T, aspect='auto', cmap=HEAT_CMAP)
        ax4.set_title("Trial-Averaged Neural Activity")
        ax4.set_xlabel("Time")
        ax4.set_ylabel("Neurons")
        plt.colorbar(im, ax=ax4)
        
        plt.suptitle("Raw Neural Dynamics → Latent Structure", fontsize=16)
        plt.tight_layout()
        
        base_dir = os.path.dirname(self.mat_file_path)
        raw_fig_path = os.path.join(base_dir, "Raw_Neural_Representation.png")
        plt.savefig(raw_fig_path, dpi=300, bbox_inches='tight')
        print(f"{raw_fig_path}")
        plt.close('all')
        
        return {
            "X_pca": X_pca,
            "X_tsne": X_tsne,
            "trial_mean": X_mean,
            "labels": go_labels
        }
    
    def compare_delay_impact(self):
        print("\n Analyze the influence of time delay on neural state clustering...")
        
        X_no_delay = self.X_raw  
        
        X_with_delay = self.X
        
        #  t-SNE
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        results = {}
        
        for i, (data, title) in enumerate([(X_no_delay, "Without Delay"), (X_with_delay, "With Delay")]):
            print(f" {title}...")

            pca_feat = PCA(n_components=min(50, data.shape[1])).fit_transform(data)
            tsne_feat = TSNE(n_components=2, random_state=42, perplexity=30).fit_transform(pca_feat)
            labels = KMeans(n_clusters=6, random_state=42, n_init=10).fit_predict(pca_feat)
            
            t_mat, K = self.compute_transition_matrix(labels)
            trans_entropy = entropy(t_mat.flatten()[t_mat.flatten() > 0] + 1e-12, base=2)
            
            try:
                sil_score = silhouette_score(pca_feat, labels)
            except:
                sil_score = np.nan
            

            results[title] = {
                'trans_entropy': trans_entropy,
                'silhouette': sil_score,
                'labels': labels,
                'X_2d': tsne_feat
            }
            
            sc = axes[i].scatter(tsne_feat[:, 0], tsne_feat[:, 1], 
                                c=labels, cmap=CLUSTER_CMAP, s=10, alpha=0.6)
            axes[i].set_title(f"{title}\nTrans Entropy: {trans_entropy:.3f} bits | Sil: {sil_score:.3f}")
            axes[i].set_xlabel("t-SNE 1")
            axes[i].set_ylabel("t-SNE 2")
            plt.colorbar(sc, ax=axes[i])
            
            print(f" {trans_entropy:.4f} bits")
            print(f" {sil_score:.4f}")
        
        plt.suptitle("Impact of Time Delay on Neural State Clustering", fontsize=14)
        plt.tight_layout()
        
        base_dir = os.path.dirname(self.mat_file_path)
        delay_path = os.path.join(base_dir, "Delay_Impact_Analysis.png")
        plt.savefig(delay_path, dpi=300, bbox_inches='tight')
        print(f" {delay_path}")
        plt.close('all')
        
        print("\n" + "="*50)
        print("delay affects summary:")
        print("="*50)
        print(f"Without delay - transfer entropy: {results['Without Delay']['trans_entropy']:.4f}")
        print(f"With delay-transfer entropy:: {results['With Delay']['trans_entropy']:.4f}")
        
        entropy_change = results['With Delay']['trans_entropy'] - results['Without Delay']['trans_entropy']
        if entropy_change < 0:
            print(f"Delay reduces the transfer entropy {abs(entropy_change):.4f} bits ")
        else:
            print(f"Delay increases the transfer entropy {entropy_change:.4f} bits ")
        
        return results

    def state_selective_neuron_analysis(
            self,
            cluster_labels,
            X_original=None,
            save_fig=True):

        print("\n State-selective Neuron Analysis")

        if X_original is None:
            X_original = self.X_raw

        unique_states = np.unique(cluster_labels)
        K = len(unique_states)
        N = X_original.shape[1]

        state_activity = np.zeros((N, K))
        state_counts = np.zeros(K, dtype=int)
        for ki, k in enumerate(unique_states):
            idx = np.where(cluster_labels == k)[0]
            if len(idx) > 0:
                state_activity[:, ki] = np.mean(X_original[idx], axis=0)
                state_counts[ki] = len(idx)

        # ---- (Amax - Aothers) / (Amax + Aothers) ----

        act_shifted = state_activity - np.min(state_activity)

        A_max = np.max(act_shifted, axis=1)                     # (N,)
        pref_tmp = np.argmax(act_shifted, axis=1)               # (N,) 0-based


        activity_others = act_shifted.copy()                    # (N, K)
        activity_others[np.arange(N), pref_tmp] = np.nan
        if K > 1:
            A_others = np.nanmean(activity_others, axis=1)      # (N,)
        else:
            A_others = np.zeros(N)


        denom = A_max + A_others
        selectivity_index = np.where(
            denom > 1e-12,
            (A_max - A_others) / denom,
            0.0
        )


        preferred_state = pref_tmp    
        sort_order = np.argsort(preferred_state)
        state_activity_sorted = state_activity[sort_order]

        # ----（SI > mean + 1*std）----
        si_thresh = np.mean(selectivity_index) + np.std(selectivity_index)
        high_sel_mask = selectivity_index > si_thresh
        n_high = int(np.sum(high_sel_mask))

        print(f"   {N}")
        print(f"  state numbers K: {K}")
        print(f"  SI : (Amax − Aothers) / (Amax + Aothers)")
        print(f"  mean SI: {np.mean(selectivity_index):.3f} ± {np.std(selectivity_index):.3f}")
        print(f"  SI range: [{np.min(selectivity_index):.3f}, {np.max(selectivity_index):.3f}]")
        print(f"  Highly selective neurons (SI > {si_thresh:.3f}): {n_high} / {N} ({n_high/N*100:.1f}%)")

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))


        im0 = axes[0].imshow(
            state_activity_sorted,
            aspect='auto',
            cmap=HEAT_CMAP,
            interpolation='nearest'
        )
        axes[0].set_title("Neuron–State Activity Matrix\n(neurons sorted by preferred state)",
                          fontsize=11)
        axes[0].set_xlabel("State ID", fontsize=10)
        axes[0].set_ylabel("Neuron (sorted)", fontsize=10)
        axes[0].set_xticks(np.arange(K))
        axes[0].set_xticklabels([f'C{k}' for k in unique_states])
        plt.colorbar(im0, ax=axes[0], label="Mean Activity")


        axes[1].hist(selectivity_index, bins=25, color=_C_PURPLE, edgecolor='white', linewidth=0.5)
        axes[1].axvline(np.mean(selectivity_index), color=_C_BLUE, linestyle='--', linewidth=2,
                        label=f'Mean={np.mean(selectivity_index):.3f}')
        axes[1].axvline(si_thresh, color=_C_GREEN, linestyle=':', linewidth=2,
                        label=f'Threshold={si_thresh:.3f}')
        axes[1].set_title("Selectivity Index Distribution", fontsize=11)
        axes[1].set_xlabel("SI = (Amax − Aothers) / (Amax + Aothers)", fontsize=10)
        axes[1].set_ylabel("Neuron Count", fontsize=10)
        axes[1].legend(fontsize=9)
        axes[1].text(0.97, 0.95, f"High-sel: {n_high}/{N} ({n_high/N*100:.0f}%)",
                     transform=axes[1].transAxes, ha='right', va='top', fontsize=10,
                     bbox=dict(boxstyle='round', facecolor=_C_LAVENDER, alpha=0.6))


        pref_counts = np.bincount(preferred_state[high_sel_mask], minlength=K) if n_high > 0 else np.zeros(K, int)
        axes[2].bar(np.arange(K), pref_counts, color=_CLUSTER_COLORS[:K], edgecolor='white')
        axes[2].set_title("High-Selectivity Neurons\nper Preferred State", fontsize=11)
        axes[2].set_xlabel("State ID", fontsize=10)
        axes[2].set_ylabel("Count", fontsize=10)
        axes[2].set_xticks(np.arange(K))
        axes[2].set_xticklabels([f'C{k}' for k in unique_states])
        for xi, cnt in enumerate(pref_counts):
            if cnt > 0:
                axes[2].text(xi, cnt + 0.2, str(cnt), ha='center', va='bottom', fontsize=9)

        plt.suptitle(
            "State-selective Neuron Analysis\n"
            "Magnetic modulation reshapes state-selective tuning of local neuronal populations",
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()

        if save_fig:
            base_dir = os.path.dirname(self.mat_file_path)
            fig_path = os.path.join(base_dir, "State_Selective_Neurons.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"  {fig_path}")
        plt.close('all')


        state_results = {
            "state_activity": state_activity,
            "state_activity_sorted": state_activity_sorted,
            "preferred_state": preferred_state,
            "selectivity_index": selectivity_index,
            "high_sel_mask": high_sel_mask,
            "n_high_selective": n_high,
            "si_threshold": si_thresh,
            "unique_states": unique_states,
            "state_counts": state_counts
        }

        self._export_si_excel(state_results)

        self._plot_si_barchart(state_results)

        return state_results

    # ----------------------------------------------------------------
    def _export_si_excel(self, sr):

        try:
            import openpyxl
            from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
            from openpyxl.utils import get_column_letter
        except ImportError:
            print("  pip install openpyxl")
            return

        print("  Saving Excel...")

        N              = len(sr["selectivity_index"])
        K              = len(sr["unique_states"])
        unique_states  = sr["unique_states"]
        si             = sr["selectivity_index"]
        pref_idx       = sr["preferred_state"]
        state_activity = sr["state_activity"]   # (N, K)
        high_mask      = sr["high_sel_mask"]
        si_thresh      = sr["si_threshold"]

        wb = openpyxl.Workbook()

        # ──────────────────────────────────────────────────
        hdr_fill = PatternFill("solid", start_color="1F4E79")
        hdr_font = Font(bold=True, color="FFFFFF", name="Arial", size=10)
        hi_fill  = PatternFill("solid", start_color="E2EFDA")  
        body_fnt = Font(name="Arial", size=9)
        ctr      = Alignment(horizontal="center", vertical="center")
        thin     = Side(style="thin", color="BFBFBF")
        bdr      = Border(left=thin, right=thin, top=thin, bottom=thin)

        def _hdr(ws, r, c, txt):
            cell = ws.cell(row=r, column=c, value=txt)
            cell.fill = hdr_fill; cell.font = hdr_font
            cell.alignment = ctr; cell.border = bdr

        def _cell(ws, r, c, val, fmt=None, fill=None):
            cell = ws.cell(row=r, column=c, value=val)
            cell.font = body_fnt; cell.alignment = ctr; cell.border = bdr
            if fmt:   cell.number_format = fmt
            if fill:  cell.fill = fill

        # ══════════════════════════════════════════════
        # Sheet 1：Neuron_SI
        # Neuron_ID | Preferred_State | SI | High_Sel | MeanAct_C0…CK
        # ══════════════════════════════════════════════
        ws1 = wb.active
        ws1.title = "Neuron_SI"
        ws1.freeze_panes = "A2"

        fixed_hdrs = ["Neuron_ID", "Preferred_State",
                      "SI = (Amax-Aothers)/(Amax+Aothers)",
                      f"High_Selective (SI>{si_thresh:.3f})"]
        act_hdrs   = [f"MeanAct_C{s}" for s in unique_states]
        all_hdrs   = fixed_hdrs + act_hdrs

        ws1.row_dimensions[1].height = 28
        for ci, h in enumerate(all_hdrs, 1):
            _hdr(ws1, 1, ci, h)

        for ni in range(N):
            r    = ni + 2
            is_h = bool(high_mask[ni])
            rf   = hi_fill if is_h else None
            _cell(ws1, r, 1, ni,                       "#,##0",  rf)
            _cell(ws1, r, 2, int(unique_states[pref_idx[ni]]), "#,##0", rf)
            _cell(ws1, r, 3, float(si[ni]),            "0.0000", rf)
            _cell(ws1, r, 4, "Yes" if is_h else "No",  None,     rf)
            for ki in range(K):
                _cell(ws1, r, 5 + ki, float(state_activity[ni, ki]), "0.0000", rf)

        col_ws = [12, 18, 30, 28] + [14] * K
        for ci, w in enumerate(col_ws, 1):
            ws1.column_dimensions[get_column_letter(ci)].width = w

        sr_row = N + 2
        ws1.cell(row=sr_row, column=1, value="MEAN").font = Font(bold=True, name="Arial", size=9)
        ws1.cell(row=sr_row, column=3,
                 value=f"=AVERAGE(C2:C{N+1})").number_format = "0.0000"
        ws1.cell(row=sr_row, column=4,
                 value=f'=COUNTIF(D2:D{N+1},"Yes")')
        ws1.cell(row=sr_row, column=1).fill = PatternFill("solid", start_color="FFF2CC")

        # ══════════════════════════════════════════════
        # Sheet 2：State_Summary
        # State | n_Preferred | Pct% | n_HighSel | Mean_SI | Max_SI | Min_SI
        # ══════════════════════════════════════════════
        ws2 = wb.create_sheet("State_Summary")
        ws2.freeze_panes = "A2"

        s2_hdrs = ["State_Label", "n_Preferred_Neurons",
                   "Pct_of_Total (%)",
                   f"n_HighSel (SI>{si_thresh:.3f})",
                   "Mean_SI", "Max_SI", "Min_SI",
                   "Mean_Activity_in_State"]
        ws2.row_dimensions[1].height = 28
        for ci, h in enumerate(s2_hdrs, 1):
            _hdr(ws2, 1, ci, h)

        row_colors = ["DCE6F1","E2EFDA","FFF2CC","FCE4D6","F2CEEF","DDEBF7"]
        for ki, ks in enumerate(unique_states):
            mask  = (pref_idx == ki)
            n_p   = int(np.sum(mask))
            n_hs  = int(np.sum(mask & high_mask))
            m_si  = float(np.mean(si[mask]))  if n_p > 0 else 0.0
            mx_si = float(np.max(si[mask]))   if n_p > 0 else 0.0
            mn_si = float(np.min(si[mask]))   if n_p > 0 else 0.0
            m_act = float(np.mean(state_activity[mask, ki])) if n_p > 0 else 0.0
            pct   = n_p / N * 100
            rf2   = PatternFill("solid", start_color=row_colors[ki % len(row_colors)])
            r = ki + 2
            _cell(ws2, r, 1, int(ks),  "#,##0",  rf2)
            _cell(ws2, r, 2, n_p,      "#,##0",  rf2)
            _cell(ws2, r, 3, pct,      "0.0",    rf2)
            _cell(ws2, r, 4, n_hs,     "#,##0",  rf2)
            _cell(ws2, r, 5, m_si,     "0.0000", rf2)
            _cell(ws2, r, 6, mx_si,    "0.0000", rf2)
            _cell(ws2, r, 7, mn_si,    "0.0000", rf2)
            _cell(ws2, r, 8, m_act,    "0.0000", rf2)

        last = K + 2
        ws2.cell(row=last, column=1, value="TOTAL").font = Font(bold=True, name="Arial")
        ws2.cell(row=last, column=2, value=f"=SUM(B2:B{K+1})")
        ws2.cell(row=last, column=3,
                 value=f"=SUM(C2:C{K+1})").number_format = "0.0"
        ws2.cell(row=last, column=1).fill = PatternFill("solid", start_color="FFF2CC")

        col_ws2 = [14, 22, 20, 26, 12, 12, 12, 24]
        for ci, w in enumerate(col_ws2, 1):
            ws2.column_dimensions[get_column_letter(ci)].width = w

        base_dir  = os.path.dirname(self.mat_file_path)
        xlsx_path = os.path.join(base_dir, "State_Selective_Neuron_Analysis.xlsx")
        wb.save(xlsx_path)
        print(f"  Excel was saved: {xlsx_path}")

    # ----------------------------------------------------------------
    def _plot_si_barchart(self, sr):
 
        print("  SI")

        si             = sr["selectivity_index"]
        pref_idx       = sr["preferred_state"]
        unique_states  = sr["unique_states"]
        high_mask      = sr["high_sel_mask"]
        si_thresh      = sr["si_threshold"]
        K              = len(unique_states)
        N              = len(si)
        x              = np.arange(K)
        xlabels        = [f"C{s}" for s in unique_states]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        mean_si_per_state = []
        std_si_per_state  = []
        for ki in range(K):
            mask = (pref_idx == ki)
            vals = si[mask] if np.sum(mask) > 0 else np.array([0.0])
            mean_si_per_state.append(np.mean(vals))
            std_si_per_state.append(np.std(vals))

        mean_si_arr = np.array(mean_si_per_state)
        std_si_arr  = np.array(std_si_per_state)

        bars1 = axes[0].bar(x, mean_si_arr,
                            color=_CLUSTER_COLORS[:K],
                            edgecolor='white', linewidth=0.8,
                            yerr=std_si_arr, capsize=5,
                            error_kw=dict(ecolor=_C_GRAY, lw=1.5))
        axes[0].axhline(np.mean(si), color=_C_BLUE, ls='--', lw=1.5,
                        label=f'Overall mean={np.mean(si):.3f}')
        axes[0].axhline(si_thresh, color=_C_GREEN, ls=':', lw=1.5,
                        label=f'Threshold={si_thresh:.3f}')
        axes[0].set_xticks(x); axes[0].set_xticklabels(xlabels)
        axes[0].set_title("Mean SI per Preferred State\n(error bars = ±1 std)", fontsize=11)
        axes[0].set_xlabel("State", fontsize=10)
        axes[0].set_ylabel("Selectivity Index", fontsize=10)
        axes[0].legend(fontsize=8)
        for xi, (mv, sv) in enumerate(zip(mean_si_arr, std_si_arr)):
            axes[0].text(xi, mv + sv + 0.01, f'{mv:.3f}',
                         ha='center', va='bottom', fontsize=8, color=_C_GRAY)


        n_pref   = np.array([int(np.sum(pref_idx == ki)) for ki in range(K)])
        n_high_s = np.array([int(np.sum((pref_idx == ki) & high_mask)) for ki in range(K)])
        n_low_s  = n_pref - n_high_s

        axes[1].bar(x, n_low_s,
                    color=_CLUSTER_COLORS[:K],
                    edgecolor='white', linewidth=0.8,
                    label='Non high-sel')
        axes[1].bar(x, n_high_s,
                    bottom=n_low_s,
                    color=_C_GREEN,
                    edgecolor='white', linewidth=0.8,
                    alpha=0.85, label=f'High-sel (SI>{si_thresh:.2f})')
        axes[1].set_xticks(x); axes[1].set_xticklabels(xlabels)
        axes[1].set_title("Preferred Neuron Count per State\n(green = high-selectivity)", fontsize=11)
        axes[1].set_xlabel("State", fontsize=10)
        axes[1].set_ylabel("Neuron Count", fontsize=10)
        axes[1].legend(fontsize=8)
        for xi, (tot, hi) in enumerate(zip(n_pref, n_high_s)):
            axes[1].text(xi, tot + 0.3, str(tot),
                         ha='center', va='bottom', fontsize=9)


        bins_si   = [(-np.inf, 0), (0, 0.33), (0.33, 0.66), (0.66, np.inf)]
        bin_names = ["SI < 0", "0–0.33", "0.33–0.66", "SI > 0.66"]
        bin_cols  = [_C_GRAY, _CLUSTER_COLORS[2], _CLUSTER_COLORS[0], _C_GREEN]

        counts = np.zeros((K, len(bins_si)))
        for ki in range(K):
            vals = si[pref_idx == ki] if np.sum(pref_idx == ki) > 0 else np.array([])
            for bi, (lo, hi_b) in enumerate(bins_si):
                counts[ki, bi] = np.sum((vals > lo) & (vals <= hi_b))

        row_sum = counts.sum(axis=1, keepdims=True)
        pcts    = np.where(row_sum > 0, counts / row_sum * 100, 0)

        bottoms = np.zeros(K)
        for bi, (bname, bcol) in enumerate(zip(bin_names, bin_cols)):
            axes[2].bar(x, pcts[:, bi], bottom=bottoms,
                        color=bcol, edgecolor='white', linewidth=0.8,
                        label=bname, alpha=0.9)
            for xi in range(K):
                if pcts[xi, bi] > 5:
                    axes[2].text(xi, bottoms[xi] + pcts[xi, bi] / 2,
                                 f'{pcts[xi, bi]:.0f}%',
                                 ha='center', va='center', fontsize=7,
                                 color='white' if bcol != _C_GRAY else 'black')
            bottoms += pcts[:, bi]

        axes[2].set_xticks(x); axes[2].set_xticklabels(xlabels)
        axes[2].set_ylim(0, 100)
        axes[2].set_title("SI Tier Distribution per Preferred State\n(stacked %)", fontsize=11)
        axes[2].set_xlabel("State", fontsize=10)
        axes[2].set_ylabel("Percentage (%)", fontsize=10)
        axes[2].legend(fontsize=8, loc='upper right')

        # ── title ────────────────────────────────────────────────
        plt.suptitle(
            "State Selectivity Index – Bar Chart Summary\n"
            f"N={N} neurons  |  K={K} states  |  "
            f"Overall SI: {np.mean(si):.3f}±{np.std(si):.3f}  |  "
            f"High-sel (SI>{si_thresh:.2f}): {int(np.sum(high_mask))}/{N}",
            fontsize=12, fontweight='bold'
        )
        plt.tight_layout()

        base_dir  = os.path.dirname(self.mat_file_path)
        fig_path  = os.path.join(base_dir, "State_Selective_SI_Barchart.png")
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        plt.close('all')
        print(f"  SI: {fig_path}")

    def detect_neuronal_assemblies(
            self,
            state_results,
            n_assemblies=4,
            save_fig=True):

        print("\n Detect Neuronal Assemblies")

        neuron_features = state_results["state_activity"]   # (N, K)
        N, K = neuron_features.shape
        selectivity_index = state_results["selectivity_index"]
        unique_states = state_results.get("unique_states", np.arange(K))

        # ---- KMeans→ assembly ----
        n_assemblies = min(n_assemblies, N)
        km = KMeans(n_clusters=n_assemblies, random_state=42, n_init=20)
        assembly_labels = km.fit_predict(neuron_features)

        try:
            sil = silhouette_score(neuron_features, assembly_labels) if n_assemblies > 1 else np.nan
        except Exception:
            sil = np.nan

        for a in range(n_assemblies):
            members = np.sum(assembly_labels == a)
            mean_si = np.mean(selectivity_index[assembly_labels == a]) if members > 0 else 0
            print(f"  Assembly {a}: {members} , mean SI={mean_si:.3f}")
        if not np.isnan(sil):
            print(f"  Assembly silhouette coefficient: {sil:.4f}")


        sort_order = np.argsort(assembly_labels)
        neuron_sorted = neuron_features[sort_order]
        asm_sorted = assembly_labels[sort_order]

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))


        im = axes[0].imshow(neuron_sorted, aspect='auto', cmap=HEAT_CMAP, interpolation='nearest')
        axes[0].set_title("Assembly-sorted Neuron–State Matrix", fontsize=11)
        axes[0].set_xlabel("State ID", fontsize=10)
        axes[0].set_ylabel("Neuron (assembly-sorted)", fontsize=10)
        axes[0].set_xticks(np.arange(K))
        axes[0].set_xticklabels([f'C{s}' for s in unique_states])
        plt.colorbar(im, ax=axes[0], label="Mean Activity")

        boundaries = np.where(np.diff(asm_sorted) != 0)[0] + 1
        for b in boundaries:
            axes[0].axhline(b - 0.5, color='white', linewidth=1.5, linestyle='--')


        for a in range(n_assemblies):
            mask = assembly_labels == a
            if np.sum(mask) == 0:
                continue
            mean_resp = np.mean(neuron_features[mask], axis=0)
            axes[1].plot(np.arange(K), mean_resp,
                         color=_CLUSTER_COLORS[a % len(_CLUSTER_COLORS)],
                         marker='o', linewidth=2, markersize=6,
                         label=f'Asm {a} (n={np.sum(mask)})')
        axes[1].set_title("Assembly Mean Tuning Curves", fontsize=11)
        axes[1].set_xlabel("State ID", fontsize=10)
        axes[1].set_ylabel("Mean Activity", fontsize=10)
        axes[1].set_xticks(np.arange(K))
        axes[1].set_xticklabels([f'C{s}' for s in unique_states])
        axes[1].legend(fontsize=8)
        sil_str = f"{sil:.3f}" if not np.isnan(sil) else "N/A"
        axes[1].text(0.97, 0.02, f"Silhouette={sil_str}",
                     transform=axes[1].transAxes, ha='right', va='bottom', fontsize=9,
                     bbox=dict(boxstyle='round', facecolor=_C_LAVENDER, alpha=0.6))


        sizes = [np.sum(assembly_labels == a) for a in range(n_assemblies)]
        colors_pie = [_CLUSTER_COLORS[a % len(_CLUSTER_COLORS)] for a in range(n_assemblies)]
        wedges, texts, autotexts = axes[2].pie(
            sizes,
            labels=[f'Asm {a}' for a in range(n_assemblies)],
            colors=colors_pie,
            autopct='%1.0f%%',
            startangle=90,
            wedgeprops=dict(edgecolor='white', linewidth=1.2)
        )
        for at in autotexts:
            at.set_fontsize(9)
        axes[2].set_title("Assembly Size Distribution", fontsize=11)

        plt.suptitle(
            "Neuronal Assembly Detection\n"
            "Magnetic modulation reorganizes neuronal assemblies",
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()

        if save_fig:
            base_dir = os.path.dirname(self.mat_file_path)
            fig_path = os.path.join(base_dir, "Neuron_Assemblies.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"   {fig_path}")
        plt.close('all')

        return {
            "assembly_labels": assembly_labels,
            "n_assemblies": n_assemblies,
            "assembly_silhouette": sil,
            "assembly_sizes": sizes,
            "assembly_centroids": km.cluster_centers_
        }

    def assembly_dynamics_analysis(
            self,
            cluster_labels,
            assembly_results,
            X_original=None,
            save_fig=True):

        print("\n Assembly Dynamics Analysis")

        if X_original is None:
            X_original = self.X_raw

        assembly_labels = assembly_results["assembly_labels"]
        n_assemblies = assembly_results["n_assemblies"]
        unique_states = np.unique(cluster_labels)
        K_state = len(unique_states)
        K_ass = n_assemblies

        dynamics = np.zeros((K_state, K_ass))
        for si, s in enumerate(unique_states):
            idx = np.where(cluster_labels == s)[0]
            if len(idx) == 0:
                continue
            state_mean = np.mean(X_original[idx], axis=0)   # (N,)
            for a in range(K_ass):
                neurons = np.where(assembly_labels == a)[0]
                if len(neurons) > 0:
                    dynamics[si, a] = np.mean(state_mean[neurons])


        assembly_timecourse = np.zeros((X_original.shape[0], K_ass))
        for a in range(K_ass):
            neurons = np.where(assembly_labels == a)[0]
            if len(neurons) > 0:
                assembly_timecourse[:, a] = np.mean(X_original[:, neurons], axis=1)

        dominant_assembly = np.argmax(assembly_timecourse, axis=1)

        T_asm, _ = self.compute_transition_matrix(dominant_assembly)
        asm_trans_entropy = entropy(
            T_asm.flatten()[T_asm.flatten() > 0] + 1e-12, base=2
        )
        T_state, _ = self.compute_transition_matrix(cluster_labels)
        state_trans_entropy = entropy(
            T_state.flatten()[T_state.flatten() > 0] + 1e-12, base=2
        )

        print(f"  State transition entropy (cluster):   {state_trans_entropy:.4f} bits")
        print(f"  State Transition Entropy(assembly):  {asm_trans_entropy:.4f} bits")
        print(f"  Assembly range (dynamics range): "
              f"{np.max(dynamics) - np.min(dynamics):.4f}")


        fig, axes = plt.subplots(1, 3, figsize=(18, 6))


        im = axes[0].imshow(
            dynamics, cmap=HEAT_CMAP, aspect='auto',
            interpolation='nearest'
        )
        axes[0].set_title("Assembly Dynamics Matrix\n(mean activation per state)", fontsize=11)
        axes[0].set_xlabel("Assembly ID", fontsize=10)
        axes[0].set_ylabel("State ID", fontsize=10)
        axes[0].set_xticks(np.arange(K_ass))
        axes[0].set_xticklabels([f'Asm{a}' for a in range(K_ass)])
        axes[0].set_yticks(np.arange(K_state))
        axes[0].set_yticklabels([f'C{s}' for s in unique_states])
        plt.colorbar(im, ax=axes[0], label="Mean Activity")

        for si in range(K_state):
            for a in range(K_ass):
                axes[0].text(a, si, f"{dynamics[si, a]:.2f}",
                             ha='center', va='center', fontsize=7,
                             color='white' if dynamics[si, a] > np.median(dynamics) else _C_GRAY)

        show_len = min(600, X_original.shape[0])
        for a in range(K_ass):
            axes[1].plot(
                np.arange(show_len),
                assembly_timecourse[:show_len, a],
                color=_CLUSTER_COLORS[a % len(_CLUSTER_COLORS)],
                linewidth=1.2, alpha=0.85, label=f'Asm {a}'
            )
        axes[1].set_title("Assembly Activation Timecourse\n(first 600 frames)", fontsize=11)
        axes[1].set_xlabel("Time Frame", fontsize=10)
        axes[1].set_ylabel("Mean Activity", fontsize=10)
        axes[1].legend(fontsize=8, loc='upper right')

        labels_bar = ['State\n(cluster)', 'Assembly\n(dominant)']
        vals_bar = [state_trans_entropy, asm_trans_entropy]
        bar_colors = [_C_PURPLE, _C_BLUE]
        bars = axes[2].bar(labels_bar, vals_bar, color=bar_colors, edgecolor='white', width=0.5)
        axes[2].set_title("Transition Entropy Comparison\n(state vs assembly level)", fontsize=11)
        axes[2].set_ylabel("Shannon Entropy (bits)", fontsize=10)
        for bar, val in zip(bars, vals_bar):
            axes[2].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.02,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        axes[2].set_ylim(0, max(vals_bar) * 1.25)
        axes[2].text(
            0.5, 0.92,
            "Lower entropy → more structured transitions",
            transform=axes[2].transAxes, ha='center', va='top', fontsize=9,
            style='italic', color=_C_GRAY
        )

        plt.suptitle(
            "Assembly Dynamics Analysis\n"
            "Magnetic modulation reshapes the state-transition landscape "
            "of localized neuronal populations",
            fontsize=13, fontweight='bold'
        )
        plt.tight_layout()

        if save_fig:
            base_dir = os.path.dirname(self.mat_file_path)
            fig_path = os.path.join(base_dir, "Assembly_Dynamics.png")
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f" {fig_path}")
        plt.close('all')

        return {
            "dynamics_matrix": dynamics,
            "assembly_timecourse": assembly_timecourse,
            "dominant_assembly": dominant_assembly,
            "assembly_transition_entropy": asm_trans_entropy,
            "state_transition_entropy": state_trans_entropy
        }

    def representation_structure_analysis(self, n_clusters=6, method="kmeans"):
        print("\n clustering + manifold...")

        X = self.X.copy()
        
        print("manifold embedding (PCA + t-SNE)...")
        pca = PCA(n_components=50)
        X_pca = pca.fit_transform(X)
        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_2d = tsne.fit_transform(X_pca)
        
        print(f"Cluster analysis: {method}")
        if method == "kmeans":
            cluster_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = cluster_model.fit_predict(X_pca)
        elif method == "gmm":
            cluster_model = GaussianMixture(n_components=n_clusters, random_state=42)
            cluster_labels = cluster_model.fit_predict(X_pca)
        else:
            raise ValueError("method must be 'kmeans' or 'gmm'")
        
        sil_score = np.nan
        ch_score = np.nan
        try:
            sil_score = silhouette_score(X_pca, cluster_labels)
        except Exception as e:
            print(f"cannot calculate silhouette score: {e}")
        try:
            ch_score = calinski_harabasz_score(X_pca, cluster_labels)
        except Exception as e:
            print(f"cannot calculate Calinski-Harabasz score: {e}")
        
        cluster_entropy = self.compute_cluster_entropy(cluster_labels)
        temporal_entropy = self.compute_temporal_entropy(X_pca)
        trajectory_entropy = self.compute_trajectory_entropy(X_2d)
        
        print(f"silhouette score: {sil_score:.4f}")
        print(f"Calinski-Harabasz index: {ch_score:.2f}")
        print(f"Cluster entropy: {cluster_entropy:.4f}")
        print(f"Temporal entropy: {temporal_entropy:.4f}")
        print(f"Trajectory entropy: {trajectory_entropy:.4f}")
        
        go_labels = np.argmax(self.Y_Go, axis=1)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        scatter = axes[0, 0].scatter(X_2d[:, 0], X_2d[:, 1], c=cluster_labels, 
                                     cmap=CLUSTER_CMAP, s=3, alpha=0.7)
        axes[0, 0].set_title("Neural Manifold + Clustering")
        plt.colorbar(scatter, ax=axes[0, 0])
        
        scatter2 = axes[0, 1].scatter(X_2d[:, 0], X_2d[:, 1], c=go_labels, 
                                      cmap=MANIFOLD_CMAP, s=3, alpha=0.7)
        axes[0, 1].set_title("Neural Manifold Colored by Task")
        plt.colorbar(scatter2, ax=axes[0, 1])
        
        X_flow, dX_flow = self.compute_flow_field(X_2d)
        axes[1, 0].quiver(X_flow[::10, 0], X_flow[::10, 1], 
                          dX_flow[::10, 0], dX_flow[::10, 1],
                          angles='xy', scale_units='xy', scale=1.5, alpha=0.6,
                          color=_C_BLUE)
        axes[1, 0].scatter(X_2d[:, 0], X_2d[:, 1], s=1, alpha=0.3, color=_C_LAVENDER)
        axes[1, 0].set_title("Neural Flow Field")
        
        T_matrix, K = self.compute_transition_matrix(cluster_labels)
        im = axes[1, 1].imshow(T_matrix, cmap=HEAT_CMAP, vmin=0, vmax=1)
        axes[1, 1].set_title(f"Transition Matrix (K={K})")
        plt.colorbar(im, ax=axes[1, 1])
        
        sil_str = f"{sil_score:.3f}" if not np.isnan(sil_score) else "N/A"
        ch_str = f"{ch_score:.1f}" if not np.isnan(ch_score) else "N/A"
        plt.suptitle(f"Neural Dynamics Analysis | Sil={sil_str} | CH={ch_str} | Hc={cluster_entropy:.2f}", 
                    fontsize=13)
        
        plt.tight_layout()
        base_dir = os.path.dirname(self.mat_file_path)
        manifold_path = os.path.join(base_dir, "Neural_Dynamics_Full_Analysis.png")
        plt.savefig(manifold_path, dpi=300, bbox_inches='tight')
        print(f" {manifold_path}")
        plt.close('all')
        
        evidence = self.build_evidence_chain(cluster_labels, X_pca, X_2d)
        evidence['cluster_labels'] = cluster_labels
        self.plot_evidence_summary(evidence)

        state_results = self.state_selective_neuron_analysis(
            cluster_labels
        )

        assembly_results = self.detect_neuronal_assemblies(
            state_results,
            n_assemblies=4
        )

        assembly_dynamics = self.assembly_dynamics_analysis(
            cluster_labels,
            assembly_results
        )

        self._cached_state_results = state_results
        self._cached_assembly_dynamics = assembly_dynamics
        
        return {
            "X_pca": X_pca,
            "X_2d": X_2d,
            "cluster_labels": cluster_labels,
            "silhouette": sil_score,
            "go_labels": go_labels,
            "evidence": evidence,
            "state_results": state_results,
            "assembly_results": assembly_results,
            "assembly_dynamics": assembly_dynamics,
            "H_cluster": evidence.get("H_cluster"),
            "T_matrix": evidence.get("T_matrix")
        }

    # ============================================================
    # Assembly Reorganization Analysis
    # ============================================================
    def assembly_reorganization_analysis(
            self,
            cluster_labels,
            assembly_results,
            X_original=None,
            window=100,
            n_segments=4,
            save_fig=True):
        print("\n" + "=" * 60)
        print(" Assembly Reorganization Analysis）")
        print("=" * 60)

        if X_original is None:
            X_original = self.X_raw

        assembly_labels = assembly_results["assembly_labels"]
        n_assemblies    = assembly_results["n_assemblies"]
        T, N            = X_original.shape
        base_dir        = os.path.dirname(self.mat_file_path)

        go_event_times  = [i * self.go_env.frames_per_move for i in range(3)]
        go_event_labels = ['Go-1', 'Go-2', 'Go-3']

        def _mark_events(ax, ymin=None, ymax=None):
            ylim = ax.get_ylim()
            y0 = ylim[0] if ymin is None else ymin
            y1 = ylim[1] if ymax is None else ymax
            for ei, et in enumerate(go_event_times):
                if et < T:
                    ax.axvline(et, color=_C_GRAY, lw=1.2, ls='--', alpha=0.7)
                    ax.text(et + T * 0.005, y0 + (y1 - y0) * 0.92,
                            go_event_labels[ei], fontsize=8, color=_C_GRAY, rotation=0)

        # ─────────────────────────────────────────────────────────────
        # Layer 1：dominant_assembly 
        # ─────────────────────────────────────────────────────────────
        print("\n[Layer 1] dominant_assembly Dominant authority schedule...")

        asm_tc = np.zeros((T, n_assemblies))
        for a in range(n_assemblies):
            members = np.where(assembly_labels == a)[0]
            if len(members) > 0:
                asm_tc[:, a] = np.mean(X_original[:, members], axis=1)
        dominant = np.argmax(asm_tc, axis=1)                       # (T,)

        n_win = T - window
        dominance_ts = np.zeros((n_win, n_assemblies))
        for t in range(n_win):
            seg = dominant[t:t + window]
            for a in range(n_assemblies):
                dominance_ts[t, a] = np.mean(seg == a)

        fig1, axes1 = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

        ax = axes1[0]
        ax.scatter(np.arange(T), dominant,
                   c=dominant, cmap=CLUSTER_CMAP,
                   s=3, alpha=0.7, vmin=0, vmax=n_assemblies - 1)
        ax.set_yticks(np.arange(n_assemblies))
        ax.set_yticklabels([f'Asm{a}' for a in range(n_assemblies)])
        ax.set_ylabel("Dominant Assembly", fontsize=10)
        ax.set_title("Layer 1 – Dominant Assembly per Frame", fontsize=11)
        ax.set_xlim(0, T)
        _mark_events(ax)

        ax = axes1[1]
        bottom = np.zeros(n_win)
        xs = np.arange(n_win) + window // 2
        for a in range(n_assemblies):
            ax.fill_between(xs, bottom, bottom + dominance_ts[:, a],
                            color=_CLUSTER_COLORS[a % len(_CLUSTER_COLORS)],
                            alpha=0.8, label=f'Asm{a}')
            bottom += dominance_ts[:, a]
        ax.set_ylim(0, 1)
        ax.set_ylabel(f"Dominance (window={window}fr)", fontsize=10)
        ax.set_xlabel("Time Frame", fontsize=10)
        ax.set_title("Sliding-window Dominance Proportion", fontsize=11)
        ax.set_xlim(xs[0], xs[-1])
        ax.legend(loc='upper right', fontsize=8, ncol=n_assemblies)
        _mark_events(ax, ymin=0, ymax=1)

        plt.suptitle("Assembly Reorganization – Layer 1: Dominance Dynamics",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        p1 = os.path.join(base_dir, "Assembly_Reorganization_L1_Dominance.png")
        plt.savefig(p1, dpi=300, bbox_inches='tight')
        plt.close('all')
        print(f"{p1}")

        # ─────────────────────────────────────────────────────────────
        # Layer 2：Assembly  cohesion
        # ─────────────────────────────────────────────────────────────
        print("\n[Layer 2] Assembly (cohesion)...")

        cohesion_ts = np.full((n_win, n_assemblies), np.nan)
        for a in range(n_assemblies):
            members = np.where(assembly_labels == a)[0]
            if len(members) < 2:
                print(f" Asm{a} member < 2, skip the coherence calculation")
                continue
            X_asm = X_original[:, members]          # (T, n_members)
            for t in range(n_win):
                seg = X_asm[t:t + window]           # (window, n_members)
                if seg.shape[0] < 2:
                    continue

                with np.errstate(invalid='ignore'):
                    C = np.corrcoef(seg.T)          # (n_members, n_members)
                idx_u = np.triu_indices(len(members), k=1)
                vals = C[idx_u]
                vals = vals[~np.isnan(vals)]
                cohesion_ts[t, a] = np.mean(vals) if len(vals) > 0 else np.nan

        fig2, ax2 = plt.subplots(figsize=(14, 5))
        xs = np.arange(n_win) + window // 2
        for a in range(n_assemblies):
            col = _CLUSTER_COLORS[a % len(_CLUSTER_COLORS)]
            y = cohesion_ts[:, a]
            ax2.plot(xs, y, color=col, lw=1.2, alpha=0.5)
            if np.sum(~np.isnan(y)) > 20:
                y_s = np.where(np.isnan(y), np.nanmean(y), y)
                wl = min(51, (len(y_s) // 2) * 2 + 1)
                y_sm = savgol_filter(y_s, window_length=wl, polyorder=3)
                ax2.plot(xs, y_sm, color=col, lw=2.2, label=f'Asm{a}')
        ax2.axhline(0, color='gray', lw=0.8, ls=':')
        ax2.set_xlabel("Time Frame", fontsize=10)
        ax2.set_ylabel("Mean Intra-Assembly Pearson r", fontsize=10)
        ax2.set_title("Layer 2 – Assembly Internal Cohesion over Time\n"
                      "(bold = Savitzky-Golay trend)", fontsize=11)
        ax2.legend(fontsize=9)
        ax2.set_xlim(xs[0], xs[-1])
        _mark_events(ax2)
        plt.tight_layout()
        p2 = os.path.join(base_dir, "Assembly_Reorganization_L2_Cohesion.png")
        plt.savefig(p2, dpi=300, bbox_inches='tight')
        plt.close('all')
        print(f" {p2}")

        # ─────────────────────────────────────────────────────────────
        # Layer 3: The difference before and after the transition matrix delta_T (State level, Assembly level)
        # ─────────────────────────────────────────────────────────────
        print("\n[Layer 3] The difference before and after the transition matrix(delta_T)...")

        mid = T // 2

        def _pad_tmat(M, K_full):
            if M.shape[0] == K_full:
                return M
            M_full = np.zeros((K_full, K_full))
            k = M.shape[0]
            M_full[:k, :k] = M
            return M_full

        K_state_full = len(np.unique(cluster_labels))
        T_state_e_raw, Ks = self.compute_transition_matrix(cluster_labels[:mid])
        T_state_l_raw, _  = self.compute_transition_matrix(cluster_labels[mid:])
        T_state_e  = _pad_tmat(T_state_e_raw, K_state_full)
        T_state_l  = _pad_tmat(T_state_l_raw, K_state_full)
        Ks         = K_state_full
        delta_state = T_state_l - T_state_e

        T_asm_e_raw, Ka_e = self.compute_transition_matrix(dominant[:mid])
        T_asm_l_raw, Ka_l = self.compute_transition_matrix(dominant[mid:])
        T_asm_e  = _pad_tmat(T_asm_e_raw, n_assemblies)
        T_asm_l  = _pad_tmat(T_asm_l_raw, n_assemblies)
        Ka       = n_assemblies
        delta_asm = T_asm_l - T_asm_e

        def _trans_entropy(Tmat):
            flat = Tmat.flatten()
            flat = flat[flat > 0]
            return entropy(flat + 1e-12, base=2)

        He_state = _trans_entropy(T_state_e)
        Hl_state = _trans_entropy(T_state_l)
        He_asm   = _trans_entropy(T_asm_e)
        Hl_asm   = _trans_entropy(T_asm_l)

        vmax_s = np.max(np.abs(delta_state)) + 1e-6
        vmax_a = np.max(np.abs(delta_asm))   + 1e-6

        fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))

        def _plot_tmat(ax, M, title, vmin=0, vmax=1, cmap=HEAT_CMAP, labels=None):
            im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect='auto')
            K_ = M.shape[0]
            lbs = labels if labels else [str(i) for i in range(K_)]
            ax.set_xticks(np.arange(K_)); ax.set_xticklabels(lbs, fontsize=8)
            ax.set_yticks(np.arange(K_)); ax.set_yticklabels(lbs, fontsize=8)
            ax.set_xlabel("Next", fontsize=9); ax.set_ylabel("Current", fontsize=9)
            ax.set_title(title, fontsize=10)
            plt.colorbar(im, ax=ax, shrink=0.8)
            for i in range(K_):
                for j in range(K_):
                    val = M[i, j]
                    if abs(val) > 0.04:
                        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                                fontsize=6, color='white' if abs(val) > vmax * 0.5 else 'black')

        state_lbs = [f'C{i}' for i in range(Ks)]
        asm_lbs   = [f'A{i}' for i in range(Ka)]
        div_cmap  = LinearSegmentedColormap.from_list('div', [_C_BLUE, 'white', _C_PURPLE])

        _plot_tmat(axes3[0, 0], T_state_e, f'State T-matrix (Early)\nH={He_state:.3f}bits', labels=state_lbs)
        _plot_tmat(axes3[0, 1], T_state_l, f'State T-matrix (Late)\nH={Hl_state:.3f}bits',  labels=state_lbs)
        _plot_tmat(axes3[0, 2], delta_state,
                   f'ΔT State (Late−Early)\nΔH={Hl_state-He_state:+.3f}bits',
                   vmin=-vmax_s, vmax=vmax_s, cmap=div_cmap, labels=state_lbs)

        _plot_tmat(axes3[1, 0], T_asm_e, f'Assembly T-matrix (Early)\nH={He_asm:.3f}bits', labels=asm_lbs)
        _plot_tmat(axes3[1, 1], T_asm_l, f'Assembly T-matrix (Late)\nH={Hl_asm:.3f}bits',  labels=asm_lbs)
        _plot_tmat(axes3[1, 2], delta_asm,
                   f'ΔT Assembly (Late−Early)\nΔH={Hl_asm-He_asm:+.3f}bits',
                   vmin=-vmax_a, vmax=vmax_a, cmap=div_cmap, labels=asm_lbs)

        plt.suptitle("Layer 3 – Transition Matrix Changes (Early vs Late Half)\n"
                     "Blue=decreased  White=unchanged  Purple=increased",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        p3 = os.path.join(base_dir, "Assembly_Reorganization_L3_TransitionDelta.png")
        plt.savefig(p3, dpi=300, bbox_inches='tight')
        plt.close('all')
        print(f" {p3}")

        # ─────────────────────────────────────────────────────────────
        # Layer 4：The time-segmented dynamics matrix Frobenius drift
        # ─────────────────────────────────────────────────────────────
        print(f"\n[Layer 4] The time segment dynamics matrix ({n_segments} segment) Frobenius drift...")

        seg_len = T // n_segments
        unique_states = np.unique(cluster_labels)
        K_state = len(unique_states)

        def _compute_dynamics_seg(cl_seg, X_seg, asm_lbs_arr):
            K  = len(np.unique(cl_seg))
            Ka = len(np.unique(asm_lbs_arr))
            dm = np.zeros((K, Ka))
            for si, s in enumerate(np.unique(cl_seg)):
                idx = np.where(cl_seg == s)[0]
                if len(idx) == 0:
                    continue
                sm = np.mean(X_seg[idx], axis=0)
                for a in range(Ka):
                    ns = np.where(asm_lbs_arr == a)[0]
                    if len(ns) > 0:
                        dm[si, a] = np.mean(sm[ns])
            return dm

        dyn_segs = []
        seg_labels_list = []
        for i in range(n_segments):
            i0 = i * seg_len
            i1 = (i + 1) * seg_len if i < n_segments - 1 else T
            cl_seg = cluster_labels[i0:i1]
            X_seg  = X_original[i0:i1]
            dm = _compute_dynamics_seg(cl_seg, X_seg, assembly_labels)
            dyn_segs.append(dm)
            seg_labels_list.append(f'Seg{i+1}\n[{i0}–{i1}]')

        frob_dists = []
        for i in range(n_segments - 1):
            d0, d1 = dyn_segs[i], dyn_segs[i + 1]
            r = min(d0.shape[0], d1.shape[0])
            c = min(d0.shape[1], d1.shape[1])
            fd = np.linalg.norm(d1[:r, :c] - d0[:r, :c], 'fro')
            frob_dists.append(fd)
            print(f"  Seg{i+1}→Seg{i+2} Frobenius distance: {fd:.4f}")

        all_vals = np.concatenate([dm.flatten() for dm in dyn_segs])
        vmin_d, vmax_d = np.percentile(all_vals, 2), np.percentile(all_vals, 98)

        fig4, axes4 = plt.subplots(2, n_segments, figsize=(4.5 * n_segments, 10))
        if n_segments == 1:
            axes4 = axes4.reshape(2, 1)

        for i, dm in enumerate(dyn_segs):
            ax = axes4[0, i]
            r_show = dm.shape[0]
            c_show = dm.shape[1]
            im = ax.imshow(dm, cmap=HEAT_CMAP, aspect='auto',
                           vmin=vmin_d, vmax=vmax_d,
                           interpolation='nearest')
            ax.set_title(seg_labels_list[i], fontsize=10)
            ax.set_xlabel("Assembly", fontsize=9)
            ax.set_ylabel("State", fontsize=9) if i == 0 else None
            ax.set_xticks(np.arange(c_show))
            ax.set_xticklabels([f'A{a}' for a in range(c_show)], fontsize=8)
            ax.set_yticks(np.arange(r_show))
            ax.set_yticklabels([f'C{s}' for s in range(r_show)], fontsize=8)
            plt.colorbar(im, ax=ax, shrink=0.7)
            for si in range(r_show):
                for ai in range(c_show):
                    ax.text(ai, si, f'{dm[si, ai]:.1f}',
                            ha='center', va='center', fontsize=6,
                            color='white' if dm[si, ai] > (vmin_d + vmax_d) / 2 else _C_GRAY)

        for i in range(n_segments):
            ax = axes4[1, i]
            if i == 0:
                ax.text(0.5, 0.5, 'Reference\n(Seg 1)',
                        ha='center', va='center', transform=ax.transAxes, fontsize=11)
                ax.axis('off')
                continue
            d0 = dyn_segs[0]
            d1 = dyn_segs[i]
            r = min(d0.shape[0], d1.shape[0])
            c = min(d0.shape[1], d1.shape[1])
            delta = d1[:r, :c] - d0[:r, :c]
            vm = np.max(np.abs(delta)) + 1e-6
            div_cmap2 = LinearSegmentedColormap.from_list('div2', [_C_BLUE, 'white', _C_PURPLE])
            im2 = ax.imshow(delta, cmap=div_cmap2, aspect='auto',
                            vmin=-vm, vmax=vm, interpolation='nearest')
            fd_label = f"Frob={np.linalg.norm(delta,'fro'):.3f}" if i < len(frob_dists) + 1 else ""
            ax.set_title(f'Δ(Seg{i+1} − Seg1)\n{fd_label}', fontsize=10)
            ax.set_xlabel("Assembly", fontsize=9)
            ax.set_xticks(np.arange(c)); ax.set_xticklabels([f'A{a}' for a in range(c)], fontsize=8)
            ax.set_yticks(np.arange(r)); ax.set_yticklabels([f'C{s}' for s in range(r)], fontsize=8)
            plt.colorbar(im2, ax=ax, shrink=0.7)

        plt.suptitle(f"Layer 4 – Segmented Dynamics Matrices & Drift\n"
                     f"Frobenius distances (adjacent): "
                     + "  →  ".join([f"{d:.3f}" for d in frob_dists]),
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        p4 = os.path.join(base_dir, "Assembly_Reorganization_L4_DynamicsDrift.png")
        plt.savefig(p4, dpi=300, bbox_inches='tight')
        plt.close('all')
        print(f"  {p4}")

        # ─────────────────────────────────────────────────────────────
        # Summary
        # ─────────────────────────────────────────────────────────────
        print("\n[Summary] ...")

        fig_s = plt.figure(figsize=(20, 14))
        gs = fig_s.add_gridspec(3, 4, hspace=0.48, wspace=0.38)

        # ---- Row 0: Layer 1 ----
        ax_dom = fig_s.add_subplot(gs[0, :2])
        ax_dom.scatter(np.arange(T), dominant,
                       c=dominant, cmap=CLUSTER_CMAP, s=2, alpha=0.6,
                       vmin=0, vmax=n_assemblies - 1)
        ax_dom.set_yticks(np.arange(n_assemblies))
        ax_dom.set_yticklabels([f'A{a}' for a in range(n_assemblies)], fontsize=8)
        ax_dom.set_title("L1 – Dominant Assembly per Frame", fontsize=10)
        ax_dom.set_xlim(0, T)
        _mark_events(ax_dom)

        ax_dom2 = fig_s.add_subplot(gs[0, 2:])
        xs_w = np.arange(n_win) + window // 2
        bot = np.zeros(n_win)
        for a in range(n_assemblies):
            ax_dom2.fill_between(xs_w, bot, bot + dominance_ts[:, a],
                                 color=_CLUSTER_COLORS[a % len(_CLUSTER_COLORS)],
                                 alpha=0.8, label=f'A{a}')
            bot += dominance_ts[:, a]
        ax_dom2.set_ylim(0, 1); ax_dom2.set_xlim(xs_w[0], xs_w[-1])
        ax_dom2.set_title(f"L1 – Sliding Dominance (w={window}fr)", fontsize=10)
        ax_dom2.legend(fontsize=7, ncol=n_assemblies, loc='upper right')
        _mark_events(ax_dom2, ymin=0, ymax=1)

        # ---- Row 1: Layer 2 + Layer 3 delta ----
        ax_coh = fig_s.add_subplot(gs[1, :2])
        xs_w2 = np.arange(n_win) + window // 2
        for a in range(n_assemblies):
            col = _CLUSTER_COLORS[a % len(_CLUSTER_COLORS)]
            y = cohesion_ts[:, a]
            if np.sum(~np.isnan(y)) > 20:
                y_f = np.where(np.isnan(y), np.nanmean(y), y)
                wl = min(51, (len(y_f) // 2) * 2 + 1)
                ax_coh.plot(xs_w2, savgol_filter(y_f, wl, 3), color=col, lw=2, label=f'A{a}')
        ax_coh.axhline(0, color='gray', lw=0.7, ls=':')
        ax_coh.set_title("L2 – Assembly Cohesion (Pearson r trend)", fontsize=10)
        ax_coh.legend(fontsize=7); ax_coh.set_xlim(xs_w2[0], xs_w2[-1])
        _mark_events(ax_coh)

        div_cm3 = LinearSegmentedColormap.from_list('dv3', [_C_BLUE, 'white', _C_PURPLE])
        ax_ds = fig_s.add_subplot(gs[1, 2])
        im_ds = ax_ds.imshow(delta_state, cmap=div_cm3, aspect='auto',
                             vmin=-vmax_s, vmax=vmax_s)
        ax_ds.set_title(f"L3 – ΔT State\nΔH={Hl_state-He_state:+.3f}b", fontsize=10)
        ax_ds.set_xticks(np.arange(Ks)); ax_ds.set_xticklabels([f'C{i}' for i in range(Ks)], fontsize=7)
        ax_ds.set_yticks(np.arange(Ks)); ax_ds.set_yticklabels([f'C{i}' for i in range(Ks)], fontsize=7)
        plt.colorbar(im_ds, ax=ax_ds, shrink=0.7)

        ax_da = fig_s.add_subplot(gs[1, 3])
        im_da = ax_da.imshow(delta_asm, cmap=div_cm3, aspect='auto',
                             vmin=-vmax_a, vmax=vmax_a)
        ax_da.set_title(f"L3 – ΔT Assembly\nΔH={Hl_asm-He_asm:+.3f}b", fontsize=10)
        ax_da.set_xticks(np.arange(Ka)); ax_da.set_xticklabels([f'A{i}' for i in range(Ka)], fontsize=7)
        ax_da.set_yticks(np.arange(Ka)); ax_da.set_yticklabels([f'A{i}' for i in range(Ka)], fontsize=7)
        plt.colorbar(im_da, ax=ax_da, shrink=0.7)

        # ---- Row 2: Layer 4 Frobenius 
        ax_frob = fig_s.add_subplot(gs[2, :2])
        seg_mids = [(i + 0.5) * seg_len for i in range(n_segments - 1)]
        ax_frob.plot(seg_mids, frob_dists,
                     color=_C_PURPLE, marker='o', lw=2.5, markersize=8)
        for xi, yi in zip(seg_mids, frob_dists):
            ax_frob.text(xi, yi + max(frob_dists) * 0.03,
                         f'{yi:.3f}', ha='center', fontsize=9, color=_C_PURPLE)
        ax_frob.set_xlabel("Time (segment midpoint)", fontsize=9)
        ax_frob.set_ylabel("Frobenius Distance\n(vs previous segment)", fontsize=9)
        ax_frob.set_title("L4 – Assembly Activation Landscape Drift", fontsize=10)
        ax_frob.set_xlim(0, T)
        _mark_events(ax_frob)

        ax_d1 = fig_s.add_subplot(gs[2, 2])
        im_d1 = ax_d1.imshow(dyn_segs[0], cmap=HEAT_CMAP, aspect='auto',
                              vmin=vmin_d, vmax=vmax_d)
        ax_d1.set_title("L4 – Dynamics Seg1 (ref)", fontsize=10)
        ax_d1.set_xlabel("Assembly", fontsize=8); ax_d1.set_ylabel("State", fontsize=8)
        plt.colorbar(im_d1, ax=ax_d1, shrink=0.7)

        ax_dn = fig_s.add_subplot(gs[2, 3])
        im_dn = ax_dn.imshow(dyn_segs[-1], cmap=HEAT_CMAP, aspect='auto',
                              vmin=vmin_d, vmax=vmax_d)
        ax_dn.set_title(f"L4 – Dynamics Seg{n_segments} (last)", fontsize=10)
        ax_dn.set_xlabel("Assembly", fontsize=8)
        plt.colorbar(im_dn, ax=ax_dn, shrink=0.7)

        fig_s.suptitle(
            "Assembly Reorganization – Four-Layer Summary\n"
            "Magnetic modulation reorganizes neuronal assemblies "
            "and reshapes the state-transition landscape",
            fontsize=13, fontweight='bold'
        )
        ps = os.path.join(base_dir, "Assembly_Reorganization_Summary.png")
        plt.savefig(ps, dpi=300, bbox_inches='tight')
        plt.close('all')
        print(f"   {ps}")

        # ─────────────────────────────────────────────────────────────
        # print
        # ─────────────────────────────────────────────────────────────
        print("\n" + "=" * 60)
        print(" Assembly Reorganization")
        print("=" * 60)
        early_dom = np.bincount(dominant[:mid], minlength=n_assemblies) / mid
        late_dom  = np.bincount(dominant[mid:], minlength=n_assemblies) / (T - mid)
        print("  Dominant Assembly proportion (first half → second half)")
        for a in range(n_assemblies):
            arrow = "↑" if late_dom[a] > early_dom[a] else "↓"
            print(f"    Asm{a}: {early_dom[a]*100:.1f}% → {late_dom[a]*100:.1f}% {arrow}")
        print(f"\n  State transfer entropy change: {He_state:.3f} → {Hl_state:.3f} ({Hl_state-He_state:+.3f}bits)")
        print(f"  Assembly transfer entropy change: {He_asm:.3f} → {Hl_asm:.3f} ({Hl_asm-He_asm:+.3f}bits)")
        print(f"\n  Frobenius : " + " → ".join([f"{d:.3f}" for d in frob_dists]))
        total_drift = sum(frob_dists)
        print(f"  {total_drift:.4f}")

        reorg_results = {
            "dominant_assembly":       dominant,
            "dominance_timeseries":    dominance_ts,
            "cohesion_timeseries":     cohesion_ts,
            "T_state_early":           T_state_e,
            "T_state_late":            T_state_l,
            "delta_T_state":           delta_state,
            "T_assembly_early":        T_asm_e,
            "T_assembly_late":         T_asm_l,
            "delta_T_assembly":        delta_asm,
            "state_entropy_early":     He_state,
            "state_entropy_late":      Hl_state,
            "assembly_entropy_early":  He_asm,
            "assembly_entropy_late":   Hl_asm,
            "dynamics_segments":       dyn_segs,
            "frobenius_distances":     frob_dists,
            "total_drift":             total_drift,
            "early_dominance":         early_dom,
            "late_dominance":          late_dom,
        }
        self._cached_reorg_results = reorg_results
        return reorg_results

    def plot_evidence_chain_flow(self, rep_results, struct_results):
        plt.figure(figsize=(18, 5))
        
        # 1.Raw Complexity
        plt.subplot(1, 4, 1)
        plt.bar(["Raw", "Parsed"], [rep_results.get('H_raw', 0), struct_results.get('H_cluster', 0)], 
                color=[_C_GRAY, _C_PURPLE])
        plt.title("1. Entropy Reduction\n(Information Compression)")
        plt.ylabel("Entropy (Bits)")
        
        # 2.Dynamic Backbone)
        plt.subplot(1, 4, 2)
        X_2d = struct_results.get('X_2d')
        if X_2d is not None:
            plt.scatter(X_2d[:, 0], X_2d[:, 1], 
                       c=struct_results.get('cluster_labels'), 
                       s=1, alpha=0.3, cmap=CLUSTER_CMAP)
        plt.title("2. Neural Flow Field\n(Extracting Geometry)")
        
        # 3.Logic Extraction
        plt.subplot(1, 4, 3)
        T_matrix = struct_results.get('T_matrix')
        if T_matrix is not None:
            plt.imshow(T_matrix, cmap=HEAT_CMAP)
        plt.title("3. State Transition Matrix\n(The 'Algorithm' Fingerprint)")
        
        # 4.Task Performance)
        plt.subplot(1, 4, 4)
        go_acc = getattr(self, 'go_acc', 0.5)
        plt.bar(["Baseline", "Our Model"], [0.5, go_acc], color=[_C_GRAY, _C_PURPLE])
        plt.title("4. Decoding Accuracy\n(Task Success)")
        plt.ylim(0, 1)
        
        plt.tight_layout()
        base_dir = os.path.dirname(self.mat_file_path)
        chain_path = os.path.join(base_dir, "Evidence_Chain_Flow.png")
        plt.savefig(chain_path, dpi=300, bbox_inches='tight')
        print(f" {chain_path}")
        plt.close('all')


    def plot_temporal_parsing_comparison(self, structure_results):
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        show_len = min(1000, self.X.shape[0])
        im1 = axes[0].imshow(self.X[:show_len, :].T, aspect='auto', cmap=MANIFOLD_CMAP)
        axes[0].set_title("Stage 1: Without Temporal Parsing\n(Raw High-Dimensional Signal)", fontsize=14)
        axes[0].set_ylabel("Neural Channels (49+)")
        axes[0].set_xlabel("Time Frames")
        plt.colorbar(im1, ax=axes[0], label="Activity")
        
        labels = structure_results.get('cluster_labels')
        if labels is not None:
            labels = labels[:show_len]
            axes[1].plot(labels, color=_C_PURPLE, linewidth=0.5, alpha=0.3)
            sc = axes[1].scatter(range(show_len), labels, c=labels, cmap=CLUSTER_CMAP, s=10)
            axes[1].set_title("Stage 2: With Temporal Parsing\n(Discretized State Logic)", fontsize=14)
            axes[1].set_ylabel("State ID (C0 - C5)")
            axes[1].set_xlabel("Time Frames")
            axes[1].set_yticks(range(len(np.unique(labels))))
        
        plt.tight_layout()
        base_dir = os.path.dirname(self.mat_file_path)
        parsing_path = os.path.join(base_dir, "Temporal_Parsing_Comparison.png")
        plt.savefig(parsing_path, dpi=300, bbox_inches='tight')
        print(f" {parsing_path}")
        plt.close('all')

    def run_permutation_test(self, n_permutations=50):
        print(f"\n Rapid displacement test ({n_permutations} times)...")

        X_train, X_test, y_go_train, y_go_test = train_test_split(
            self.X, self.Y_Go, test_size=self.test_size, shuffle=False
        )
        actual_go_acc = self.go_acc
        n_classes = len(np.unique(np.argmax(self.Y_Go, axis=1)))
        chance_level = 1.0 / n_classes

        go_null_scores = []
        for i in range(n_permutations):
            y_go_shuffled = np.random.permutation(np.argmax(y_go_train, axis=1))
            model_go = RidgeClassifierCV(alphas=np.logspace(-1, 4, 10)).fit(X_train, y_go_shuffled)
            pred = model_go.predict(X_test)
            true = np.argmax(y_go_test, axis=1)
            go_null_scores.append(np.mean(pred == true))

        null_arr = np.array(go_null_scores)
        p_empirical = (np.sum(null_arr >= actual_go_acc) + 1) / (n_permutations + 1)
        from scipy.stats import norm as _norm
        null_mean, null_std = null_arr.mean(), null_arr.std() + 1e-9
        p_approx = 1.0 - _norm.cdf(actual_go_acc, loc=null_mean, scale=null_std)

        print(f"  {null_mean:.4f}  ±  {null_arr.std():.4f}")
        print(f"  ({n_permutations} 次): {p_empirical:.4f}")
        print(f"   {p_approx:.4f}")
        print(f"  ({chance_level:.3f},  {actual_go_acc:.4f})")
        return p_empirical, go_null_scores

    def plot_permutation_test(self, p_go, null_scores, actual_acc):
        plt.figure(figsize=(8, 6))
        plt.hist(null_scores, bins=30, color=_C_LAVENDER, alpha=0.7, label='Null Distribution')
        plt.axvline(actual_acc, color=_C_PURPLE, linestyle='dashed', linewidth=2, 
                   label=f'Actual Acc: {actual_acc:.2%}')
        plt.title(f'Permutation Test (P-value: {p_go:.4f})')
        plt.xlabel('Accuracy')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        base_dir = os.path.dirname(self.mat_file_path)
        perm_path = os.path.join(base_dir, "Permutation_Test.png")
        plt.savefig(perm_path, dpi=300, bbox_inches='tight')
        print(f"{perm_path}")
        plt.close('all')

    def run_ablation_study(self):
        print("\n ablation study..")
        configs = [
            {"name": "Full Model", "pca": True, "delay": True, "tanh": True, "smooth": True},
            {"name": "w/o PCA", "pca": False, "delay": True, "tanh": True, "smooth": True},
            {"name": "w/o Delay", "pca": True, "delay": False, "tanh": True, "smooth": True},
            {"name": "w/o Tanh", "pca": True, "delay": True, "tanh": False, "smooth": True},
            {"name": "Random Control", "pca": True, "delay": True, "tanh": True, "smooth": True, "random": True},
        ]
        
        results = []
        for cfg in configs:
            X = self._prepare_features(use_pca=cfg["pca"], use_delay=cfg["delay"], use_tanh=cfg["tanh"])
            Y_Go, Y_Maze = self._prepare_targets(use_smoothing=cfg["smooth"])
            
            X_train, X_test, y_go_train, y_go_test, y_maze_train, y_maze_test = train_test_split(
                X, Y_Go, Y_Maze, test_size=0.3, shuffle=True
            )
            
            if cfg.get("random"):
                y_go_train = np.random.permutation(y_go_train)
                y_maze_train = np.random.permutation(y_maze_train)
            
            model_go = RidgeClassifierCV(alphas=np.logspace(-1, 4, 10)).fit(X_train, np.argmax(y_go_train, axis=1))
            acc_go = np.mean(model_go.predict(X_test) == np.argmax(y_go_test, axis=1))
            
            model_maze = RidgeCV(alphas=np.logspace(-3, 6, 20)).fit(X_train, y_maze_train)
            pred_maze = model_maze.predict(X_test)
            
            corr_x = self._safe_pearson(y_maze_test[:, 0], pred_maze[:, 0])
            corr_y = self._safe_pearson(y_maze_test[:, 1], pred_maze[:, 1])
            maze_acc = (corr_x + corr_y) / 2.0 * 100
            
            results.append({"name": cfg["name"], "go_acc": acc_go * 100, "maze_acc": maze_acc})
            print(f"  - {cfg['name']}: accuracy of Go {acc_go*100:.2f}%,  accuracy of maze {maze_acc:.2f}")
        
        base_dir = os.path.dirname(self.mat_file_path)
        csv_path = os.path.join(base_dir, "ablation_results.csv")
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=["name", "go_acc", "maze_acc"])
            writer.writeheader()
            writer.writerows(results)
        print(f" {csv_path}")
        return results

    def plot_ablation_study(self, results):
        names = [r['name'] for r in results]
        go_accs = [r['go_acc'] for r in results]
        maze_accs = [r['maze_acc'] for r in results]
        
        x = np.arange(len(names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width/2, go_accs, width, label='Go Accuracy (%)', color=_C_BLUE)
        rects2 = ax.bar(x + width/2, maze_accs, width, label='Maze Accuracy (%)', color=_C_TEAL)
        
        ax.set_ylabel('Performance (%)')
        ax.set_title('Ablation Study: Go vs Maze')
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        plt.tight_layout()
        base_dir = os.path.dirname(self.mat_file_path)
        ablation_path = os.path.join(base_dir, "Ablation_Study.png")
        plt.savefig(ablation_path, dpi=300, bbox_inches='tight')
        print(f" {ablation_path}")
        plt.close('all')

    def find_entropy_milestones(self, X_feat, n_moves=3):

        window = 10
        entropies = []
        for t in range(len(X_feat) - window):
            hist, _ = np.histogram(X_feat[t:t+window, 0], bins=10, density=True)
            entropies.append(entropy(hist + 1e-12))
        
        entropies = np.array(entropies)
        
        peaks, _ = find_peaks(-entropies, distance=40, prominence=0.1)
        
        if len(peaks) < n_moves:
            sorted_idx = np.argsort(entropies)[:n_moves]
            milestones = np.sort(sorted_idx)
        else:
            milestones = peaks[:n_moves]
            
        return milestones, entropies

    def calculate_adaptive_go_accuracy(self, X_feat):
        window = 10
        entropy_trace = []
        for t in range(len(X_feat) - window):
            hist, _ = np.histogram(X_feat[t:t+window, 0], bins=10, density=True)
            entropy_trace.append(entropy(hist + 1e-12))
        entropy_trace = np.array(entropy_trace)

        peaks, _ = find_peaks(-entropy_trace, distance=40, prominence=0.1)
        
        if len(peaks) < 3:
            milestones = np.sort(np.argsort(entropy_trace)[:3])
        else:
            milestones = peaks[:3]

        target_indices = self.go_env.target_indices
        
        X_train_list = []
        y_train_list = []
        for i, m_idx in enumerate(milestones):
            window_slice = X_feat[max(0, m_idx-2) : min(len(X_feat), m_idx+3)]
            X_train_list.append(window_slice)
            y_train_list.extend([target_indices[i]] * len(window_slice))
        
        X_train = np.vstack(X_train_list)
        y_train = np.array(y_train_list)
        
        go_model_adaptive = RidgeClassifierCV().fit(X_train, y_train)

        correct_count = 0
        results_log = []
        for i, m_idx in enumerate(milestones):
            prediction = go_model_adaptive.predict(X_feat[m_idx].reshape(1, -1))[0]
            is_correct = (prediction == target_indices[i])
            if is_correct: correct_count += 1
            results_log.append({
                "round": i+1,
                "frame": m_idx,
                "pred": prediction,
                "actual": target_indices[i],
                "status": "yes" if is_correct else "no"
            })
        
        acc = correct_count / 3.0
        return acc, milestones, results_log

    def run_go_ablation_comparison(self):
        print("\n" + "="*40)
        print("Start to execute adaptive Go decision analysis (triggered by entropy collapse)")
        print("="*40)


        X_with = self._prepare_features(use_delay=True) 
        acc_with, points_with, log_with = self.calculate_adaptive_go_accuracy(X_with)
        
        print(f"\n[Mode: time delay]")
        for entry in log_with:
            print(f"  Move {entry['round']} (frame {entry['frame']}): {entry['status']} predict:{entry['pred']}")


        X_without = self._prepare_features(use_delay=False)
        acc_without, points_without, log_without = self.calculate_adaptive_go_accuracy(X_without)
        
        print(f"\n[Mode: No time delay]")
        for entry in log_without:
            print(f"  Move {entry['round']} (frame {entry['frame']}): {entry['status']} predict:{entry['pred']}")

        print(f"\n Final accuracy rate comparison:")
        print(f"  > With Delay:    {acc_with*100:.1f}%")
        print(f"  > Without Delay: {acc_without*100:.1f}%")
        
        return acc_with, acc_without

    def train_and_evaluate_real_game(self, n_train_moves=2, n_test_moves=1):
        print("\n" + "="*50)
        print("Dynamic feedforward prediction evaluation (neural state evolution prediction)")
        print("="*50)

        results = {}
        for mode in ['With Delay', 'Without Delay']:
            is_delay = (mode == 'With Delay')
            X = self._prepare_features(use_delay=is_delay)
            
            milestones, _ = self.find_entropy_milestones(X, n_moves=3)
            
            if len(milestones) < 3:
                print(f" {mode} , skip")
                continue

            # X_input:  [Move 1 的状态, Move 2 的状态]
            # X_target: [Move 2 的状态, Move 3 的状态]
            X_input = X[milestones[:-1]] 
            X_target = X[milestones[1:]]

            model = LinearRegression().fit(X_input, X_target)
            X_pred = model.predict(X_input)

            r_vals = []
            for i in range(len(X_target)):
                r, _ = pearsonr(X_target[i], X_pred[i])
                r_vals.append(r)
            
            avg_r = np.mean(r_vals)
            results[mode] = avg_r
            print(f"  [{mode}]  (R): {avg_r:.4f}")

        print("\n Comparison of dynamic prediction capabilities:")
        diff = results.get('With Delay', 0) - results.get('Without Delay', 0)
        print(f"  > Dynamic gain (Prediction Gain): {diff:.4f}")
        
        return results

    def train_and_evaluate(self, test_size=0.3):
        self.test_size = test_size
        X_train, X_test, y_go_train, y_go_test, y_maze_train, y_maze_test = train_test_split(
            self.X, self.Y_Go, self.Y_Maze, test_size=test_size, shuffle=False
        )
        
        print("\n--- Train the reading layer of Go (RidgeClassifierCV) ---")
        self.go_model = RidgeClassifierCV(alphas=np.logspace(-1, 4, 10))
        self.go_model.fit(X_train, np.argmax(y_go_train, axis=1))
        
        scores = self.go_model.decision_function(X_test)
        if scores.ndim == 1:
            probs = np.vstack([1 - 1/(1+np.exp(-scores)), 1/(1+np.exp(-scores))]).T
        else:
            probs = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
        
        go_pred_probs = np.zeros((probs.shape[0], 49))
        for i, cls in enumerate(self.go_model.classes_):
            go_pred_probs[:, cls] = probs[:, i]
        
        pred_moves = np.argmax(go_pred_probs, axis=1)
        true_moves = np.argmax(y_go_test, axis=1)
        self.go_acc = np.mean(pred_moves == true_moves)
        print(f"The accuracy rate of Go move prediction: {self.go_acc * 100:.2f}%")
        
        print("\n--- Train the maze readout layer (automatic optimization) ---")
        self.maze_model = RidgeCV(alphas=np.logspace(-1, 4, 20))
        self.maze_model.fit(X_train, y_maze_train)
        
        y_maze_pred_all = self.maze_model.predict(self.X)
        num_runs = int(np.ceil(self.T_frames / self.maze_env.frames_per_run))
        
        self.trial_accuracies = []
        
        for i in range(num_runs):
            start_idx = i * self.maze_env.frames_per_run
            end_idx = min((i + 1) * self.maze_env.frames_per_run, self.T_frames)
            
            if (end_idx - start_idx) == self.maze_env.frames_per_run:
                y_true_run = self.Y_Maze[start_idx:end_idx]
                y_pred_run = y_maze_pred_all[start_idx:end_idx]
                
                corr_x = self._safe_pearson(y_true_run[:, 0], y_pred_run[:, 0])
                corr_y = self._safe_pearson(y_true_run[:, 1], y_pred_run[:, 1])
                mean_corr = (corr_x + corr_y) / 2.0
                self.trial_accuracies.append(max(0, mean_corr) * 100)
            else:
                print(f"  - Trial {i+1}  is imcomplete,skipped")
        
        self.maze_mean_acc = np.mean(self.trial_accuracies)
        self.maze_sem_acc = sem(self.trial_accuracies) if len(self.trial_accuracies) > 1 else 0.0
        
        print(f"\n---  MazeTrials ---")
        print(f" {num_runs}  trials in total")
        for i, acc in enumerate(self.trial_accuracies):
            print(f"  -  accuracy of {i+1} trail: {acc:.2f}%")
        print(f"Average accuracy rate of the maze: {self.maze_mean_acc:.2f}% ± {self.maze_sem_acc:.2f}% (SEM)")
        
        return y_go_test, go_pred_probs, y_maze_test, y_maze_pred_all

    def save_results(self):
        print("saving...")
        y_maze_pred_all = self.maze_model.predict(self.X)
        y_go_pred_all = self.go_model.predict(self.X)

        self._cached_maze_pred = y_maze_pred_all
        self._cached_go_pred = y_go_pred_all
        trials_x, trials_y = self.maze_env.simulate_physics(y_maze_pred_all)
        
        flat_x, flat_y = [], []
        for tx, ty in zip(trials_x, trials_y):
            flat_x.extend(tx)
            flat_x.append(np.nan)
            flat_y.extend(ty)
            flat_y.append(np.nan)
        
        results_dict = {
            'maze_target_velocity': self.Y_Maze,
            'maze_predicted_velocity': y_maze_pred_all,
            'maze_simulated_trajectory_X': np.array(flat_x),
            'maze_simulated_trajectory_Y': np.array(flat_y),
            'maze_perfect_path': np.array(self.maze_env.path),
            'maze_map': self.maze_env.maze,
            'go_target_matrix': self.Y_Go,
            'go_predicted_probs': y_go_pred_all,
            'metrics_go_accuracy': self.go_acc,
            'metrics_maze_trial_accuracies': np.array(self.trial_accuracies),
            'metrics_maze_mean_acc': self.maze_mean_acc,
            'metrics_maze_sem_acc': self.maze_sem_acc
        }


        if hasattr(self, '_cached_state_results') and self._cached_state_results is not None:
            sr = self._cached_state_results
            results_dict['state_selective_activity_matrix'] = sr.get('state_activity', np.array([]))
            results_dict['state_selective_index'] = sr.get('selectivity_index', np.array([]))
            results_dict['state_preferred_state'] = sr.get('preferred_state', np.array([]))
            results_dict['state_n_high_selective'] = sr.get('n_high_selective', 0)
            results_dict['state_si_threshold'] = sr.get('si_threshold', 0.0)


        if hasattr(self, '_cached_assembly_dynamics') and self._cached_assembly_dynamics is not None:
            ad = self._cached_assembly_dynamics
            results_dict['assembly_dynamics_matrix'] = ad.get('dynamics_matrix', np.array([]))
            results_dict['assembly_transition_entropy'] = ad.get('assembly_transition_entropy', 0.0)
            results_dict['assembly_state_transition_entropy'] = ad.get('state_transition_entropy', 0.0)
            if 'dominant_assembly' in ad:
                results_dict['assembly_dominant_sequence'] = ad['dominant_assembly']

        if hasattr(self, '_cached_reorg_results') and self._cached_reorg_results is not None:
            rr = self._cached_reorg_results
            results_dict['reorg_dominant_assembly']      = rr['dominant_assembly']
            results_dict['reorg_dominance_timeseries']   = rr['dominance_timeseries']
            results_dict['reorg_cohesion_timeseries']    = rr['cohesion_timeseries']
            results_dict['reorg_delta_T_state']          = rr['delta_T_state']
            results_dict['reorg_delta_T_assembly']       = rr['delta_T_assembly']
            results_dict['reorg_state_entropy_early']    = rr['state_entropy_early']
            results_dict['reorg_state_entropy_late']     = rr['state_entropy_late']
            results_dict['reorg_assembly_entropy_early'] = rr['assembly_entropy_early']
            results_dict['reorg_assembly_entropy_late']  = rr['assembly_entropy_late']
            results_dict['reorg_frobenius_distances']    = np.array(rr['frobenius_distances'])
            results_dict['reorg_total_drift']            = rr['total_drift']
            results_dict['reorg_early_dominance']        = rr['early_dominance']
            results_dict['reorg_late_dominance']         = rr['late_dominance']
            for si, dm in enumerate(rr['dynamics_segments']):
                results_dict[f'reorg_dynamics_seg{si+1}'] = dm

        base_dir = os.path.dirname(self.mat_file_path)
        base_name = os.path.splitext(os.path.basename(self.mat_file_path))[0]
        
        save_mat_path = os.path.join(base_dir, f"{base_name}_BioRC_Results.mat")
        self.save_fig_path = os.path.join(base_dir, f"{base_name}_BioRC_Plot.png")
        
        sio.savemat(save_mat_path, results_dict)
        print(f" {save_mat_path}")

    def save_velocity_csv(self):
        y_maze_pred_all = getattr(self, '_cached_maze_pred', self.maze_model.predict(self.X))
        
        base_dir = os.path.dirname(self.mat_file_path)
        base_name = os.path.splitext(os.path.basename(self.mat_file_path))[0]
        csv_path = os.path.join(base_dir, f"{base_name}_Velocity_Comparison.csv")
        
        print(f" {csv_path}...")
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Frame', 'Target_VX', 'Target_VY', 'Predicted_VX', 'Predicted_VY', 'L2_Error'])
            
            for t in range(self.T_frames):
                t_vx, t_vy = self.Y_Maze[t]
                p_vx, p_vy = y_maze_pred_all[t]
                error = np.sqrt((t_vx - p_vx)**2 + (t_vy - p_vy)**2)
                writer.writerow([t, t_vx, t_vy, p_vx, p_vy, error])
        
        print(f" {self.T_frames} frame")

    def plot_all_results(self, go_pred_probs):
        fig = plt.figure(figsize=(16, 10))
        
        ax_maze = plt.subplot(2, 1, 1)
        y_maze_pred_all = getattr(self, '_cached_maze_pred', self.maze_model.predict(self.X))
        trials_x, trials_y = self.maze_env.simulate_physics(y_maze_pred_all)
        
        ax_maze.imshow(self.maze_env.maze, cmap='binary')
        if self.maze_env.path:
            path_y = [p[0] for p in self.maze_env.path]
            path_x = [p[1] for p in self.maze_env.path]
            ax_maze.plot(path_x, path_y, color=_C_GREEN, linewidth=4, alpha=0.6, label='Target Path (Perfect)')
        
        for i in range(len(trials_x)):
            ax_maze.plot(trials_x[i], trials_y[i], color=_C_PURPLE, linewidth=2, alpha=0.7)
        ax_maze.plot([], [], color=_C_PURPLE, linewidth=2, label='Predicted Path (Trials)')
        
        ax_maze.plot(self.maze_env.start_pos[1], self.maze_env.start_pos[0], 'go', markersize=8)
        ax_maze.plot(self.maze_env.end_pos[1], self.maze_env.end_pos[0], 'bo', markersize=8)
        
        split_idx = int(self.T_frames * (1 - self.test_size))
        trial_idx = split_idx // self.maze_env.frames_per_run
        idx_in_trial = split_idx % self.maze_env.frames_per_run
        if trial_idx < len(trials_x) and idx_in_trial < len(trials_x[trial_idx]):
            ax_maze.plot(trials_x[trial_idx][idx_in_trial], trials_y[trial_idx][idx_in_trial], 
                        'y*', markersize=15, markeredgecolor='black', label='Test Set Start')
        
        ax_maze.set_title(f'Task 1: Maze Navigation (Acc: {self.maze_mean_acc:.1f}% ± {self.maze_sem_acc:.1f}%, {len(self.trial_accuracies)} Valid Trials)', fontsize=14)
        ax_maze.set_xticks([]), ax_maze.set_yticks([])
        ax_maze.legend(loc='lower right')
        
        test_frames = go_pred_probs.shape[0]
        for move_idx in range(3):
            ax_go = plt.subplot(2, 3, 4 + move_idx)
            
            move_mask = np.array([((i + split_idx) // self.go_env.frames_per_move) % 3 == move_idx for i in range(test_frames)])
            if np.any(move_mask):
                avg_probs = np.mean(go_pred_probs[move_mask], axis=0)
                avg_probs = np.clip(avg_probs, 0, None)
                if np.max(avg_probs) > 0:
                    avg_probs = avg_probs / np.max(avg_probs)
            else:
                avg_probs = np.zeros(49)
                
            self.go_env.plot_go_heatmap(move_idx, avg_probs, ax_go)
            if move_idx == 0:
                ax_go.legend(loc='upper left', bbox_to_anchor=(-0.1, 1.15), fontsize='small')
        
        plt.suptitle("Bio-Reservoir Computing: Automated Hyperparameter Tuning", fontsize=18, fontweight='bold')
        plt.tight_layout()
        
        if hasattr(self, 'save_fig_path'):
            plt.savefig(self.save_fig_path, dpi=300, bbox_inches='tight')
            print(f"The image has been successfully saved to: {self.save_fig_path}")
        
        plt.close('all')


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw() 
    root.attributes('-topmost', True) 
    
    print("select .mat ...")
    mat_file = filedialog.askopenfilename(
        title="Select the calcium signal data file (.mat)",
        filetypes=[("MATLAB Files", "*.mat"), ("All Files", "*.*")]
    )
    
    root.destroy() 
    
    if not mat_file:
        print("no file is selected and the program exits")
        sys.exit(0)

    decoder = BioReservoirDecoder(
        mat_file_path=mat_file, 
        data_key='norm_df',
        transpose=False,  
        delay_steps=20 
    )
    

    delay_results = decoder.compare_delay_impact()

    rep_results = decoder.plot_representation_analysis()
    
    structure_results = decoder.representation_structure_analysis(n_clusters=6)

    decoder.plot_temporal_parsing_comparison(structure_results)

    decoder.plot_evidence_chain_flow(rep_results, structure_results)

    go_acc_with, go_acc_without = decoder.run_go_ablation_comparison()

    real_game_results = decoder.train_and_evaluate_real_game(n_train_moves=2, n_test_moves=1)


    y_go_true, go_pred_probs, y_maze_true, y_maze_pred = decoder.train_and_evaluate(test_size=0.3)
    
    if 'X_2d' in structure_results and 'cluster_labels' in structure_results:
        decoder.plot_flow_field(structure_results['X_2d'], stride=10, 
                                cluster_labels=structure_results['cluster_labels'])
    
    p_go, null_scores = decoder.run_permutation_test(n_permutations=50)
    decoder.plot_permutation_test(p_go, null_scores, decoder.go_acc)

    ablation_results = decoder.run_ablation_study()
    decoder.plot_ablation_study(ablation_results)

    if 'cluster_labels' in structure_results and 'assembly_results' in structure_results:
        assembly_reorg = decoder.assembly_reorganization_analysis(
            structure_results['cluster_labels'],
            structure_results['assembly_results'],
            window=100,
            n_segments=4
        )

    decoder.save_results()
    decoder.save_velocity_csv()
    decoder.plot_all_results(go_pred_probs)
    
    print("All tasks have been completed and the program exits normally.")
    print("The key evidence of the generated paper (all images have been automatically saved, no need to manually close the window)")
    print("  1.  Neural_Flow_Field.png")
    print("  2.  Transition_Matrix.png")
    print("  3.  Evidence_Chain_Summary.png")
    print("  4.  Neural_Dynamics_Full_Analysis.png")
    print("  5.  Raw_Neural_Representation.png")
    print("  6.  Delay_Impact_Analysis.png")
    print("  7.  Evidence_Chain_Flow.png")
    print("  8.  Temporal_Parsing_Comparison.png")
    print("  9.  Permutation_Test.png")
    print("  10. Ablation_Study.png")
    print("  11. State_Selective_Neurons.png ")
    print("  12. State_Selective_SI_Barchart.png")
    print("  13. State_Selective_Neuron_Analysis.xlsx")
    print("  14. Neuron_Assemblies.png")
    print("  15. Assembly_Dynamics.png ")
    print("  16. Assembly_Reorganization_L1_Dominance.png")
    print("  17. Assembly_Reorganization_L2_Cohesion.png")
    print("  18. Assembly_Reorganization_L3_TransitionDelta.png")
    print("  19. Assembly_Reorganization_L4_DynamicsDrift.png")
    print("  20. Assembly_Reorganization_Summary.png ")
    print("  21. <basename>_BioRC_Plot.png")
    
    sys.exit(0)