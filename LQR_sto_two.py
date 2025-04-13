import numpy as np
import LQR_class
import matplotlib.pyplot as plt
from numpy.linalg import norm, inv
from scipy.linalg import solve_continuous_lyapunov as solve_lyapunov
from scipy.integrate import solve_ivp

lqr = LQR_class.LQR()

# システム行列の例
A = np.array([[-1., 1.], [0., -2.]])
Ap = 0.5 * A
B1 = np.array([[0.], [1.]])
B2 = np.array([[1.], [1.]])
Q1 = np.array([[1., 0.2], [0.2, 1.0]])
Q2 = np.array([[2., 0.], [0., 2.0]])
R11 = np.array([[10.]])
R22 = np.array([[1.]])

P1, P2 = lqr.newton_two_sto(A, Ap, B1, B2, Q1, Q2, R11, R22)
print("解 P1:")
print(P1)
print("解 P2:")
print(P2)

# 3. フィードバックゲイン計算
K1 = inv(R11) @ B1.T @ P1
K2 = inv(R22) @ B2.T @ P2
A_cl = A - B1 @ K1 - B2 @ K2  # 閉ループ系
eigvals = np.linalg.eigvals(A_cl)
print("A_cl の固有値:", eigvals)

x0 = np.array([1.0, 1.0])  # 初期状態
T = 10.0
dt = 0.01
Sigma = Ap  # 拡散係数（ノイズ強度）

t_vals, x_vals = lqr.simulate_sde(A_cl, Sigma, x0, T, dt)

# 6. グラフ描画
def plot_state_trajectories(t_vals, x_vals, plot_norm=True):
    """
    任意次元の状態ベクトル x(t) の各成分をプロット
    plot_norm=True の場合、ノルム ||x(t)|| も追加でプロット
    """
    n = x_vals.shape[1]
    
    plt.figure(figsize=(10, 5))
    
    # 各成分のプロット
    for i in range(n):
        plt.rcParams['font.family'] = 'Meiryo'
        plt.plot(t_vals, x_vals[:, i], label=f'$x_{i+1}(t)$')
    
    # ノルムのプロット（オプション）
    if plot_norm:
        x_norms = np.linalg.norm(x_vals, axis=1)
        plt.rcParams['font.family'] = 'Meiryo'
        plt.plot(t_vals, x_norms, '--', label=r'$\|x(t)\|$', linewidth=2, color='black')
    
    plt.xlabel("Time [s]")
    plt.ylabel("State value")
    plt.title("状態ベクトルの時間変化")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
plot_state_trajectories(t_vals, x_vals, plot_norm=False)



