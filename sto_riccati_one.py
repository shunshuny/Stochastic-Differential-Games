import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_lyapunov as solve_lyapunov
from numpy.linalg import norm, inv
from scipy.integrate import solve_ivp

def simulate_sde(A_cl, Sigma, x0, T, dt):
    """
    SDE: dx = A_cl x dt + Sigma x dW_t
    Euler-Maruyama法で近似解を求める
    """
    num_steps = int(T / dt)
    n = len(x0)
    xs = np.zeros((num_steps + 1, n))
    xs[0] = x0

    for i in range(num_steps):
        t = i * dt
        x = xs[i]
        dW = np.random.randn(n) * np.sqrt(dt)  # Brownian increment N(0,1) に従う実数をn個生成
        dx_det = A_cl @ x * dt
        dx_sto = Sigma @ x * dW
        xs[i + 1] = x + dx_det + dx_sto

    return np.linspace(0, T, num_steps + 1), xs

def solve_riccati_newton(A, B, Q, R, tol=1e-10, max_iter=100):
    """
    ニュートン法で連続時間型リカッチ方程式を解く（確定版）
    A, B, Q, R: numpy配列
    tol: 収束許容誤差
    max_iter: 最大反復回数
    """
    n = A.shape[0]
    X = np.zeros((n, n))  # 初期解

    R_inv = inv(R)

    for i in range(max_iter):
        # 関数 F(X) の定義
        F = A.T @ X + X @ A - X @ B @ R_inv @ B.T @ X + Q  # リカッチ代数方程式の残差

        # 収束判定
        err = norm(F, ord='fro')  # フロベニウスノルムを計算
        # print(f"Iteration {i}, Residual norm: {err:.2e}")
        if err < tol:
            break

        # ヤコビアンに基づく方向計算 (近似線形化)
        # Solve Lyapunov equation for Newton step: A_Tilde^T dX + dX A_Tilde = -F
        A_tilde = A - B @ R_inv @ B.T @ X
        dX = solve_lyapunov(A_tilde.T, -F)

        # 更新
        X += dX

        # 対称性を強制（数値誤差で非対称になる場合）
        X = (X + X.T) / 2

    return X

def solve_riccati_newton_sto(A, Ap, B, Q, R, tol=1e-10, max_iter=100):
    """
    ニュートン法で連続時間型確率リカッチ方程式を解く（確率版）
    A, B, Q, R: numpy配列
    tol: 収束許容誤差
    max_iter: 最大反復回数
    """
    n = A.shape[0]
    X2 = solve_riccati_newton(A, B, Q, R, tol=1e-10, max_iter=100)  # 初期解を確定版AREの解でおく　

    R_inv = inv(R)
    S = B @ R_inv @ B.T

    for i in range(max_iter):
        # 関数 F(X) の定義
        # F = X2 @ (A - S @ X1) + (A - S @ X1).T @ X2 @ + Ap.T @ X2 @ Ap + X1 @ S @ X1 + Q  # リカッチ代数方程式の残差
        F = A.T @ X2 + X2 @ A + Ap.T @ X2 @ Ap - X2 @ B @ R_inv @ B.T @ X2 + Q # 確率リカッチ代数方程式の残差

        # 収束判定
        err = norm(F, ord='fro')  # フロベニウスノルムを計算
        print(f"Iteration {i}, Residual norm: {err:.2e}")
        if err < tol:
            break

        # ヤコビアンに基づく方向計算 (近似線形化)
        # Solve Lyapunov equation for Newton step: A_Tilde^T dX + dX A_Tilde = -F
        A_tilde = A - B @ R_inv @ B.T @ X2
        dX = solve_lyapunov(A_tilde.T, -F)

        # 更新
        X2 += dX

        # 対称性を強制（数値誤差で非対称になる場合）
        X2 = (X2 + X2.T) / 2

    return X2

# システム行列の例
A = np.array([[-2., 1.], [0., -1.]])
Ap = 0.5 * A
B = np.array([[1.], [-5.]])
Q = np.array([[1., 0.], [0., 2.]])
R = np.array([[10.]])

P = solve_riccati_newton_sto(A, Ap, B, Q, R)
print("解 P:")
print(P)

# 3. フィードバックゲイン計算
K = inv(R) @ B.T @ P
A_cl = A - B @ K  # 閉ループ系

x0 = np.array([1.0, 1.0])  # 初期状態
T = 10.0
dt = 0.01
Sigma = Ap  # 拡散係数（ノイズ強度）

t_vals, x_vals = simulate_sde(A_cl, Sigma, x0, T, dt)

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

