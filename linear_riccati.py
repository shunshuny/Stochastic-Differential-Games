import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_lyapunov as solve_lyapunov
from numpy.linalg import norm, inv
from scipy.integrate import solve_ivp

def solve_riccati_newton(A, B, Q, R, tol=1e-10, max_iter=100):
    """
    ニュートン法で連続時間型リカッチ方程式を解く
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
        print(f"Iteration {i}, Residual norm: {err:.2e}")
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

# システム行列の例
A = np.array([[-1., 1.], [0., -2.]])
B = np.array([[1.], [1.]])
Q = np.array([[1., 0.2], [0.2, 1.0]])
R = np.array([[1.]])

P = solve_riccati_newton(A, B, Q, R)
print("解 X:")
print(P)

# 3. フィードバックゲイン計算
K = inv(R) @ B.T @ P
A_cl = A - B @ K  # 閉ループ系
eigvals = np.linalg.eigvals(A_cl)
print("A_cl の固有値:", eigvals)

# 4. シミュレーション
def dynamics(t, x):
    return A_cl @ x

x0 = np.array([1.0, 0.0])  # 初期状態
t_span = (0, 10)
t_eval = np.linspace(*t_span, 300)

sol = solve_ivp(dynamics, t_span, x0, t_eval=t_eval)

# 5. ノルムの時間変化を計算
x_t = sol.y
x_norms = np.linalg.norm(x_t, axis=0)

# 6. グラフ描画
plt.rcParams['font.family'] = 'Meiryo'
plt.figure(figsize=(8, 4))
plt.plot(sol.t, x_norms)
plt.xlabel('Time [s]')
plt.ylabel('||x(t)||')
plt.title('状態ベクトル x(t) のノルムの時間変化')
plt.grid(True)
plt.tight_layout()
plt.show()