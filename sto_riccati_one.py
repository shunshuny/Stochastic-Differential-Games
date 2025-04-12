import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_lyapunov as solve_lyapunov
from numpy.linalg import norm, inv
from scipy.integrate import solve_ivp

def solve_riccati_newton(A, B, Q, R, tol=1e-10, max_iter=100):
    """
    ニュートン法で連続時間型確率リカッチ方程式を解く
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