import numpy as np
import jax.numpy as jnp
from jax import jacfwd
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_lyapunov as solve_lyapunov
from numpy.linalg import norm, inv
from scipy.integrate import solve_ivp

class LQR:
    def simulate_sde(self, A_cl, Sigma, x0, T, dt):
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

    def newton_one_det(self, A, B, Q, R, tol=1e-10, max_iter=100):
        """
        ニュートン法で連続時間型リカッチ方程式を解く
        プレイヤー数１、確定的システム
        A, B, Q, R: numpy配列
        tol: 収束許容誤差
        max_iter: 最大反復回数
        """
        n = A.shape[0]
        X = np.zeros((n, n))  # 初期解

        self.__R_inv = inv(R)

        for i in range(max_iter):
            # 関数 F(X) の定義
            F = A.T @ X + X @ A - X @ B @ self.__R_inv @ B.T @ X + Q  # リカッチ代数方程式の残差

            # 収束判定
            err = norm(F, ord='fro')  # フロベニウスノルムを計算
            print(f"Iteration {i}, Residual norm: {err:.2e}")
            if err < tol:
                break

            # ヤコビアンに基づく方向計算 (近似線形化)
            # Solve Lyapunov equation for Newton step: A_Tilde^T dX + dX A_Tilde = -F
            A_tilde = A - B @ self.__R_inv @ B.T @ X
            dX = solve_lyapunov(A_tilde.T, -F)

            # 更新
            X += dX

            # 対称性を強制（数値誤差で非対称になる場合）
            self.__X = (X + X.T) / 2

        return self.__X
    
    def newton_one_sto(self, A, Ap, B, Q, R, tol=1e-10, max_iter=100):
        """
        ニュートン法で連続時間型確率リカッチ方程式を解く
        プレイヤー数１、確率的システム
        A, B, Q, R: numpy配列
        tol: 収束許容誤差
        max_iter: 最大反復回数
        """
        # self.__n = A.shape[0]
        X2 = self.newton_one_det(A, B, Q, R, tol=1e-10, max_iter=100)  # 初期解を確定版AREの解でおく　

        self.__R_inv = inv(R)
        self.__S = B @ self.__R_inv @ B.T

        for i in range(max_iter):
            # 関数 F(X) の定義
            # F = X2 @ (A - S @ X1) + (A - S @ X1).T @ X2 @ + Ap.T @ X2 @ Ap + X1 @ S @ X1 + Q  # リカッチ代数方程式の残差
            F = A.T @ X2 + X2 @ A + Ap.T @ X2 @ Ap - X2 @ self.__S @ X2 + Q # 確率リカッチ代数方程式の残差

            # 収束判定
            err = norm(F, ord='fro')  # フロベニウスノルムを計算
            print(f"Iteration {i}, Residual norm: {err:.2e}")
            if err < tol:
                break

            # ヤコビアンに基づく方向計算 (近似線形化)
            # Solve Lyapunov equation for Newton step: A_Tilde^T dX + dX A_Tilde = -F
            A_tilde = A - B @ self.__R_inv @ B.T @ X2
            dX = solve_lyapunov(A_tilde.T, -F)

            # 更新
            X2 += dX

            # 対称性を強制（数値誤差で非対称になる場合）
            self.__X2 = (X2 + X2.T) / 2

        return self.__X2
    
    def newton_two_det(self, A, B1, B2, Q1, Q2, R11, R12, R21, R22, tol=1e-10, max_iter=100):
        """
        ニュートン法で連続時間型リカッチ連立方程式を解く
        プレイヤー数２、確定的システム
        A, B, Q, R: numpy配列
        tol: 収束許容誤差
        max_iter: 最大反復回数
        """
        n = A.shape[0]
        P1 = P2 = np.zeros((n, n))  # 初期解
        
        self.__R11_inv = inv(R11); self.__R22_inv = inv(R22)  
        self.__S1 = B1 @ self.__R11_inv @ B1.T; self.__S2 = B2 @ self.__R22_inv @ B2.T
        self.__G1 = B1 @ self.__R11_inv @ R21 @ self.__R11_inv @ B1.T
        self.__G2 = B2 @ self.__R22_inv @ R12 @ self.__R22_inv @ B2.T 

        def riccati_two(P1, P2):
            F1 = A.T @ P1 + P1 @ A - P1 @ self.__S1 @ P1 - P1 @ self.__S2 @ P2 - P2 @ self.__S2 @ P1 + P2 @ self.__G2 @ P2 + Q1  # リカッチ代数方程式の残差
            F2 = A.T @ P2 + P2 @ A - P2 @ self.__S2 @ P2 - P2 @ self.__S1 @ P1 - P1 @ self.__S1 @ P2 + P1 @ self.__G1 @ P1 + Q2
            return F1, F2
        
        for i in range(max_iter):
            # 関数 F(X) の定義
            F1, F2 = riccati_two(P1, P2)

            # 1. 残差関数をベクトル化（f = [vec(F1); vec(F2)]）
            f1 = F1.reshape(-1)   # or F1.flatten()
            f2 = F2.reshape(-1)
            f = np.concatenate([f1, f2])  # サイズ: 2n^2

            # 2. ヤコビアンの計算（自作 or JAX/autograd で）
            def compute_jacobian(P1, P2):
                n = P1.shape[0]

                def residuals(vecP):
                    # ベクトルを行列に変換
                    P1_ = vecP[:n*n].reshape((n, n))
                    P2_ = vecP[n*n:].reshape((n, n))
                    F1, F2 = riccati_two(P1_, P2_)
                    return jnp.concatenate([F1.reshape(-1), F2.reshape(-1)])

                # 入力をベクトル化
                vecP0 = jnp.concatenate([P1.reshape(-1), P2.reshape(-1)])

                # 自動微分でヤコビアン計算
                J_fn = jacfwd(residuals)
                J = J_fn(vecP0)
                return J
            J = compute_jacobian(P1, P2)  # サイズ: 2n^2 × 2n^2

            # 3. ニュートンステップの解を求める
            delta_p = np.linalg.solve(J, -f)  # サイズ: 2n^2

            # 4. 解の分割・行列への整形
            delta_P1 = delta_p[:n*n].reshape((n, n))
            delta_P2 = delta_p[n*n:].reshape((n, n))

            # 5. 更新 ＋ 対称性を保つために symmetrize
            self.__P1 = 0.5 * (P1 + delta_P1 + (P1 + delta_P1).T)
            self.__P2 = 0.5 * (P2 + delta_P2 + (P2 + delta_P2).T)

            # 6. 収束判定（残差のノルム）
            if np.linalg.norm(f) < tol:
                print(f"収束しました。反復回数: {i+1}")
                break
            if i == max_iter - 1:
                print(f"収束しませんでした。err = {np.linalg.norm(f)}")

            P1 = self.__P1
            P2 = self.__P2
        return self.__P1, self.__P2
    
    def newton_two_sto(self, A, Ap, B1, B2, Q1, Q2, R11, R22, tol=1e-10, max_iter=100):
        """
        ニュートン法で連続時間型確率リカッチ連立方程式を解く
        プレイヤー数２、確率的システム
        A, B, Q, R: numpy配列
        tol: 収束許容誤差
        max_iter: 最大反復回数
        Rij = 0 を仮定
        """
        n = A.shape[0]
        m = R11.shape[0]
        R12 = R21 = np.zeros((m,m))
        P1, P2  = self.newton_two_det(A, B1, B2, Q1, Q2, R11, R12, R21, R22)  # 初期解を確定版AREの解でおく　

        self.__R11_inv = inv(R11); self.__R22_inv = inv(R22)  
        self.__S1 = B1 @ self.__R11_inv @ B1.T; self.__S2 = B2 @ self.__R22_inv @ B2.T
        
        def riccati_two(P1, P2):
            F1 = (A - self.__S1 @ P1 - self.__S2 @ P2).T @ P1 + P1 @ (A - self.__S1 @ P1 - self.__S2 @ P2) + P1 @ self.__S1 @ P1 + Ap.T @ P1 @Ap + Q1  # リカッチ代数方程式の残差
            F2 = (A - self.__S1 @ P1 - self.__S2 @ P2).T @ P2 + P2 @ (A - self.__S1 @ P1 - self.__S2 @ P2) + P2 @ self.__S2 @ P2 + Ap.T @ P2 @Ap + Q2
            return F1, F2

        for i in range(max_iter):
            # 関数 F(X) の定義
            F1, F2 =  riccati_two(P1, P2) # 確率リカッチ代数方程式の残差

            # 1. 残差関数をベクトル化（f = [vec(F1); vec(F2)]）
            f1 = F1.reshape(-1)   # or F1.flatten()
            f2 = F2.reshape(-1)
            f = np.concatenate([f1, f2])  # サイズ: 2n^2

            # 2. ヤコビアンの計算（自作 or JAX/autograd で）
            def compute_jacobian(P1, P2):
                n = P1.shape[0]

                def residuals(vecP):
                    # ベクトルを行列に変換
                    P1_ = vecP[:n*n].reshape((n, n))
                    P2_ = vecP[n*n:].reshape((n, n))
                    F1, F2 = riccati_two(P1_, P2_)
                    return jnp.concatenate([F1.reshape(-1), F2.reshape(-1)])

                # 入力をベクトル化
                vecP0 = jnp.concatenate([P1.reshape(-1), P2.reshape(-1)])

                # 自動微分でヤコビアン計算
                J_fn = jacfwd(residuals)
                J = J_fn(vecP0)
                return J
            J = compute_jacobian(P1, P2)  # サイズ: 2n^2 × 2n^2

            # 3. ニュートンステップの解を求める
            delta_p = np.linalg.solve(J, -f)  # サイズ: 2n^2

            # 4. 解の分割・行列への整形
            delta_P1 = delta_p[:n*n].reshape((n, n))
            delta_P2 = delta_p[n*n:].reshape((n, n))

            # 5. 更新 ＋ 対称性を保つために symmetrize
            self.__P1 = 0.5 * (P1 + delta_P1 + (P1 + delta_P1).T)
            self.__P2 = 0.5 * (P2 + delta_P2 + (P2 + delta_P2).T)

            # 6. 収束判定（残差のノルム）
            if np.linalg.norm(f) < tol:
                print(f"収束しました。反復回数: {i+1}")
                break
            if i == max_iter - 1:
                print(f"収束しませんでした。err = {np.linalg.norm(f)}")
            
            P1 = self.__P1
            P2 = self.__P2
        return self.__P1, self.__P2