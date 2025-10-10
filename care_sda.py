import numpy as np


def care_sda(A: np.ndarray, B: np.ndarray, H: np.ndarray, R: np.ndarray,
             r: float = 2.4, tol: float = 1e-9, max_iter: int = 10000) -> np.ndarray:
    """
    Solve CARE using the Structure-Preserving Doubling Algorithm (SDA).
    This is a direct translation of the provided MATLAB code.

    Args:
        A, B, H, R: numpy arrays with compatible dimensions.
        r: SDA parameter (suggested 2.1 ~ 2.6; default 2.4).
        tol: convergence tolerance on ||H_k|| difference.
        max_iter: safety cap on iterations.

    Returns:
        X: Solution matrix (same as H_hat at convergence).
    """
    n = A.shape[0]
    I = np.eye(n)

    # Helper to invert with a clear error if singular
    def inv(M):
        return np.linalg.inv(M)

    G = B @ inv(R) @ B.T
    A_T = A.T
    A_r = A - r * I

    # Initialize (SDA)
    A_hat_last = I + 2 * r * inv(A_r + G @ inv(A_r.T) @ H)
    G_hat_last = 2 * r * inv(A_r) @ G @ inv(A_r.T + H @ inv(A_r) @ G)
    H_hat_last = 2 * r * inv(A_r.T + H @ inv(A_r) @ G) @ H @ inv(A_r)

    for _ in range(max_iter):
        # Precompute to reduce redundancy
        inv_I_plus_HG = inv(I + H_hat_last @ G_hat_last)
        A_hat_last_T = A_hat_last.T

        # Updates
        A_hat_new = A_hat_last @ inv(I + G_hat_last @ H_hat_last) @ A_hat_last
        G_hat_new = G_hat_last + A_hat_last @ G_hat_last @ inv_I_plus_HG @ A_hat_last_T
        H_hat_new = H_hat_last + A_hat_last_T @ inv_I_plus_HG @ H_hat_last @ A_hat_last

        # Convergence check (same criterion as MATLAB)
        norm_H_last = np.linalg.norm(H_hat_last)
        norm_H_now = np.linalg.norm(H_hat_new)

        # Prepare next iteration
        A_hat_last = A_hat_new
        G_hat_last = G_hat_new
        H_hat_last = H_hat_new

        if abs(norm_H_now - norm_H_last) < tol:
            return H_hat_new

    # If not converged within max_iter, return the last iterate (or raise if you prefer)
    # raise RuntimeError("SDA did not converge within max_iter.")
    return H_hat_last
