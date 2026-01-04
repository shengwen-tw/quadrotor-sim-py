import numpy as np


def care_sda(A: np.ndarray, H: np.ndarray, G: np.ndarray,
             r: float = 2.4, tol: float = 1e-9,
             max_iter: int = 50, raise_on_fail: bool = True) -> np.ndarray:
    """
    Solve the continuous-time Algebraic Riccati Equation (CARE) via the
    Structure-Preserving Doubling Algorithm (SDA):

        A.T @ X + X @ A - X @ G @ X + H = 0.

    Reference:
        "A structure-preserving doubling algorithm for continuous-time
        algebraic Riccati equations"
    """
    def inv(M):
        return np.linalg.inv(M)

    def norm(M):
        return np.linalg.norm(M)

    I = np.eye(A.shape[0])
    At = A.T
    Ar = A - r * I

    # Initialization
    A_old = I + 2 * r * inv(Ar + G @ inv(Ar.T) @ H)
    G_old = 2 * r * inv(Ar) @ G @ inv(Ar.T + H @ inv(Ar) @ G)
    H_old = 2 * r * inv(Ar.T + H @ inv(Ar) @ G) @ H @ inv(Ar)

    for _ in range(max_iter):
        # Precomputation
        I_HG_inv = inv(I + H_old @ G_old)
        A_old_t = A_old.T

        # Updates
        A_new = A_old @ inv(I + G_old @ H_old) @ A_old
        G_new = G_old + A_old @ G_old @ I_HG_inv @ A_old_t
        H_new = H_old + A_old_t @ I_HG_inv @ H_old @ A_old

        # Computate matrix norm for convergence check
        norm_H_old = norm(H_old)
        norm_H_now = norm(H_new)

        # Prepare next iteration
        A_old = A_new
        G_old = G_new
        H_old = H_new

        # Convergence check
        if abs(norm_H_now - norm_H_old) < tol:
            return H_new  # X = H_new

    if raise_on_fail:
        raise RuntimeError("SDA did not converge within max_iter.")

    return None
