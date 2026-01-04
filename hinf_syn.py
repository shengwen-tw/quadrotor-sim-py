import numpy as np

from care_sda import care_sda


def hinf_syn(A: np.ndarray, B1: np.ndarray, B2: np.ndarray, C1: np.ndarray,
             gamma_lb: float = 0, gamma_eps: float = 1e-5,
             residual_eps: float = 1e-5, lhp_eps: float = -1e-8):
    At = A.T
    B1B1t = B1 @ B1.T
    B2B2t = B2 @ B2.T
    C1tC1 = C1.T @ C1

    # Approximate an upper bound gamma
    rho_max = np.max(np.linalg.eigvals(B1.T @ B1))
    rho_0 = np.min(np.linalg.eigvals(B2.T @ B2))
    gamma_u = rho_max / rho_0

    # The lower bound γ of the H-infinity synthesis corresponds to the smallest
    # value for which a stabilizing solution exists. However, computing this
    # value can be computationally expensive, and the resulting optimal control
    # gain matrix may suffer from numerical ill-conditioning. In practice, we
    # allow the user to specify a larger lower bound on γ to obtain a
    # computationally efficient and numerically reliable suboptimal solution.
    # If gamma_lb is set to 0, the function returns an estimate of the true lower
    # bound γ.
    gamma_l = gamma_lb

    iteration = 0
    optimal_residual = 1e10
    optimal_gamma = gamma_u
    optimal_X = None

    # Bisection searching for optimal gamma and CARE solution
    while True:
        iteration += 1

        gamma = (gamma_l + gamma_u) / 2.0

        H = C1tC1
        G = B2B2t - (B1B1t / (gamma * gamma))
        X = care_sda(A, H, G, raise_on_fail=False)
        if X is None:
            # Infeasible, increase lower bound
            gamma_l = gamma
            continue

        # Compute CARE residual
        ric_residual = \
            np.linalg.norm(At @ X + X @ A - X @ G @ X + H, ord='fro')

        # Check if residual is small enough
        if ric_residual < residual_eps:
            # Stabilizing check: eigenvalues of (A - GX) must be in LHP
            eig_cl = np.linalg.eigvals(A - (G @ X))
            if np.max(np.real(eig_cl)) < lhp_eps:
                # Stabilizing solution, decrease the upper bound
                gamma_u = gamma

                # Record current best
                optimal_residual = ric_residual
                optimal_gamma = gamma
                optimal_X = X
            else:
                # Non-stabilizing solution, increase lower bound
                gamma_l = gamma
        else:
            # Residual too large, increase lower bound
            gamma_l = gamma

        # Bisection searching is converged
        if gamma_u - gamma_l < gamma_eps:
            return optimal_gamma, gamma_l, optimal_X, optimal_residual
