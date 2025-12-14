import numpy as np


def lineSearch(a, b, tol=1e-8):
    d = b - a
    d_norm_squared = d.T @ d

    if d_norm_squared < tol:
        return a

    optimal_gamma = -d.T @ a / d_norm_squared
    optimal_gamma = np.clip(optimal_gamma, 0.0, 1.0)
    c = a + optimal_gamma * d
    return c


def get_barycentric_coords(p, a, b):
    v = b - a
    w = p - a
    d00 = np.dot(v, v)
    d01 = np.dot(v, w)
    if d00 < 1e-15:
        return 0.0  # Degenerate
    return max(0.0, min(1.0, d01 / d00))


def project_origin_to_simplex(W):
    """
    Robustly projects origin (0,0,0) onto the simplex W.
    Returns: (closest_point, new_active_W)
    """
    # 1. Point Case
    if len(W) == 1:
        return W[0], W

    # 2. Segment Case
    elif len(W) == 2:
        A, B = W[0], W[1]
        AB = B - A
        AO = -A

        denom = np.dot(AB, AB)
        if denom < 1e-15:
            return A, [A]  # Degenerate segment

        t = np.dot(AB, AO) / denom
        if t <= 0:
            return A, [A]  # Voronoi region A
        if t >= 1:
            return B, [B]  # Voronoi region B
        return A + t * AB, [A, B]  # Edge region

    # 3. Triangle Case
    elif len(W) == 3:
        A, B, C = W[0], W[1], W[2]

        # Normals
        AB = B - A
        AC = C - A
        n = np.cross(AB, AC)

        # Check if degenerate
        if np.dot(n, n) < 1e-10:
            # Fallback to edges if triangle is thin
            p1, w1 = project_origin_to_simplex([A, B])
            p2, w2 = project_origin_to_simplex([A, C])
            p3, w3 = project_origin_to_simplex([B, C])
            d1, d2, d3 = np.dot(p1, p1), np.dot(p2, p2), np.dot(p3, p3)
            best_idx = np.argmin([d1, d2, d3])
            return [p1, p2, p3][best_idx], [w1, w2, w3][best_idx]

        # Voronoi region checks for Triangle faces
        # (This is a simplified robust method: check all sub-features)

        # Check closest point on edges first
        p_ab, w_ab = project_origin_to_simplex([A, B])
        p_ac, w_ac = project_origin_to_simplex([A, C])
        p_bc, w_bc = project_origin_to_simplex([B, C])

        # Determine if origin projects onto the face interior
        # Compute Barycentric coordinates
        # P = uA + vB + wC
        v0 = B - A
        v1 = C - A
        v2 = -A
        d00 = np.dot(v0, v0)
        d01 = np.dot(v0, v1)
        d11 = np.dot(v1, v1)
        d20 = np.dot(v2, v0)
        d21 = np.dot(v2, v1)
        denom = d00 * d11 - d01 * d01

        if abs(denom) > 1e-15:
            v = (d11 * d20 - d01 * d21) / denom
            w = (d00 * d21 - d01 * d20) / denom
            u = 1.0 - v - w

            # If strictly inside triangle
            if u > 0 and v > 0 and w > 0:
                return u * A + v * B + w * C, [A, B, C]

        # If not inside, return closest edge point
        d_ab, d_ac, d_bc = np.dot(p_ab, p_ab), np.dot(p_ac, p_ac), np.dot(p_bc, p_bc)
        min_d = min(d_ab, d_ac, d_bc)
        if d_ab == min_d:
            return p_ab, w_ab
        if d_ac == min_d:
            return p_ac, w_ac
        return p_bc, w_bc

    # 4. Tetrahedron Case
    elif len(W) == 4:
        # Check if origin is inside
        # We can check strict signed volumes or simply check all faces
        A, B, C, D = W[0], W[1], W[2], W[3]

        # Attempt to solve P = uA + vB + wC + zD with P=Origin
        # System:
        # [Ax Bx Cx Dx] [u]   [0]
        # [Ay By Cy Dy] [v] = [0]
        # [Az Bz Cz Dz] [w]   [0]
        # [1  1  1  1 ] [z]   [1]

        M = np.array(
            [
                [A[0], B[0], C[0], D[0]],
                [A[1], B[1], C[1], D[1]],
                [A[2], B[2], C[2], D[2]],
                [1.0, 1.0, 1.0, 1.0],
            ]
        )

        try:
            # Check for degeneracy (volume ~ 0)
            if abs(np.linalg.det(M)) < 1e-15:
                raise np.linalg.LinAlgError

            # Solve for barycentric coordinates
            target = np.array([0, 0, 0, 1])
            coords = np.linalg.solve(M, target)

            # Check if Origin is inside (all coords positive)
            # Use negative epsilon for robust "on-boundary" detection
            if np.all(coords >= -1e-10):
                return np.zeros(3), W  # INTERSECTION DETECTED

        except np.linalg.LinAlgError:
            # Degenerate tetrahedron (flat), fall through to faces
            pass

        # If not inside, the closest point MUST be on one of the faces
        faces = [[A, B, C], [A, B, D], [A, C, D], [B, C, D]]

        best_p = None
        best_w = None
        min_d = float("inf")

        for face in faces:
            p, w_sub = project_origin_to_simplex(face)
            d = np.dot(p, p)
            if d < min_d:
                min_d = d
                best_p = p
                best_w = w_sub

        return best_p, best_w

    return W[-1], [W[-1]]


# Algorithm 4
def corrective_fw(
    x_0, A: np.ndarray, B: np.ndarray, max_iter=1000, opt_criterion=1e-12
):
    x_k = x_0
    W_k = []
    eps = 10e-12  # Threshold for two numbers to be equal

    opt_criterions = []
    for k in range(max_iter):
        d_k = 2 * x_k

        s_a = A[np.argmin(A @ d_k)]
        s_b = B[np.argmax(B @ d_k)]
        s_k = s_a - s_b

        g_fw = 2 * x_k.T @ (x_k - s_k)
        opt_criterions.append(g_fw)
        if g_fw < opt_criterion:
            return x_k, k + 1, x_k.T @ x_k, opt_criterions

        W_k.append(s_k)
        x_k, W_k = project_origin_to_simplex(W_k)

        if np.linalg.norm(x_k) < eps:
            return x_k, k + 1, 0.0, opt_criterions

    return x_k, k + 1, x_k.T @ x_k, opt_criterions


def corrective_fw_analytic(x_0, shape_A, shape_B, max_iter=1000, opt_criterion=1e-12):
    """
    Vanilla GJK (Frank-Wolfe) adapted for Analytic Shapes.
    Replaces the argmin/argmax over vertices with analytic support functions.
    """
    x_k = x_0.copy()
    W_k = []  # Simplex
    eps = 10e-12

    opt_criterions = []

    for k in range(max_iter):
        # Gradient of f(x) = ||x||^2 is 2x
        d_k = 2 * x_k

        # --- ANALYTIC SUPPORT MAP ---
        # The Minkowski difference D = A - B.
        # Support of D in direction -d_k is: s_A(-d_k) - s_B(d_k)
        # Note: The gradient direction is descent, so we look in -d_k relative to x_k?
        # Standard FW: s_k = argmin <s, gradient>.
        # So we want s in D that minimizes <s, 2x_k>.
        # This is equivalent to support of D in direction -x_k.
        # s_D(-v) = s_A(-v) - s_B(v)

        # Direction for support finding
        search_dir = -d_k

        # s_a = Support of A in direction search_dir
        s_a = shape_A.support_function(search_dir)
        # s_b = Support of B in direction -search_dir (which is d_k)
        s_b = shape_B.support_function(-search_dir)

        s_k = s_a - s_b

        # Duality Gap
        g_fw = 2 * x_k.T @ (x_k - s_k)
        opt_criterions.append(g_fw)

        if g_fw < opt_criterion:
            return x_k, k + 1, x_k.T @ x_k, opt_criterions

        # Update Simplex and Project
        W_k.append(s_k)
        x_k, W_k = project_origin_to_simplex(W_k)

        if np.linalg.norm(x_k) < eps:
            return x_k, k + 1, 0.0, opt_criterions

    return x_k, k + 1, x_k.T @ x_k, opt_criterions


# Algorithm 7
def nesterov_fw(
    x_0,
    A: np.ndarray,
    B: np.ndarray,
    max_iter=1000,
    opt_criterion=1e-12,
    normalization=False,
    restart=False,
):
    x_k = x_0.copy()
    s_prev = x_0.copy()
    d_prev = x_0.copy()
    W_k = []

    eps = 10e-12  # Threshold for two numbers to be equal
    use_nesterov = True

    opt_criterions = []
    j = 0
    for k in range(max_iter):
        delta_k = (j + 1) / (j + 3)

        if use_nesterov:
            y_k = delta_k * x_k + (1 - delta_k) * s_prev
            grad_y = 2 * y_k

            if normalization:
                # --- Algorithm 8: Normalized Update for Meshes ---
                # "Normalize direction for non-strictly convex shapes" [cite: 430]
                norm_d = np.linalg.norm(d_prev)
                norm_g = np.linalg.norm(grad_y)

                v1 = d_prev / norm_d if norm_d > 1e-9 else d_prev
                v2 = grad_y / norm_g if norm_g > 1e-9 else grad_y
                d_k = delta_k * v1 + (1 - delta_k) * v2
            else:
                # --- Algorithm 7: Standard Update for Ellipsoids ---
                d_k = delta_k * d_prev + (1 - delta_k) * grad_y
        else:
            d_k = 2 * x_k

        s_a = A[np.argmin(A @ d_k)]
        s_b = B[np.argmax(B @ d_k)]
        s_k = s_a - s_b

        g_fw = 2 * x_k.T @ (x_k - s_k)
        opt_criterions.append(g_fw)
        if g_fw < opt_criterion:
            # Fixed-point / Alignment Check
            aligned = False
            grad = 2 * x_k

            if normalization:
                # Check alignment of normalized vectors
                norm_dk = np.linalg.norm(d_k)
                norm_grad = np.linalg.norm(grad)
                if norm_dk > 1e-9 and norm_grad > 1e-9:
                    cos_sim = np.dot(d_k, grad) / (norm_dk * norm_grad)
                    if cos_sim > 0.99:
                        aligned = True
            else:
                if np.linalg.norm(d_k - grad) < eps:
                    aligned = True

            if not use_nesterov or aligned:
                return x_k, k + 1, x_k.T @ x_k, opt_criterions

            d_k = 2 * x_k

            s_a = A[np.argmin(A @ d_k)]
            s_b = B[np.argmax(B @ d_k)]
            s_k = s_a - s_b
            use_nesterov = False

        W_k.append(s_k)
        old_norm = x_k.T @ x_k
        x_k, W_k = project_origin_to_simplex(W_k)
        new_norm = x_k.T @ x_k

        if np.linalg.norm(x_k) < eps:
            return x_k, k + 1, 0.0, opt_criterions

        s_prev = s_k
        d_prev = d_k

        if restart and (new_norm > old_norm):
            print("j'ai restart")
            j = 0  # Reset momentum counter
            # Crucial: Reset momentum vector to current gradient (kill inertia)
            d_prev = 2 * x_k.copy()
            # Note: We keep x_k as is, but next d_k will be pure gradient based.
            s_prev = x_k.copy()
        else:
            j = j + 1

    return x_k, max_iter, x_k.T @ x_k, opt_criterions


def nesterov_fw_analytic(
    x_0,
    shape_A,
    shape_B,
    max_iter=1000,
    opt_criterion=1e-12,
    normalization=False,
    restart=True,
):
    """
    Nesterov-accelerated GJK for Analytic Shapes.
    Corresponds to Algorithm 7 in the paper[cite: 367].

    Args:
        x_0: Initial guess (np.array).
        shape_A, shape_B: Objects with a .support_function(d) method.
        max_iter: Maximum iterations.
        opt_criterion: Convergence threshold for duality gap.
        normalization: Whether to normalize directions (Alg 8).
                       Usually False for Ellipsoids, True for Polytopes.
        restart: Whether to use momentum restart (Hard Reset).
    """
    x_k = x_0.copy()

    # Initialize "previous" values for the first iteration
    # Paper sets d_{-1} = s_{-1} = x_0 [cite: 368]
    s_prev = x_0.copy()
    d_prev = x_0.copy()

    W_k = []  # Simplex
    use_nesterov = True

    opt_criterions = []
    j = 0  # Momentum schedule counter

    for k in range(max_iter):
        # 1. Update Momentum Parameter
        # delta_k = (k+1)/(k+3) usually, but we use j for restart capability
        delta_k = (j + 1) / (j + 3)

        if use_nesterov:
            # --- Nesterov Step ---
            # Eq (25a): Intermediary point y_k
            y_k = delta_k * x_k + (1 - delta_k) * s_prev
            grad_y = 2 * y_k  # Gradient of f(x)=||x||^2 is 2x

            if normalization:
                # Eq (30): Normalized update (Alg 8) [cite: 428]
                norm_d = np.linalg.norm(d_prev)
                norm_g = np.linalg.norm(grad_y)

                v1 = d_prev / norm_d if norm_d > 1e-9 else d_prev
                v2 = grad_y / norm_g if norm_g > 1e-9 else grad_y
                d_k = delta_k * v1 + (1 - delta_k) * v2
            else:
                # Eq (25b): Standard update [cite: 359]
                d_k = delta_k * d_prev + (1 - delta_k) * grad_y
        else:
            # Fallback to Vanilla GJK direction
            d_k = 2 * x_k

        # 2. Compute Analytic Support Vector
        # Eq (25c): s_k in S_D(d_k)
        # Note: The paper defines S_D(d) as argmin <d,s>.
        # For A-B, this means s_A(-d) - s_B(d).
        s_a = shape_A.support_function(-d_k)
        s_b = shape_B.support_function(d_k)
        s_k = s_a - s_b

        # 3. Duality Gap & Convergence Check
        g_fw = 2 * x_k.T @ (x_k - s_k)  # Eq (29)
        opt_criterions.append(g_fw)

        if g_fw < opt_criterion:
            # We are close to a solution or a fixed point.
            # Check Fixed Point Condition: Eq (28) <x_k, x_k - s_k> == 0
            # If d_k was NOT aligned with gradient, this might be a spurious fixed point.

            aligned = False
            grad = 2 * x_k

            # Check alignment between Nesterov direction d_k and actual gradient
            if normalization:
                norm_dk = np.linalg.norm(d_k)
                norm_grad = np.linalg.norm(grad)
                if norm_dk > 1e-9 and norm_grad > 1e-9:
                    cos_sim = np.dot(d_k, grad) / (norm_dk * norm_grad)
                    if cos_sim > 0.99:
                        aligned = True
            else:
                if np.linalg.norm(d_k - grad) < 1e-5:
                    aligned = True

            # If we are aligned (or not using Nesterov), it's a true optimum.
            if not use_nesterov or aligned:
                return x_k, k + 1, x_k.T @ x_k, opt_criterions

            # Otherwise: The acceleration stalled (Fixed Point case 2). [cite: 406-409]
            # Action: Stop Nesterov, switch to Vanilla GJK for this step.
            d_k = 2 * x_k
            s_a = shape_A.support_function(-d_k)
            s_b = shape_B.support_function(d_k)
            s_k = s_a - s_b
            use_nesterov = False

        # 4. Update Simplex and Project
        # Eq (25d) and (25e)
        W_k.append(s_k)
        old_norm = x_k.T @ x_k

        x_k, W_k = project_origin_to_simplex(W_k)

        new_norm = x_k.T @ x_k

        # 5. Check Intersection (Origin inside simplex)
        if np.linalg.norm(x_k) < opt_criterion:
            return x_k, k + 1, 0.0, opt_criterions

        # 6. Prepare Next Step
        s_prev = s_k
        d_prev = d_k

        # --- RESTART LOGIC ---
        if restart and (new_norm > old_norm):
            j = 0
            # FIX: Kill momentum by resetting to current gradient step
            d_prev = 2 * x_k.copy()
            s_prev = x_k.copy()
        else:
            j = j + 1

    return x_k, max_iter, x_k.T @ x_k, opt_criterions
