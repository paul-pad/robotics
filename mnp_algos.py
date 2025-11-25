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


def point_in_tetrahedron(p, A, B, C, D):
    """
    Checks if point p is inside tetrahedron ABCD.
    Uses the 'Same Side' technique.
    """

    def same_side(v1, v2, v3, v4, p):
        normal = np.cross(v2 - v1, v3 - v1)
        dotV4 = np.dot(normal, v4 - v1)
        dotP = np.dot(normal, p - v1)
        return np.sign(dotV4) == np.sign(dotP)

    return (
        same_side(A, B, C, D, p)
        and same_side(A, B, D, C, p)
        and same_side(A, C, D, B, p)
        and same_side(B, C, D, A, p)
    )


# --- Geometric Primitives (FCL Style) ---


def solve_segment(A, B):
    AB = B - A
    AO = -A
    dot_AB_AB = np.dot(AB, AB)

    if dot_AB_AB < 1e-12:
        return A, [A]

    t = np.dot(AO, AB) / dot_AB_AB

    if t <= 0:
        return A, [A]
    elif t >= 1:
        return B, [B]
    else:
        return A + t * AB, [A, B]


def solve_triangle(A, B, C):
    # Check Edge AB
    AB = B - A
    AC = C - A
    n = np.cross(AB, AC)

    # Degenerate check
    if np.dot(n, n) < 1e-12:
        return solve_segment(A, B)  # Fallback

    # Edge AB
    n_AB = np.cross(AB, n)
    if np.dot(n_AB, -A) > 0:
        return solve_segment(A, B)

    # Edge AC
    n_AC = np.cross(n, AC)
    if np.dot(n_AC, -A) > 0:
        return solve_segment(A, C)

    # Edge BC
    BC = C - B
    n_BC = np.cross(n, BC)
    if np.dot(n_BC, -B) > 0:
        return solve_segment(B, C)

    # Face Region
    denom = np.dot(n, n)
    # Projection of origin onto the plane
    P = (np.dot(A, n) / denom) * n
    return P, [A, B, C]


def solve_tetrahedron(A, B, C, D):
    """
    Project Origin onto Tetrahedron ABCD.
    Handles 'Inside' case by returning Origin.
    """
    # 1. Check if Origin is inside
    if point_in_tetrahedron(np.zeros(3), A, B, C, D):
        return np.zeros(3), [A, B, C, D]

    # 2. If outside, closest point is on the boundary (one of the faces)
    faces = [(A, B, C), (A, C, D), (A, D, B), (B, D, C)]

    best_pt = None
    best_W = None
    min_dist_sq = float("inf")

    for p1, p2, p3 in faces:
        pt, W_face = solve_triangle(p1, p2, p3)
        dist_sq = np.dot(pt, pt)

        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            best_pt = pt
            best_W = W_face

    return best_pt, best_W


def solve_simplex_subproblem(W):
    if len(W) == 1:
        return W[0], W
    if len(W) == 2:
        return solve_segment(W[0], W[1])
    if len(W) == 3:
        return solve_triangle(W[0], W[1], W[2])
    if len(W) == 4:
        return solve_tetrahedron(W[0], W[1], W[2], W[3])
    return W[-1], [W[-1]]


def fw(x_0, A: np.ndarray, B: np.ndarray, max_iter=1000, eps=0.01):
    x_k = x_0

    for _ in range(max_iter):
        d_k = 2 * x_k

        s_a = A[np.argmin(A @ d_k)]
        s_b = B[np.argmax(B @ d_k)]
        s_k = s_a - s_b

        g_fw = 2 * x_k.T @ (x_k - s_k)
        if g_fw < eps:
            return x_k

        x_k = lineSearch(x_k, s_k)

    return x_k, x_k.T @ x_k


def nesterov_fw(x_0, A: np.ndarray, B: np.ndarray, max_iter=1000, eps=0.01, tol=1e-8):
    x_k = x_0.copy()
    s_prev = x_0.copy()
    d_prev = x_0.copy()
    W_k = []

    d_k_to_be_replaced = False

    for k in range(max_iter):
        delta_k = (k + 1) / (k + 3)
        if d_k_to_be_replaced:
            d_k = 2 * x_k
        else:
            y_k = delta_k * x_k + (1 - delta_k) * s_prev
            d_k = delta_k * d_prev + (1 - delta_k) * 2 * y_k  # df(yk) = 2*yk

        s_a = A[np.argmin(A @ d_k)]
        s_b = B[np.argmax(B @ d_k)]
        s_k = s_a - s_b

        g_fw = 2 * x_k.T @ (x_k - s_k)
        if g_fw < eps:
            if d_k_to_be_replaced or np.linalg.norm(d_k - 2 * x_k) < tol:
                return x_k.T @ x_k

            d_k = 2 * x_k

            s_a = A[np.argmin(A @ d_k)]
            s_b = B[np.argmax(B @ d_k)]
            s_k = s_a - s_b
            d_k_to_be_replaced = True

        W_k.append(s_k)
        x_k, W_k = solve_simplex_subproblem(W_k)

        if np.linalg.norm(x_k) < tol:
            return 0.0

        s_prev = s_k
        d_prev = d_k

    return x_k, x_k.T @ x_k
