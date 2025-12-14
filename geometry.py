import numpy as np
import trimesh


class AnalyticEllipsoid:
    def __init__(self, center=None, A_matrix=None):
        # A_matrix corresponds to LL^T in the derivation above, or just the shape matrix
        if center is None:
            self.center = np.zeros(3)
        else:
            self.center = center

        if A_matrix is None:
            # Generate random shape if none provided
            M = np.random.randn(3, 3)
            self.B = M @ M.T + 0.1 * np.eye(3)  # B = L L^T
        else:
            self.B = A_matrix

    def support_function(self, d):
        """
        Computes the support vector of the ellipsoid in direction d.
        s = center + (B * d) / sqrt(d.T * B * d)
        """
        # Avoid division by zero
        norm = np.linalg.norm(d)
        if norm < 1e-12:
            return self.center

        Bd = self.B @ d
        denom = np.sqrt(d.T @ Bd)

        # Support vector formula for x^T B^-1 x <= 1
        return self.center + Bd / denom

    # Make it compatible with the previous code's list/matrix interface
    def __matmul__(self, d):
        # This is a hack to allow the code `A @ d` to call our support function
        # In the previous code, A was a list of points. Here A is an object.
        # We assume d is a single vector (3,)
        return self.support_function(d)


def generate_analytic_ellipsoid():
    return AnalyticEllipsoid()


def generate_ellipsoid(subdivisions=2):
    """
    Generates a random ellipsoid by sampling a Symmetric Positive Definite (SPD)
    matrix first, then applying the Cholesky factor to a unit sphere.
    """
    # 1. Create Unit Sphere (u)
    mesh = trimesh.creation.icosphere(subdivisions=subdivisions)
    unit_sphere_pts = mesh.vertices  # Shape (N, 3)

    M = np.random.randn(3, 3)
    A = M @ M.T
    # Positive Definite (strictly convex, no zero volume).
    A += 0.1 * np.eye(3)
    L = np.linalg.cholesky(A)

    # 4. Apply Transformation
    # Points x = L @ u. Since our points are rows (N,3), we do u @ L.T
    ellipsoid_pts = unit_sphere_pts @ L.T

    return ellipsoid_pts


def generate_polytope(n_vertices=100, radius=1.0):
    """
    Generates a random convex polytope represented by its vertices.
    Equivalent to taking the convex hull of random points on a sphere.
    """
    # Sample points on unit sphere
    vec = np.random.randn(n_vertices, 3)
    vec /= np.linalg.norm(vec, axis=1)[:, None]
    return vec * radius
