import numpy as np

from geometry import generate_analytic_ellipsoid, generate_polytope
from mnp_algos import (
    corrective_fw,
    corrective_fw_analytic,
    nesterov_fw,
    nesterov_fw_analytic,
)


def setup_scene_analytic(shape_A, shape_B, target_dist):
    """
    Translates shape_B so that dist(A, B) == target_dist.
    Uses analytic GJK to find the separation axis.
    """
    # 1. Initial Separation
    # Move B far away to ensure no intersection initially
    # We create a copy or modify B in place? Let's modify position in place.

    # Reset B to a safe distance (e.g. 50 units along X axis relative to A)

    shape_B.center = shape_A.center + np.array([50.0, 0.0, 0.0])

    # 2. Run GJK to find closest vector (x*)
    x_0 = shape_A.center - shape_B.center  # Initial guess

    closest_vec, _, _, _ = corrective_fw_analytic(
        x_0, shape_A, shape_B, opt_criterion=10e-6
    )
    current_dist = np.linalg.norm(closest_vec)

    # Safety: If they intersect by accident (bad random shape), push further
    if current_dist < 1e-6:
        shape_B.center += np.array([20.0, 0.0, 0.0])
        # Recursively retry
        return setup_scene_analytic(shape_A, shape_B, target_dist)

    # 3. Translate B along the separation axis
    # closest_vec = x* = closest point in D = closest(A) - closest(B)
    # The vector points from B towards A (roughly) in configuration space

    normal = closest_vec / current_dist

    # We want to move B such that the new distance is target_dist.
    # We currently have distance `current_dist`.
    # We need to move B *towards* A by (current_dist - target_dist).
    # Since `normal` points outwards from the Minkowski difference origin towards the boundary,
    # it represents the direction to move to INCREASE separation.
    # Therefore, to adjust distance, we move along the normal.

    translation = normal * (current_dist - target_dist)

    # Apply translation
    shape_B.center = shape_B.center + translation

    return shape_A, shape_B


def setup_scene(A_verts, B_verts, target_dist):
    """
    Translates B so that dist(A, B) == target_dist.
    Strategies from paper: Sample relative poses, translate along separating axis. [cite: 445-448]
    """
    # 1. Initial Separation (move B far away to ensure no intersection initially)
    B_centered = B_verts - B_verts.mean(axis=0) + np.array([50.0, 0, 0])

    # 2. Compute current distance and closest points using Vanilla GJK
    # We use x_0 = center_A - center_B
    x_0 = A_verts.mean(axis=0) - B_centered.mean(axis=0)

    closest_vec, _, _, _ = corrective_fw(x_0, A_verts, B_centered)
    current_dist = np.linalg.norm(closest_vec)

    if current_dist < 1e-6:
        # If they accidentally intersect, move B further and retry
        return setup_scene(A_verts, B_verts + np.array([10.0, 0, 0]), target_dist)

    # 3. Translate B along the separation vector
    # closest_vec points from B to A (or A to B depending on definition).
    # We want to move B towards A by (current_dist - target_dist)

    normal = closest_vec / current_dist

    # Paper: "translate the shapes along the axis given by their closest-points" [cite: 447]
    # If target_dist is negative (intersecting), we move it deeper.
    translation = normal * (current_dist - target_dist)
    B_final = B_centered + translation

    return A_verts, B_final


def run_experiment_ellipsoids(n_pairs=100):
    distances = [-0.1, -0.01, -0.001, -0.0005, 0.0005, 0.001, 0.01, 0.1, 1.0]
    results = {
        "mean_iters_gjk": np.zeros(len(distances)),
        "mean_iters_nesterov": np.zeros(len(distances)),
        "mean_iters_nesterov_restart": np.zeros(len(distances)),
        "std_iters_gjk": np.zeros(len(distances)),
        "std_iters_nesterov": np.zeros(len(distances)),
        "std_iters_nesterov_restart": np.zeros(len(distances)),
        "iters_opt_criterions_gjk": [],
        "iters_opt_criterions_nesterov": [],
    }

    print(f"Running benchmark on {n_pairs} pairs of Ellipsoids...")

    tol = 10e-6

    for i_d, d in enumerate(distances):
        iters_gjk = []
        iters_nesterov = []
        iters_nesterov_restart = []

        for _ in range(n_pairs):
            A = generate_analytic_ellipsoid()
            B = generate_analytic_ellipsoid()

            # Setup specific distance
            A, B = setup_scene_analytic(A, B, d)

            x_0 = A.center - B.center

            # Run Vanilla
            _, k_gjk, _, opt_criterions_gjk = corrective_fw_analytic(
                x_0, A, B, opt_criterion=tol
            )
            iters_gjk.append(k_gjk)

            # Run Nesterov (No normalization needed for Ellipsoids per paper)
            _, k_nest, _, opt_criterions_nest = nesterov_fw_analytic(
                x_0, A, B, opt_criterion=tol
            )
            iters_nesterov.append(k_nest)

            _, k_nest_restart, _, _ = nesterov_fw_analytic(
                x_0, A, B, restart=True, opt_criterion=tol
            )
            iters_nesterov_restart.append(k_nest_restart)

        results["mean_iters_gjk"][i_d] = np.mean(iters_gjk)
        results["mean_iters_nesterov"][i_d] = np.mean(iters_nesterov)
        results["mean_iters_nesterov_restart"][i_d] = np.mean(iters_nesterov_restart)
        results["std_iters_gjk"][i_d] = np.std(iters_gjk)
        results["std_iters_nesterov"][i_d] = np.std(iters_nesterov)
        results["std_iters_nesterov_restart"][i_d] = np.std(iters_nesterov_restart)

        print(
            f"Dist: {d}m | GJK Avg: {np.mean(iters_gjk):.2f} | Nest Avg: {np.mean(iters_nesterov):.2f}"
        )

    return distances, results


def run_experiment_meshes(n_pairs=100, n_vertices=100):
    distances = [-0.1, -0.01, 0.0, 0.001, 0.01, 0.1, 1.0]
    results = {
        "mean_iters_gjk": np.zeros(len(distances)),
        "mean_iters_nesterov": np.zeros(len(distances)),
        "mean_iters_nesterov_norm": np.zeros(len(distances)),
        "mean_iters_nesterov_norm_restart": np.zeros(len(distances)),
        "std_iters_gjk": np.zeros(len(distances)),
        "std_iters_nesterov": np.zeros(len(distances)),
        "std_iters_nesterov_norm": np.zeros(len(distances)),
        "std_iters_nesterov_norm_restart": np.zeros(len(distances)),
    }

    print(f"Running benchmark on {n_pairs} pairs of Polytopes (N={n_vertices})...")

    for i_d, d in enumerate(distances):
        iters_gjk = []
        iters_nest = []
        iters_nest_norm = []
        iters_nest_norm_restart = []

        for _ in range(n_pairs):
            A = generate_polytope(n_vertices)
            B = generate_polytope(n_vertices)

            try:
                A, B = setup_scene(A, B, d)
            except RecursionError:
                continue  # Skip bad seeds

            x_0 = A.mean(axis=0) - B.mean(axis=0)

            # 1. Vanilla GJK
            _, k_gjk, _, _ = corrective_fw(x_0, A, B)
            iters_gjk.append(k_gjk)

            # 2. Nesterov (Original - likely to fail/stall on meshes)
            _, k_nest, _, _ = nesterov_fw(x_0, A, B, normalization=False)
            iters_nest.append(k_nest)

            # 3. Nesterov (Normalized - Recommended for meshes)
            _, k_norm, _, _ = nesterov_fw(x_0, A, B, normalization=True)
            iters_nest_norm.append(k_norm)

            _, k_norm_restart, _, _ = nesterov_fw(
                x_0, A, B, normalization=True, restart=True
            )
            iters_nest_norm_restart.append(k_norm_restart)

        results["mean_iters_gjk"][i_d] = np.mean(iters_gjk)
        results["mean_iters_nesterov"][i_d] = np.mean(iters_nest)
        results["mean_iters_nesterov_norm"][i_d] = np.mean(iters_nest_norm)
        results["mean_iters_nesterov_norm_restart"][i_d] = np.mean(
            iters_nest_norm_restart
        )
        results["std_iters_gjk"][i_d] = np.std(iters_gjk)
        results["std_iters_nesterov"][i_d] = np.std(iters_nest)
        results["std_iters_nesterov_norm"][i_d] = np.std(iters_nest_norm)
        results["std_iters_nesterov_norm_restart"] = np.std(iters_nest_norm_restart)

        print(
            f"Dist: {d}m | GJK: {np.mean(iters_gjk):.1f} | Nest: {np.mean(iters_nest):.1f} | Nest+Norm: {np.mean(iters_nest_norm):.1f}"
        )

    return distances, results
