import numpy as np
import trimesh

from mnp_algos import fw, nesterov_fw


def compute_mean_trajectory(jagged_list):
    """
    Pads a list of lists with 0.0 to create a 2D numpy array
    and computes the mean along axis 0.
    """
    if not jagged_list:
        return np.array([])

    # 1. Find the maximum number of iterations among all pairs
    max_len = max(len(run) for run in jagged_list)

    # 2. Create a matrix of shape (n_pairs, max_len) filled with 0.0
    # We use 0.0 because once the algorithm finishes, the optimality gap is effectively 0.
    padded_array = np.zeros((len(jagged_list), max_len))

    # 3. Fill the matrix
    for i, run in enumerate(jagged_list):
        padded_array[i, : len(run)] = run
        # If you prefer to extend the last value instead of 0, uncomment below:
        # padded_array[i, len(run):] = run[-1]

    # 4. Compute mean along axis 0 (averaging across all pairs for each time step)
    return np.mean(padded_array, axis=0)


def generate_ellipsoid(n_points=100):
    """
    Generates a random ellipsoid by scaling a unit sphere.
    """
    # Create unit sphere
    mesh = trimesh.creation.icosphere(subdivisions=2)
    pts = mesh.vertices

    # Random Scaling (strictly convex)
    scale = np.random.uniform(0.5, 2.0, size=(3,))
    # Random Rotation
    rot = trimesh.transformations.random_rotation_matrix()[:3, :3]

    pts = (pts * scale) @ rot.T
    return pts


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

    closest_vec, _, _, _ = fw(x_0, A_verts, B_centered)
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
    distances = [-0.1, -0.01, 0.0, 0.001, 0.01, 0.1, 1.0]  #
    results = {
        "mean_iters_gjk": np.zeros(len(distances)),
        "mean_iters_nesterov": np.zeros(len(distances)),
        "std_iters_gjk": np.zeros(len(distances)),
        "std_iters_nesterov": np.zeros(len(distances)),
        "iters_opt_criterions_gjk": [],
        "iters_opt_criterions_nesterov": [],
    }

    print(f"Running benchmark on {n_pairs} pairs of Ellipsoids...")

    for i_d, d in enumerate(distances):
        iters_gjk = []
        iters_nesterov = []
        iters_opt_criterions_gjk = []
        iters_opt_criterions_nesterov = []

        for _ in range(n_pairs):
            A = generate_ellipsoid()
            B = generate_ellipsoid()

            # Setup specific distance
            A, B = setup_scene(A, B, d)

            x_0 = A.mean(axis=0) - B.mean(axis=0)

            # Run Vanilla
            _, k_gjk, _, opt_criterions_gjk = fw(x_0, A, B)
            iters_gjk.append(k_gjk)
            iters_opt_criterions_gjk.append(opt_criterions_gjk)

            # Run Nesterov (No normalization needed for Ellipsoids per paper)
            _, k_nest, _, opt_criterions_nest = nesterov_fw(x_0, A, B)
            iters_nesterov.append(k_nest)
            iters_opt_criterions_nesterov.append(opt_criterions_nest)

        results["mean_iters_gjk"][i_d] = np.mean(iters_gjk)
        results["mean_iters_nesterov"][i_d] = np.mean(iters_nesterov)
        results["std_iters_gjk"][i_d] = np.std(iters_gjk)
        results["std_iters_nesterov"][i_d] = np.std(iters_nesterov)
        results["iters_opt_criterions_gjk"].append(iters_opt_criterions_gjk)
        results["iters_opt_criterions_nesterov"].append(iters_opt_criterions_nesterov)

        print(
            f"Dist: {d}m | GJK Avg: {np.mean(iters_gjk):.2f} | Nest Avg: {np.mean(iters_nesterov):.2f}"
        )

    return distances, results
