import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from scipy.interpolate import splev, splprep
from scipy.ndimage import gaussian_filter1d
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA


SAVE_DIR = "results/figures"


class Arrow3D(FancyArrowPatch):
    """A 3D arrow projected onto Matplotlib's 2D drawing surface."""

    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)


def _figure_path(filename):
    os.makedirs(SAVE_DIR, exist_ok=True)
    return os.path.join(SAVE_DIR, filename)


def _average_rate_matrix(spikes_list, num_neurons=10, bin_size=100, max_time=5e4, smooth_sigma=2.0):
    if not spikes_list:
        raise ValueError("spikes_list is empty; cannot build a firing-rate matrix.")

    bins = np.arange(0, max_time + bin_size, bin_size)
    all_rate_matrices = []

    for senders, times in spikes_list:
        senders = np.asarray(senders)
        times = np.asarray(times)
        rate_matrix = np.zeros((len(bins) - 1, num_neurons))

        for neuron_idx in range(num_neurons):
            neuron_id = neuron_idx + 1
            neuron_times = times[senders == neuron_id]
            counts, _ = np.histogram(neuron_times, bins=bins)
            rate_matrix[:, neuron_idx] = counts / (bin_size / 1000.0)

        all_rate_matrices.append(rate_matrix)

    avg_rate_matrix = np.mean(all_rate_matrices, axis=0)
    if smooth_sigma and smooth_sigma > 0:
        avg_rate_matrix = gaussian_filter1d(avg_rate_matrix, sigma=smooth_sigma, axis=0)

    return avg_rate_matrix


def _smooth_3d_trajectory(trajectory_3d, n_points=1000):
    if len(trajectory_3d) < 4:
        return trajectory_3d[:, 0], trajectory_3d[:, 1], trajectory_3d[:, 2]

    diffs = np.diff(trajectory_3d, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    filtered_traj = trajectory_3d[np.insert(dists > 1e-8, 0, True)]

    if len(filtered_traj) < 4:
        return trajectory_3d[:, 0], trajectory_3d[:, 1], trajectory_3d[:, 2]

    try:
        spline_order = min(3, len(filtered_traj) - 1)
        tck, _ = splprep(
            [filtered_traj[:, 0], filtered_traj[:, 1], filtered_traj[:, 2]],
            s=5.0,
            k=spline_order,
        )
        u_fine = np.linspace(0, 1, n_points)
        return splev(u_fine, tck)
    except ValueError:
        return trajectory_3d[:, 0], trajectory_3d[:, 1], trajectory_3d[:, 2]


def plot_attractor_activity(spikes_test, rate_val, phase_name, trial_idx=1, num_neurons=10, max_time=5e4):
    """Plot the spike activation pattern for one trial."""
    senders, times = spikes_test
    senders = np.asarray(senders)
    times = np.asarray(times)

    color = "royalblue" if phase_name == "Training" else "darkorange"

    fig, ax = plt.subplots(figsize=(12, 3), facecolor="white")
    ax.set_facecolor("white")

    if len(times) > 0:
        ax.scatter(times, senders, s=10, color=color, alpha=0.8)
    else:
        ax.text(
            0.5,
            0.5,
            "No spikes recorded",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="gray",
        )

    ax.set_title(f"Activation Pattern ({phase_name}) - Rate={rate_val} [Trial {trial_idx}]")
    ax.set_xlabel("Time in quiet period (ms)")
    ax.set_ylabel("Exc neuron ID")
    ax.set_xlim(0, max_time)
    ax.set_ylim(0.5, num_neurons + 0.5)
    ax.set_yticks(range(1, num_neurons + 1))
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    filepath = _figure_path(f"Activation_{phase_name}_Rate{rate_val}.pdf")
    plt.savefig(filepath, format="pdf", bbox_inches="tight", facecolor="white")
    plt.show()


def analyze_averaged_attractor_trajectory(
    spikes_list,
    rate_val,
    phase_name,
    num_neurons=10,
    bin_size=100,
    max_time=5e4,
    color="red",
):
    """Plot the averaged firing-rate PCA trajectory and return the 3D PCA trajectory."""
    rate_matrix = _average_rate_matrix(
        spikes_list,
        num_neurons=num_neurons,
        bin_size=bin_size,
        max_time=max_time,
    )

    if np.sum(rate_matrix) == 0:
        print(f"[{phase_name}] No firing activity was recorded, so PCA cannot be plotted.")
        return None

    pca = PCA(n_components=3)
    trajectory_3d = pca.fit_transform(rate_matrix)
    x_smooth, y_smooth, z_smooth = _smooth_3d_trajectory(trajectory_3d)

    with plt.style.context("default"):
        fig = plt.figure(figsize=(12, 10), facecolor="white")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("white")

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

        ax.plot(x_smooth, y_smooth, z_smooth, color=color, linewidth=2.0, alpha=0.9)
        ax.scatter(x_smooth[0], y_smooth[0], z_smooth[0], marker="o", color="green", s=60, zorder=5)

        if len(x_smooth) >= 10:
            dx = x_smooth[-1] - x_smooth[-10]
            dy = y_smooth[-1] - y_smooth[-10]
            dz = z_smooth[-1] - z_smooth[-10]
        else:
            dx = x_smooth[-1] - x_smooth[0]
            dy = y_smooth[-1] - y_smooth[0]
            dz = z_smooth[-1] - z_smooth[0]

        if np.linalg.norm([dx, dy, dz]) > 0:
            px, py, pz = x_smooth[-1], y_smooth[-1], z_smooth[-1]
            arrow = Arrow3D(
                [px, px + dx],
                [py, py + dy],
                [pz, pz + dz],
                mutation_scale=20,
                lw=2.0,
                arrowstyle="-|>",
                color=color,
                zorder=20,
            )
            ax.add_artist(arrow)

        ax.set_title(f"Firing-rate PCA Trajectory - {phase_name} (Rate={rate_val})", color="black")
        ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0] * 100:.1f}%)", color="black")
        ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1] * 100:.1f}%)", color="black")
        ax.set_zlabel(f"PC 3 ({pca.explained_variance_ratio_[2] * 100:.1f}%)", color="black")
        ax.tick_params(colors="black")

        plt.tight_layout()
        filepath = _figure_path(f"PCA_3D_{phase_name}_Rate{rate_val}.pdf")
        plt.savefig(filepath, format="pdf", bbox_inches="tight", facecolor="white")
        plt.show()

    return trajectory_3d


def get_pca_trajectory(spikes_list, num_neurons=10, bin_size=100, max_time=5e4):
    """Return the same averaged firing-rate PCA trajectory used by the PCA plot."""
    rate_matrix = _average_rate_matrix(
        spikes_list,
        num_neurons=num_neurons,
        bin_size=bin_size,
        max_time=max_time,
    )

    if np.sum(rate_matrix) == 0:
        return None

    pca = PCA(n_components=3)
    return pca.fit_transform(rate_matrix)


def get_full_rate_matrix(spikes_list, num_neurons=10, bin_size=100, max_time=5e4):
    """Return the smoothed averaged firing-rate matrix with shape (time, neurons)."""
    return _average_rate_matrix(
        spikes_list,
        num_neurons=num_neurons,
        bin_size=bin_size,
        max_time=max_time,
    )


def plot_attractor_energy_landscape(pca_traj, phase_name="Consolidation", rate_val=50):
    """Plot a 3D energy landscape from the first two PCA components."""
    if pca_traj is None:
        print(f"[{phase_name}] PCA trajectory is empty, so the energy landscape cannot be plotted.")
        return

    pca_traj = np.asarray(pca_traj)
    if pca_traj.ndim != 2 or pca_traj.shape[1] < 2 or len(pca_traj) < 3:
        raise ValueError("pca_traj must have at least three samples and two PCA components.")

    x = pca_traj[:, 0]
    y = pca_traj[:, 1]

    xmin, xmax = x.min() - 0.4, x.max() + 0.4
    ymin, ymax = y.min() - 0.4, y.max() + 0.4
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])

    try:
        kernel = gaussian_kde(values)
    except np.linalg.LinAlgError:
        rng = np.random.default_rng(0)
        kernel = gaussian_kde(values + rng.normal(scale=1e-6, size=values.shape))

    density = np.reshape(kernel(positions).T, X.shape)
    energy = -np.log(density + 1e-12)
    energy -= np.nanmin(energy)

    with plt.style.context("default"):
        fig = plt.figure(figsize=(12, 10), facecolor="white")
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor("white")
        ax.view_init(elev=45, azim=135)

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

        ax.plot_surface(X, Y, energy, cmap="coolwarm", linewidth=0, antialiased=True, alpha=0.9)
        ax.set_zticks([])
        ax.set_title(f"Energy Landscape ({phase_name})", fontsize=15, pad=20)
        ax.set_xlabel("PC 1", fontsize=12)
        ax.set_ylabel("PC 2", fontsize=12)

        plt.tight_layout()
        filepath = _figure_path(f"Energy_Landscape_{phase_name}_Rate{rate_val}.pdf")
        plt.savefig(filepath, format="pdf", bbox_inches="tight", facecolor="white", dpi=300)
        plt.show()


def calculate_lyapunov_exponent(traj_train, traj_cons, rate_val, dt_ms=100.0):
    """Estimate and plot the maximum Lyapunov exponent from firing-rate trajectories."""

    def estimate_mle(trajectory):
        trajectory = np.asarray(trajectory, dtype=float)
        n_samples = len(trajectory)

        if n_samples < 20:
            steps = max(2, n_samples // 4)
            return 0.0, np.arange(steps) * dt, np.zeros(steps), steps

        max_steps = max(5, min(50, int(n_samples * 0.25)))
        theiler_window = max(2, int(n_samples * 0.05))
        start_idx = int(n_samples * 0.2)
        stop_idx = max(start_idx + 1, n_samples - max_steps)
        search_space = trajectory[start_idx:stop_idx]

        divergences = []
        for local_i, state in enumerate(search_space):
            global_i = start_idx + local_i
            dists = np.linalg.norm(search_space - state, axis=1)

            win_start = max(0, local_i - theiler_window)
            win_end = min(len(search_space), local_i + theiler_window + 1)
            dists[win_start:win_end] = np.inf

            if np.isinf(np.min(dists)):
                continue

            nearest_idx = start_idx + np.argmin(dists)
            div_t = []
            for step in range(max_steps):
                if global_i + step >= n_samples or nearest_idx + step >= n_samples:
                    break
                dist = np.linalg.norm(trajectory[global_i + step] - trajectory[nearest_idx + step])
                div_t.append(dist)

            if len(div_t) == max_steps:
                divergences.append(div_t)

        if not divergences:
            print("Warning: no valid nearest-neighbor pairs found; defaulting MLE to 0.")
            return 0.0, np.arange(max_steps) * dt, np.zeros(max_steps), max_steps

        divergences = np.asarray(divergences)
        mean_log_div = np.mean(np.log(divergences + 1e-9), axis=0)
        time_axis = np.arange(max_steps) * dt

        fit_steps = max(5, int(max_steps * 0.3))
        slope, _ = np.polyfit(time_axis[:fit_steps], mean_log_div[:fit_steps], 1)

        return slope, time_axis, mean_log_div, fit_steps

    dt = dt_ms / 1000.0
    mle_train, t_train, div_train, fit_idx_train = estimate_mle(traj_train)
    mle_cons, t_cons, div_cons, fit_idx_cons = estimate_mle(traj_cons)

    plt.figure(figsize=(8, 6), facecolor="white")
    plt.plot(t_train, div_train, color="blue", alpha=0.7, label=f"Training (MLE: {mle_train:.3f})")
    fit_line_train = mle_train * t_train[:fit_idx_train] + (div_train[0] - mle_train * t_train[0])
    plt.plot(t_train[:fit_idx_train], fit_line_train, "b--", linewidth=2)

    plt.plot(
        t_cons,
        div_cons,
        color="red",
        alpha=0.9,
        linewidth=2,
        label=f"Consolidation (MLE: {mle_cons:.3f})",
    )
    fit_line_cons = mle_cons * t_cons[:fit_idx_cons] + (div_cons[0] - mle_cons * t_cons[0])
    plt.plot(t_cons[:fit_idx_cons], fit_line_cons, "r--", linewidth=2)

    plt.title("Lyapunov Exponent Estimation (Log-Divergence)", fontsize=14)
    plt.xlabel("Time Horizon (s)", fontsize=12)
    plt.ylabel("<ln(distance)>", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    filepath = _figure_path(f"Lyapunov_Exponent_Rate{rate_val}.pdf")
    plt.savefig(filepath, format="pdf", bbox_inches="tight", facecolor="white", dpi=300)
    plt.show()

    print("\n" + "=" * 50)
    print("Maximum Lyapunov Exponent Estimate")
    print("=" * 50)
    print(f"Training MLE: {mle_train:.4f}")
    print(f"Consolidation MLE: {mle_cons:.4f}")
    print("=" * 50)

    return {
        "training_mle": mle_train,
        "consolidation_mle": mle_cons,
    }


def merge_functional_receptive_field_data(rf_data_list):
    """Merge RF probe dictionaries returned by multiple simulation trials."""
    if not rf_data_list:
        raise ValueError("rf_data_list is empty; cannot merge functional RF data.")

    merged_spikes = {}
    stimulus_time = rf_data_list[0].get("stimulus_time")
    quiet_time = rf_data_list[0].get("quiet_time")

    for rf_data in rf_data_list:
        for center, trials in rf_data["spikes_by_center"].items():
            center = float(center)
            merged_spikes.setdefault(center, [])
            merged_spikes[center].extend(trials)

    stimulus_centers = np.asarray(sorted(merged_spikes), dtype=float)
    return {
        "stimulus_centers": stimulus_centers,
        "spikes_by_center": {float(center): merged_spikes[float(center)] for center in stimulus_centers},
        "stimulus_time": stimulus_time,
        "quiet_time": quiet_time,
    }


def compute_functional_receptive_field(
    rf_data,
    stimulus_centers=None,
    num_neurons=10,
    response_window=None,
    window_ms=None,
):
    """Compute a neuron-by-stimulus firing-rate tuning matrix from labeled RF spikes."""
    if isinstance(rf_data, dict) and "spikes_by_center" in rf_data:
        spikes_by_center = rf_data["spikes_by_center"]
        if stimulus_centers is None:
            stimulus_centers = rf_data.get("stimulus_centers")
        if window_ms is None:
            window_ms = rf_data.get("stimulus_time")
    else:
        spikes_by_center = rf_data

    if isinstance(spikes_by_center, dict):
        if stimulus_centers is None:
            stimulus_centers = sorted(float(center) for center in spikes_by_center)
        stimulus_centers = np.asarray(stimulus_centers, dtype=float)
        trials_by_center = []
        for center in stimulus_centers:
            trials = spikes_by_center.get(center, spikes_by_center.get(float(center), None))
            if trials is None:
                trials = spikes_by_center.get(str(center), [])
            trials_by_center.append(trials)
    else:
        if stimulus_centers is None:
            raise ValueError("stimulus_centers is required when rf_data is not a dict.")
        stimulus_centers = np.asarray(stimulus_centers, dtype=float)
        trials_by_center = list(spikes_by_center)

    if response_window is None:
        if window_ms is None:
            raise ValueError("window_ms is required when response_window is not provided.")
        start_ms = 0.0
        end_ms = float(window_ms)
    else:
        start_ms, end_ms = response_window
        start_ms = float(start_ms)
        end_ms = float(end_ms)

    duration_s = (end_ms - start_ms) / 1000.0
    if duration_s <= 0:
        raise ValueError("response_window/window_ms must define a positive duration.")

    tuning_matrix = np.zeros((num_neurons, len(stimulus_centers)))

    for center_idx, trials in enumerate(trials_by_center):
        if isinstance(trials, tuple) and len(trials) == 2:
            trials = [trials]

        trial_rates = []
        for senders, times in trials:
            senders = np.asarray(senders)
            times = np.asarray(times)
            mask = (times >= start_ms) & (times < end_ms)
            senders = senders[mask]

            counts = np.zeros(num_neurons)
            for neuron_idx in range(num_neurons):
                counts[neuron_idx] = np.sum(senders == neuron_idx + 1)
            trial_rates.append(counts / duration_s)

        if trial_rates:
            tuning_matrix[:, center_idx] = np.mean(trial_rates, axis=0)

    peak_idx = np.argmax(tuning_matrix, axis=1)
    preferred_stimulus = stimulus_centers[peak_idx]
    peak_rate = tuning_matrix[np.arange(num_neurons), peak_idx]
    mean_other_rate = np.zeros(num_neurons)
    rf_width = np.zeros(num_neurons)
    selectivity_index = np.zeros(num_neurons)

    for neuron_idx in range(num_neurons):
        rates = tuning_matrix[neuron_idx]
        if len(rates) > 1:
            mean_other_rate[neuron_idx] = (np.sum(rates) - peak_rate[neuron_idx]) / (len(rates) - 1)
        selectivity_index[neuron_idx] = (
            (peak_rate[neuron_idx] - mean_other_rate[neuron_idx])
            / (peak_rate[neuron_idx] + mean_other_rate[neuron_idx] + 1e-9)
        )

        total_rate = np.sum(rates)
        if total_rate > 0:
            center_offset = stimulus_centers - preferred_stimulus[neuron_idx]
            rf_width[neuron_idx] = np.sqrt(np.sum(rates * center_offset ** 2) / total_rate)
        else:
            rf_width[neuron_idx] = np.nan

    return {
        "stimulus_centers": stimulus_centers,
        "tuning_matrix": tuning_matrix,
        "preferred_stimulus": preferred_stimulus,
        "peak_rate": peak_rate,
        "mean_other_rate": mean_other_rate,
        "selectivity_index": selectivity_index,
        "rf_width": rf_width,
        "response_window": (start_ms, end_ms),
    }


def plot_functional_receptive_field(rf_result, rate_val=None, phase_name="Functional_RF"):
    """Plot a functional RF heatmap and each neuron's tuning curve."""
    stimulus_centers = np.asarray(rf_result["stimulus_centers"], dtype=float)
    tuning_matrix = np.asarray(rf_result["tuning_matrix"], dtype=float)
    preferred_stimulus = np.asarray(rf_result["preferred_stimulus"], dtype=float)
    peak_rate = np.asarray(rf_result["peak_rate"], dtype=float)
    num_neurons = tuning_matrix.shape[0]
    num_stimuli = len(stimulus_centers)

    heatmap_width = max(4.5, num_stimuli * 0.55)
    heatmap_height = max(4.5, num_neurons * 0.55)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(heatmap_width + 7.0, heatmap_height),
        facecolor="white",
        gridspec_kw={"width_ratios": [heatmap_width, 7.0]},
    )

    im = axes[0].imshow(tuning_matrix, aspect="equal", origin="lower", cmap="viridis")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].scatter(np.argmax(tuning_matrix, axis=1), np.arange(num_neurons), color="white", s=18, label="Preferred")
    axes[0].set_title(f"Functional Receptive Field ({phase_name})")
    axes[0].set_xlabel("Stimulus center")
    axes[0].set_ylabel("Excitatory neuron ID")
    axes[0].set_xticks(range(len(stimulus_centers)))
    axes[0].set_xticklabels([f"{center:g}" for center in stimulus_centers], rotation=45, ha="right")
    axes[0].set_yticks(range(num_neurons))
    axes[0].set_yticklabels(range(1, num_neurons + 1))
    axes[0].legend(loc="upper right", frameon=False)
    cbar = fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
    cbar.set_label("Firing rate (Hz)")

    colors = plt.cm.tab10(np.linspace(0, 1, num_neurons))
    for neuron_idx in range(num_neurons):
        axes[1].plot(
            stimulus_centers,
            tuning_matrix[neuron_idx],
            marker="o",
            linewidth=1.5,
            color=colors[neuron_idx % len(colors)],
            label=f"N{neuron_idx + 1}",
        )
        axes[1].scatter(
            preferred_stimulus[neuron_idx],
            peak_rate[neuron_idx],
            color=colors[neuron_idx % len(colors)],
            s=28,
        )

    axes[1].set_title("Tuning Curves")
    axes[1].set_xlabel("Stimulus center")
    axes[1].set_ylabel("Firing rate (Hz)")
    axes[1].grid(True, linestyle="--", alpha=0.4)
    if num_neurons <= 10:
        axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)

    plt.tight_layout()
    suffix = f"_Rate{rate_val}" if rate_val is not None else ""
    filepath = _figure_path(f"Functional_Receptive_Field_{phase_name}{suffix}.pdf")
    plt.savefig(filepath, format="pdf", bbox_inches="tight", facecolor="white", dpi=300)
    plt.show()


def analyze_functional_receptive_field(
    rf_data,
    rate_val=None,
    phase_name="Functional_RF",
    stimulus_centers=None,
    num_neurons=10,
    response_window=None,
    window_ms=None,
):
    """Compute, plot, and print a compact functional RF summary."""
    rf_result = compute_functional_receptive_field(
        rf_data,
        stimulus_centers=stimulus_centers,
        num_neurons=num_neurons,
        response_window=response_window,
        window_ms=window_ms,
    )
    plot_functional_receptive_field(rf_result, rate_val=rate_val, phase_name=phase_name)

    print("\n" + "=" * 50)
    print(f"Functional Receptive Field Summary ({phase_name})")
    print("=" * 50)
    for neuron_idx, (pref, width, si, peak) in enumerate(
        zip(
            rf_result["preferred_stimulus"],
            rf_result["rf_width"],
            rf_result["selectivity_index"],
            rf_result["peak_rate"],
        ),
        start=1,
    ):
        print(
            f"N{neuron_idx:02d}: preferred={pref:g}, width={width:.2f}, "
            f"selectivity={si:.3f}, peak={peak:.2f} Hz"
        )
    print("=" * 50)

    return rf_result


def compare_functional_receptive_fields(rf_train_result, rf_cons_result, rate_val=None):
    """Plot and summarize functional RF changes from post-training to post-consolidation."""
    train_centers = np.asarray(rf_train_result["stimulus_centers"], dtype=float)
    cons_centers = np.asarray(rf_cons_result["stimulus_centers"], dtype=float)
    if not np.array_equal(train_centers, cons_centers):
        raise ValueError("Training and consolidation RF results must use the same stimulus centers.")

    train_matrix = np.asarray(rf_train_result["tuning_matrix"], dtype=float)
    cons_matrix = np.asarray(rf_cons_result["tuning_matrix"], dtype=float)
    if train_matrix.shape != cons_matrix.shape:
        raise ValueError("Training and consolidation tuning matrices must have the same shape.")

    delta_matrix = cons_matrix - train_matrix
    num_neurons = train_matrix.shape[0]
    vmax = max(np.nanmax(train_matrix), np.nanmax(cons_matrix), 1e-9)
    delta_abs = max(abs(np.nanmin(delta_matrix)), abs(np.nanmax(delta_matrix)), 1e-9)

    num_stimuli = len(train_centers)
    fig_width = max(12.0, num_stimuli * 1.8)
    fig_height = max(4.5, num_neurons * 0.55)
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), facecolor="white")
    heatmaps = [
        (train_matrix, "Post-training RF", "viridis", 0.0, vmax),
        (cons_matrix, "Post-consolidation RF", "viridis", 0.0, vmax),
        (delta_matrix, "RF Change (Consolidation - Training)", "coolwarm", -delta_abs, delta_abs),
    ]

    for ax, (matrix, title, cmap, vmin, vmax_i) in zip(axes, heatmaps):
        im = ax.imshow(matrix, aspect="equal", origin="lower", cmap=cmap, vmin=vmin, vmax=vmax_i)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(title)
        ax.set_xlabel("Stimulus center")
        ax.set_ylabel("Excitatory neuron ID")
        ax.set_xticks(range(len(train_centers)))
        ax.set_xticklabels([f"{center:g}" for center in train_centers], rotation=45, ha="right")
        ax.set_yticks(range(num_neurons))
        ax.set_yticklabels(range(1, num_neurons + 1))
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Firing rate (Hz)")

    plt.tight_layout()
    suffix = f"_Rate{rate_val}" if rate_val is not None else ""
    filepath = _figure_path(f"Functional_Receptive_Field_Change{suffix}.pdf")
    plt.savefig(filepath, format="pdf", bbox_inches="tight", facecolor="white", dpi=300)
    plt.show()

    preferred_shift = rf_cons_result["preferred_stimulus"] - rf_train_result["preferred_stimulus"]
    selectivity_delta = rf_cons_result["selectivity_index"] - rf_train_result["selectivity_index"]
    width_delta = rf_cons_result["rf_width"] - rf_train_result["rf_width"]

    print("\n" + "=" * 50)
    print("Functional RF Change Summary")
    print("=" * 50)
    for neuron_idx, (shift, d_si, d_width) in enumerate(
        zip(preferred_shift, selectivity_delta, width_delta),
        start=1,
    ):
        print(
            f"N{neuron_idx:02d}: preferred shift={shift:g}, "
            f"selectivity delta={d_si:.3f}, width delta={d_width:.2f}"
        )
    print("=" * 50)

    return {
        "delta_matrix": delta_matrix,
        "preferred_shift": preferred_shift,
        "selectivity_delta": selectivity_delta,
        "width_delta": width_delta,
    }
