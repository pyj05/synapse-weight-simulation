import random
import matplotlib.pyplot as plt
import nest
import numpy as np
import traceback


resolution = 0.05
delay = resolution

# Duration of the three stages (ms)
training_time = 1.0e4         # training
consolidation_time = 3.0e5    # consolidation
retrieval_time = 1.0e4        # recall

# Neuron parameters
vip_inh_params = {
    "V_th": -46.0,
    "g_m": 2.67,
    "E_L": -72.0,
    "C_m": 57.9,
    "t_ref": 3.75,
    "th_spike_add": 25.4,
    "th_spike_decay": 0.029,
    "voltage_reset_fraction": 0.47,
    "voltage_reset_add": -4.15,
    "th_voltage_index": 7.1192,
    "th_voltage_decay": 32.36,
    "spike_dependent_threshold": True,
    "after_spike_currents": True,
    "adapting_threshold": True
}

Ctgf_exc_params = {
    "V_th": -46.0,
    "g_m": 3.29,
    "E_L": -71.8,
    "C_m": 118.0,
    "t_ref": 8.78,
    "th_spike_add": 24.8,
    "th_spike_decay": 0.088,
    "voltage_reset_fraction": 0.94,
    "voltage_reset_add": 11.3,
    "th_voltage_index": 6.1689,
    "th_voltage_decay": 25.70,
    "spike_dependent_threshold": True,
    "after_spike_currents": True,
    "adapting_threshold": True
}

def get_exc_exc_weight_matrix(exc_neuron):
    """
    Get the excitatory-to-excitatory weight matrix.

    Rows and columns follow the order of ``exc_neuron``. Missing connections
    and autapses are left as 0.
    """
    try:
        exc_gids = np.asarray(exc_neuron.tolist(), dtype=int)
    except AttributeError:
        exc_gids = np.asarray(list(exc_neuron), dtype=int)

    weight_matrix = np.zeros((len(exc_gids), len(exc_gids)))
    gid_to_idx = {gid: idx for idx, gid in enumerate(exc_gids)}

    exc_conns = nest.GetConnections(exc_neuron, exc_neuron)

    exc_senders = np.asarray(exc_conns.source, dtype=int)
    exc_targets = np.asarray(exc_conns.target, dtype=int)
    exc_weights = np.asarray(exc_conns.weight, dtype=float)

    for sender, target, weight in zip(exc_senders, exc_targets, exc_weights):
        sender_idx = gid_to_idx.get(sender)
        target_idx = gid_to_idx.get(target)
        if sender_idx is None or target_idx is None or sender_idx == target_idx:
            continue
        weight_matrix[sender_idx, target_idx] = weight

    return weight_matrix


def _run_functional_receptive_field_probe(
    pg,
    spike_recorder,
    exc_neuron,
    pg_rate,
    stimulus_centers=None,
    n_repeats=3,
    stimulus_time=500.0,
    quiet_time=200.0,
    input_sigma=10.0,
):
    """Probe evoked responses to labeled input centers after the current network stage."""
    if stimulus_centers is None:
        stimulus_centers = [25 + 50 * k for k in range(len(exc_neuron))]

    try:
        exc_gids = np.asarray(exc_neuron.tolist(), dtype=int)
    except AttributeError:
        exc_gids = np.asarray(list(exc_neuron), dtype=int)

    gid_to_local_id = {gid: idx + 1 for idx, gid in enumerate(exc_gids)}
    rf_spikes_by_center = {float(center): [] for center in stimulus_centers}

    def set_center_rate(center):
        for j in range(500):
            pg[j].rate = pg_rate * np.exp(-((j - center) ** 2) / (2 * (input_sigma ** 2)))

    def silence_input():
        for k in range(500):
            pg[k].rate = 0.0

    for _ in range(n_repeats):
        for center in stimulus_centers:
            set_center_rate(center)
            stim_start_time = nest.biological_time
            nest.Simulate(stimulus_time)
            stim_end_time = nest.biological_time

            events = nest.GetStatus(spike_recorder, "events")[0]
            senders = events["senders"]
            times = events["times"]

            mask = (times >= stim_start_time) & (times < stim_end_time) & np.isin(senders, exc_gids)
            local_senders = np.array([gid_to_local_id[int(sender)] for sender in senders[mask]], dtype=int)
            local_times = times[mask] - stim_start_time
            rf_spikes_by_center[float(center)].append((local_senders, local_times))

            if quiet_time > 0:
                silence_input()
                nest.Simulate(quiet_time)

    silence_input()
    return {
        "stimulus_centers": np.asarray(stimulus_centers, dtype=float),
        "spikes_by_center": rf_spikes_by_center,
        "stimulus_time": float(stimulus_time),
        "quiet_time": float(quiet_time),
    }


def run_single_simulation(
    pg_rate,
    test_early_attractor=False,
    return_functional_rf=False,
    rf_stimulus_centers=None,
    rf_repeats=3,
    rf_stimulus_time=500.0,
    rf_quiet_time=200.0,
):
    """
    Run a single network simulation, setting the sinusoidal_poisson_generator 's `rate` to `pg_rate`.
    Return the excitatory weight matrices for the three stages (w_enc, w_cons, w_recall).
    """

    random.seed(42)
    np.random.seed(42)
    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": resolution, "local_num_threads": 1, "rng_seed": 42})

    # Create neuron
    exc_neuron = nest.Create("glif_cond", 10, Ctgf_exc_params)
    inh_neuron = nest.Create("glif_cond", 3, vip_inh_params)

    # Create voltmeter
    voltmeter = nest.Create("voltmeter")
    spike_recorder = nest.Create("spike_recorder")
    wr = nest.Create("weight_recorder")

    # Poisson->Parrot
    pop_input = nest.Create("parrot_neuron", 500)

    pg = nest.Create("sinusoidal_poisson_generator", 500, params={
        "rate": pg_rate,
        "amplitude": 50.0,
        "frequency": 0.0,
        "phase": 0
        #"individual_spike_trains": False
    })

    nest.Connect(pg, pop_input, "one_to_one", {
        "synapse_model": "static_synapse",
        "weight": 1.0,
        "delay": delay
    })

    nest.SetStatus(exc_neuron, {"tau_syn": [0.2, 2.0], "E_rev": [0.0, -85.0]})
    nest.SetStatus(inh_neuron, {"tau_syn": [0.2, 2.0], "E_rev": [0.0, -85.0]})

    # input->exc (stdp)
    nest.CopyModel("stdp_triplet_synapse", "stdp_triplet_synapse_input_to_exc", {"Wmax": 5.0})
    conn_dict_input_to_exc = {"rule": "all_to_all"}
    syn_dict_input_to_exc = {
        "synapse_model": "stdp_triplet_synapse_input_to_exc",
        "weight": nest.random.uniform(0.5, 2.0),
        "delay": delay,
        "receptor_type": 1
    }
    nest.Connect(pop_input, exc_neuron, conn_dict_input_to_exc, syn_dict_input_to_exc)

    # input->inh (static)
    conn_dict_input_to_inh = {"rule": "all_to_all"}
    syn_dict_input_to_inh = {
        "synapse_model": "static_synapse",
        "weight": nest.random.uniform(0.0, 0.5),
        "delay": delay,
        "receptor_type": 1
    }
    nest.Connect(pop_input, inh_neuron, conn_dict_input_to_inh, syn_dict_input_to_inh)

    # exc<->exc, exc->inh, inh->exc, inh->inh parameters
    weight_EE = 0.1
    weight_EI = 0.6
    weight_IE = 0.2
    weight_II = 0.2
    delay_int = 1.0

    # Exc->Exc (stdp)
    nest.CopyModel("stdp_triplet_synapse", "stdp_triplet_synapse_exc_to_exc",
                   {"Wmax": 5.0, "weight_recorder": wr})
    syn_dict_exc_to_exc = {
        "synapse_model": "stdp_triplet_synapse_exc_to_exc",
        "weight": weight_EE,
        "delay": delay_int,
        "receptor_type": 1
    }
    conn_dict_exc_to_exc = {"rule": "all_to_all", "allow_autapses": False}
    nest.Connect(exc_neuron, exc_neuron, conn_dict_exc_to_exc, syn_dict_exc_to_exc)

    # Exc->Inh (stdp)
    nest.CopyModel("stdp_triplet_synapse", "stdp_triplet_synapse_exc_to_inh",
                   {"Wmax": 5.0})
    syn_dict_exc_to_inh = {
        "synapse_model": "stdp_triplet_synapse_exc_to_inh",
        "weight": weight_EI,
        "delay": delay_int,
        "receptor_type": 1
    }
    conn_dict_exc_to_inh = {"rule": "fixed_indegree", "indegree": 8}
    nest.Connect(exc_neuron, inh_neuron, conn_dict_exc_to_inh, syn_dict_exc_to_inh)

    # Inh->Exc (stdp)
    nest.CopyModel("stdp_triplet_synapse", "stdp_triplet_synapse_inh_to_exc",
                   {"Wmax": 5.0})
    syn_dict_inh_to_exc = {
        "synapse_model": "stdp_triplet_synapse_inh_to_exc",
        "weight": weight_IE,
        "delay": delay_int,
        "receptor_type": 2
    }
    conn_dict_inh_to_exc = {"rule": "fixed_outdegree", "outdegree": 6}
    nest.Connect(inh_neuron, exc_neuron, conn_dict_inh_to_exc, syn_dict_inh_to_exc)

    # Inh->Inh (static)
    nest.Connect(inh_neuron, inh_neuron, syn_spec={
        "synapse_model": "static_synapse",
        "weight": weight_II,
        "delay": delay_int,
        "receptor_type": 2
    })

    nest.Connect(voltmeter, exc_neuron)
    nest.Connect(voltmeter, inh_neuron)
    nest.Connect(exc_neuron + inh_neuron, spike_recorder)

    # background
    cg = nest.Create("ac_generator", params={"amplitude": 100.0, "frequency": 8.0})
    noise = nest.Create("noise_generator", params={"mean": 0.0, "std": 200.0, "frequency": 8.0})

    nest.Connect(noise, exc_neuron, syn_spec={"delay": resolution})
    nest.Connect(noise, inh_neuron, syn_spec={"delay": resolution})
    nest.Connect(cg, exc_neuron, syn_spec={"delay": resolution})
    nest.Connect(cg, inh_neuron, syn_spec={"delay": resolution})

    # Preheat (to prevent fluctuations in the initial state of the following stages)
    nest.Simulate(500)

    # (A) Training phase, randomly changing the center every 100 ms
    print(f">>> start [training] (10000 ms), rate={pg_rate}")
    sim_interval = 100.0
    num_steps_train = int(training_time / sim_interval)

    train_centers = []

    for i in range(num_steps_train):
        rates = np.zeros(500)
        # Use 25 + 50k (k in [0..9]) as the Gaussian center
        pg_mu = 25 + random.randint(0, 9) * 50
        train_centers.append(pg_mu)
        for j in range(500):
            rates[j] = pg_rate * np.exp(-((j - pg_mu)**2)/(2*(10.0**2)))
            pg[j].rate = rates[j]
        nest.Simulate(sim_interval)

    w_enc = get_exc_exc_weight_matrix(exc_neuron)

    if test_early_attractor:
        # no consolidation
        print(f">>> start [test: post-training] (100000 ms)")
        for k in range(500):
            pg[k].rate = 0.0

        test_start_time = nest.biological_time
        nest.Simulate(1e5)
        test_end_time = nest.biological_time

        events = nest.GetStatus(spike_recorder, "events")[0]
        senders = events["senders"]
        times = events["times"]

        mask = (times >= test_start_time) & (times < test_end_time) & (senders <= 10)
        spikes_test = (senders[mask], times[mask] - test_start_time)

        if return_functional_rf:
            rf_data = _run_functional_receptive_field_probe(
                pg,
                spike_recorder,
                exc_neuron,
                pg_rate,
                stimulus_centers=rf_stimulus_centers,
                n_repeats=rf_repeats,
                stimulus_time=rf_stimulus_time,
                quiet_time=rf_quiet_time,
            )
            return w_enc, None, spikes_test, rf_data

        return w_enc, None, spikes_test

    # (B) Consolidation phase
    else:
        print(f">>> start [consolidation] (300000 ms), rate={pg_rate}")
        sim_interval = 100.0
        num_steps_cons = int(consolidation_time / sim_interval)

        for i in range(num_steps_cons):
            rates = np.zeros(500)
            cycle_time = (i * sim_interval) % 30000.0
            if cycle_time >= 20000.0:
                pg_mu = train_centers[i % len(train_centers)]
                for j in range(500):
                    rates[j] = pg_rate * np.exp(-((j - pg_mu) ** 2) / (2 * (10.0 ** 2)))
            else:
                rates[:] = 0.0

            for k in range(500):
                pg[k].rate = rates[k]
            nest.Simulate(sim_interval)

        w_cons = get_exc_exc_weight_matrix(exc_neuron)

        print(f">>> start [test: post-consolidation] (100000 ms)")
        for k in range(500):
            pg[k].rate = 0.0

        test_start_time = nest.biological_time
        nest.Simulate(1e5)
        test_end_time = nest.biological_time

        events = nest.GetStatus(spike_recorder, "events")[0]
        senders = events["senders"]
        times = events["times"]

        mask = (times >= test_start_time) & (times < test_end_time) & (senders <= 10)
        spikes_test = (senders[mask], times[mask] - test_start_time)

        if return_functional_rf:
            rf_data = _run_functional_receptive_field_probe(
                pg,
                spike_recorder,
                exc_neuron,
                pg_rate,
                stimulus_centers=rf_stimulus_centers,
                n_repeats=rf_repeats,
                stimulus_time=rf_stimulus_time,
                quiet_time=rf_quiet_time,
            )
            return w_enc, w_cons, spikes_test, rf_data

        return w_enc, w_cons, spikes_test


def safe_run_single_simulation(rate_val, test_early_attractor):
    try:
        return run_single_simulation(rate_val, test_early_attractor=test_early_attractor)
    except Exception as e:
        print("\n" + "="*50)
        print("The subprocess caught the underlying real error！")
        traceback.print_exc()
        print("="*50 + "\n")
        raise RuntimeError(f"The subprocess crashed, the real reason was: {str(e)}")


def safe_run_single_simulation_with_functional_rf(
    rate_val,
    test_early_attractor=False,
    stimulus_centers=None,
    n_repeats=3,
    stimulus_time=500.0,
    quiet_time=200.0,
):
    try:
        return run_single_simulation(
            rate_val,
            test_early_attractor=test_early_attractor,
            return_functional_rf=True,
            rf_stimulus_centers=stimulus_centers,
            rf_repeats=n_repeats,
            rf_stimulus_time=stimulus_time,
            rf_quiet_time=quiet_time,
        )
    except Exception as e:
        print("\n" + "="*50)
        print("The subprocess caught the underlying real error")
        traceback.print_exc()
        print("="*50 + "\n")
        raise RuntimeError(f"The subprocess crashed, the real reason was: {str(e)}")
