import random
import matplotlib.pyplot as plt
import nest
import numpy as np


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
    Get the 10x10 weight matrix between excitatory neurons (exc->exc).
    This is identical to the original script, except that exc_neuron is passed as a parameter to facilitate multiple simulations.
    """
    exc_conns = nest.GetConnections(exc_neuron, exc_neuron)
    exc_senders = np.array(exc_conns.source)
    exc_targets = np.array(exc_conns.target)
    exc_weights = np.array(exc_conns.weight)

    # Sort by sender
    idx_array = np.argsort(exc_senders)
    sorted_senders = exc_senders[idx_array]
    sorted_targets = exc_targets[idx_array]
    sorted_weights = exc_weights[idx_array]

    # Reshape (each sender is connected to 9 other neurons, so shape=(10,9))
    targets_reshaped = np.reshape(sorted_targets, (10, 9))
    weights_reshaped = np.reshape(sorted_weights, (10, 9))

    # Sort each row by target again to ensure the columns are neatly arranged
    for i, (trgs, ws) in enumerate(zip(targets_reshaped, weights_reshaped)):
        idx_t = np.argsort(trgs)
        weights_reshaped[i] = ws[idx_t]
    weight_matrix = np.zeros((10, 10))
    tu9 = np.triu_indices_from(weights_reshaped)
    tl9 = np.tril_indices_from(weights_reshaped, -1)
    tu10 = np.triu_indices_from(weight_matrix, 1)
    tl10 = np.tril_indices_from(weight_matrix, -1)

    weight_matrix[tu10] = weights_reshaped[tu9]
    weight_matrix[tl10] = weights_reshaped[tl9]
    return weight_matrix


def run_single_simulation(pg_rate):
    """
    Run a single network simulation, setting the sinusoidal_poisson_generator 's `rate` to `pg_rate`.
    Return the excitatory weight matrices for the three stages (w_enc, w_cons, w_recall).
    """

    nest.ResetKernel()
    nest.SetKernelStatus({"resolution": resolution})

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
        "phase": 0,
        "individual_spike_trains": False
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

    # (B) Consolidation phase
    print(f">>> start [consolidation] (200000 ms), rate={pg_rate}")
    sim_interval = 100.0
    num_steps_cons = int(consolidation_time / sim_interval)

    # Every 30,000 ms is a cycle, with no stimulation for the first 20,000 ms and replay of training stimulation for the last 10,000 ms.
    for i in range(num_steps_cons):
        rates = np.zeros(500)
        cycle_time = (i * sim_interval) % 30000.0
        if cycle_time >= 20000.0:
            pg_mu = train_centers[i % len(train_centers)]
            for j in range(500):
                rates[j] = pg_rate * np.exp(-((j - pg_mu)**2)/(2*(10.0**2)))
        else:
            rates[:] = 0.0

        for k in range(500):
            pg[k].rate = rates[k]
        nest.Simulate(sim_interval)

    w_cons = get_exc_exc_weight_matrix(exc_neuron)

    # （C）recall：10,000 ms
   # print(f">>> start [recall] (10000 ms), rate={pg_rate}")
    #sim_interval = 100.0
    #num_steps_retrieval = int(retrieval_time / sim_interval)

    #unique_centers = list(set(train_centers))
    #n_remove = 5
    #removed_centers = random.sample(unique_centers, n_remove)
    #kept_centers = [c for c in unique_centers if c not in removed_centers]

    #for i in range(num_steps_retrieval):
        #rates = np.zeros(500)
        #pg_mu = kept_centers[i % len(kept_centers)]
        #rates[pg_mu] = pg_rate
        #for k in range(500):
            #pg[k].rate = rates[k]
        #nest.Simulate(sim_interval)
    #nest.Simulate(10000)
    #w_recall = get_exc_exc_weight_matrix(exc_neuron)

    return w_enc, w_cons