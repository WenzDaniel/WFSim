import numba
import numpy as np


@numba.njit
def _singlet_triplet_delays(phase, singlet_ratio, config):
    """
    Given the amount of the eximer, return time between excimer decay
    and their time of generation.

    :param phase: Integer indicating if event is in liquid (0)
        or gas (1).
    :param singlet_ratio: Float, fraction of excimers that become
        singlets (NOT the ratio of singlets/triplets!).
    :param config: Numbda type dict which contains the timing information of the
        config.

    :returns: Time delay for interaction.
    """
    if phase:
        delay = (config['singlet_lifetime_gas'], config['triplet_lifetime_gas'])
    else:
        delay = (config['singlet_lifetime_liquid'], config['triplet_lifetime_liquid'])

    ind = np.random.binomial(1, p=(1 - singlet_ratio))  # True = Triplet
    delay = delay[ind]
    return np.random.exponential(1) * delay


@numba.njit
def alpha(phase, config):
    return _singlet_triplet_delays(phase, config['s1_ER_alpha_singlet_fraction'], config)


@numba.njit
def nr(phase, config):
    return _singlet_triplet_delays(phase, config['s1_NR_singlet_fraction'], config)


# TODO: Add LED

@numba.njit
def er(phase, efield, config):
    """

    :param efield: Electric field in V/cm.
    """
    config['s1_ER_recombination_time'] = 3.5 / 0.18 * (1 / 20 + 0.41) * np.exp(-0.009 * efield)

    # compute timings
    recomb = np.random.binomial(1, 1 - config['s1_ER_primary_singlet_fraction'])  # 0 or 1
    timing = [0, config['s1_ER_recombination_time']][recomb]

    timing *= 1 / (1 - np.random.uniform(0, 1)) - 1
    timing = min(timing, config['maximum_recombination_time'])
    timing = max(timing, 0)

    if recomb:
        timing += _singlet_triplet_delays(phase, config['s1_ER_secondary_singlet_fraction'], config)
    else:
        timing += _singlet_triplet_delays(phase, config['s1_ER_primary_singlet_fraction'], config)
    return timing