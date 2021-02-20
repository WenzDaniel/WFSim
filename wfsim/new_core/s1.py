import numba

import numpy as np
from .common import _rand_choice_nb
from .timing_properties import *

from wfsim.load_resource import load_config


class S1:

    def __init__(self, config):
        """
        Class which generates photon timing and hit channel for a set of
        interactions and settings. Call class-instance to execute.

        Includes the following steps:
            TODO Finish me

        How to use:
            s1 = wfsim.core.S1(config)
            photons = s1(interactions)

        photons is a structure array containing time and channel
        information for the wfsim.core.Pulse class.

        :param config: fax config which can be loaded via
            ntauxfiles.get_sim_file('fax_config_nt.json', fmt='json')
        """
        self.config = config
        if self.config['detector'] != 'XENONnT':
            # TODO address this issue. It is mainly due to the maps and
            #  configs.
            raise ValueError('Currently only nT is supported!')

        self.config.update(getattr(self.config, self.__class__.__name__, {}))  # TODO: This may be needed to be changed
        # not sure yet why this line is needed
        self.resource = load_config(config)

        self.n_channels = len(config['gains'])
        self.turned_off_pmts = np.arange(self.n_channels)[np.array(config['gains']) == 0]

        # Use numba friendly typed dicts:
        keys = ['s1_ER_alpha_singlet_fraction',
                's1_NR_singlet_fraction',
                's1_ER_recombination_time',
                's1_ER_primary_singlet_fraction',
                's1_ER_secondary_singlet_fraction',
                'singlet_lifetime_liquid',
                'triplet_lifetime_liquid',
                'singlet_lifetime_gas',
                'triplet_lifetime_gas',
                'maximum_recombination_time',
                's1_decay_time']
        self.timing_dict = np.array(1, dtype=[(k, np.float64)for k in keys])

        for key in keys:
            self.timing_dict[key] = config[key]

    def __call__(self, instructions):
        """
        Function which computes S1 properties for a given set of
        interactions. Please note, that the number of photons in the
        instruction is updated according to the LY.

        :param instructions:
        :return: numpy structured array of photon arrival times and
            channels.
        """
        # Cannot use [['x', 'y', 'z']] since
        # interpolated maps do not like this.
        pos = np.array([instructions['x'],
                        instructions['y'],
                        instructions['z']]).T

        # Getting the maps:
        # TODO: Put some information about the maps, ask Lutz
        # TODO: Resolve difference in the 1T nT maps during read in
        #  adds to code cleanliness. Current version only supports nT.
        # Light yield map data driven includes already dead PMTs?
        ly_map = self.resource.s1_light_yield_map(pos).T[0]

        # Pattern map is an absolute LCE map based on simulations?
        pattern_map = self.resource.s1_pattern_map(pos)
        pattern_map[:, self.turned_off_pmts] = 0

        # Compute number of detected photons:
        # Note this is faster then called inside a numba function.
        instructions['photons'] = np.random.binomial(n=instructions['photons'], p=ly_map)

        photon_dtype = [(('Unix time of the digitized photon pulse', 'time'), np.float64),
                        (('PMT channel', 'channel'), np.int16)]
        res = np.zeros(np.sum(instructions['photons']), dtype=photon_dtype)

        get_s1_pulse_properties(instructions,
                                pattern_map,
                                np.arange(self.n_channels + 0.1, dtype=np.int16),
                                self.timing_dict,
                                self.config['s1_model_type'],
                                res)

        # As a last step at S1 smearing:
        # TODO: Only to simple model or also complex one?
        res['time'] += np.random.normal(0, self.config['s1_time_spread'], len(res['time']))

        return res


@numba.njit(cache=True)
def get_s1_pulse_properties(interactions, pattern_map, channels, config, model, res):
    """
    Function which distributes photons into PMT channels according to a light
    yield map and computes their arrival times.

    #TODO finalize doc string
    :param interactions: Structure array of interactions in the TPC.
    :param pattern_map:
    :param channels: Array of channels to be usd.
    :param config:
    :param model: Interaction model can be either "simple" to use an exponential
        decay law. "Complex" uses the more complex models which are defined in
        timing_properties.py.
    :param res: (numpy.array) Buffer array the results should be stored to.
    """
    if model not in ['simple', 'complex']:
        raise ValueError('The S1 scinitillation model must be either "simple" or "complex"!')

    offset = 0
    for ind, inter in enumerate(interactions):
        n_ph = inter['photons']
        if not n_ph:
            continue
        normalized_pattern = pattern_map[ind] / np.sum(pattern_map[ind])
        res['channel'][offset:offset + n_ph] = _rand_choice_nb(channels, normalized_pattern, n_ph)

        # Photon timing:
        t0 = inter['time']
        res['time'][offset:offset + n_ph] += t0

        # TODO: Implement field and phase per inetraction, currently fixed to 82 V/cm and liquid.
        #  Also requires a change in epix.
        res['time'][offset:offset + n_ph] += get_photon_timing(n_ph,
                                                               82,
                                                               inter['recoil'],
                                                               0,
                                                               model,
                                                               config
                                                               )
        offset += n_ph


@numba.njit(parallel=False, cache=True)
def get_photon_timing(n_ph, electrical_field, int_type, phase, recombination_model, config):
    """
    Computes the photon arrival time depending on LXe recombination and singlet
    and triplet states.

    :param n_ph: Number of photons.
    :param electrical_field: Electric field at the interaction side
    :param int_type: Interaction type must be a nest identifier (e.g. 0 for NR)
    :param phase: Phase of the xenon liquid=0 and gaseous=1.
    :param recombination_model: If ="simple" a simplified scintillation model is used.
        The model is given by an exponential distribution.
    :param config: Numba typed dict which contains the timing information of the
        config.
    """
    res = np.zeros(n_ph, dtype=np.int64)
    if recombination_model == 'simple':
        for i in numba.prange(n_ph):
            res[i] = np.random.exponential(config['s1_decay_time'])
    else:
        for i in numba.prange(n_ph):
            # TODO: Get the interaction number directly from nest?
            if int_type == 0:
                # Nuclear recoils:
                res[i] = nr(phase, config)
            elif int_type == 7 or int_type == 8 or int_type == 11:
                # Electronic recoils, number are nest identifiers
                # 7 = gamma
                # 8 = beta
                # 11 = Kr-83m which is both
                res[i] = er(phase, electrical_field, config)
            elif int_type == 6:
                # Alphas
                res[i] = alpha(phase, config)

    return res
