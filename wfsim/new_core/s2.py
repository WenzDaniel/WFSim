import numba
import numpy as np

from wfsim.load_resource import load_config
from wfsim.new_core.timing_properties import _singlet_triplet_delays
from wfsim.new_core.common import _rand_choice_nb


class S2:

    def __init__(self, config):
        #TODO update init
        self.config = config

        # TODO add all missing features:
        if self.config['detector'] != 'XENONnT':
            # TODO address this issue. It is mainly due to the maps and
            #  configs.
            raise ValueError('Currently only nT is supported!')
        if self.config['field_distortion_on']:
            raise ValueError("Field distortions are currently not supported!")
        if self.config['s2_luminescence_model'] == 'simple':
            raise ValueError("Simple luminescence model not implemented.")
        if self.config['s2_mean_area_fraction_top'] > 0:
            raise ValueError('Renormalization of pattern maps according to mean' 
                             ' area fraction to not yet supported.')

        self.config.update(getattr(self.config, self.__class__.__name__, {}))  # TODO: This may be needed to be changed
        # not sure yet why this line is needed
        self.resource = load_config(config)

        self.n_channels = len(config['gains'])
        self.turned_off_pmts = np.arange(self.n_channels)[np.array(config['gains']) == 0]

        self.init_rotation_matrix_anode_wires()

        # Init time information from settings to structure array, need
        # since numba:
        # TODO: This is a bit tedious can we do better? numba.typed.dicts
        #  are too slow...
        # Liquid information is required by timing_properties._singlet_triplet_delays,
        # but not used for S2.
        timing_dict = np.zeros(1, dtype=[('singlet_lifetime_liquid', np.float32),
                                         ('triplet_lifetime_liquid', np.float32),
                                         ('singlet_lifetime_gas', np.float32),
                                         ('triplet_lifetime_gas', np.float32),
                                         ('singlet_fraction_gas', np.float32),
                                         ])
        for field in timing_dict.dtype.names:
            timing_dict[field] = self.config[field]
        self.timing_dict = timing_dict[0]

    def init_rotation_matrix_anode_wires(self):
        """
        Function which computes the rotation matrix for the anode wires with
        respect to the simulation XENONnT coordinate system.

        The rotation matrix is required to compute the distance between an
        interaction and the very nest wire. This information is used to compute
        the luminescence timing of the garfield model.
        """
        #TODO: add infromation about this map?
        #TODO: Ask for more information about the garfield timing.
        self.x_grid, self.n_grid = np.unique(self.resource.s2_luminescence['x'], return_counts=True)
        self.i_grid = (self.n_grid.sum() - np.cumsum(self.n_grid[::-1]))[::-1]

        tilt = getattr(self.config, 'anode_xaxis_angle', np.pi / 4)
        rotation_mat = np.array(((np.cos(tilt), -np.sin(tilt)), (np.sin(tilt), np.cos(tilt))))
        self.rotation_matrix_anode = rotation_mat

    def __call__(self, interactions):
        """
        TODO: Add doc strings:

        :param interactions:
        :return:
        """
        photons_per_electron = self.propagate_electrons_and_create_photons(interactions)
        photons = self.get_photon_time_and_channels(interactions, photons_per_electron)
        return photons

    def propagate_electrons_and_create_photons(self, interactions):
        """
        Function which propagates electron cloud through LXe. Draws
        number of photons created per electron and add information to
        interactions.
        Computes distance between interaction and closest
        Anode wire. This is required for the garfield S2-luminescence
        timing model.
        Updates number of electrons in interactions based on electron
        lifetime.

        :param interactions:
        :return:
        """
        if self.config['field_distortion_on']:
            # TODO add field distortions
            raise ValueError('Field distortion not implemnted yet.')
            # self.inverse_field_distortion(x, y, z)
        else:
            xy_position = np.array([interactions['x'], interactions['y']]).T

        # Read in secondary electron gain from electroluminescens:
        # TODO get information about this map
        # TODO fix 1T nT from during read in.
        sc_gains = self.resource.s2_light_yield_map(xy_position).flatten()
        sc_gains *= self.config['s2_secondary_sc_gain']

        # Compute average drift time of electrons, absorb electrons,
        # computes distance to closest anode wire.
        charge_yields = propagate_electrons(interactions,
                                            self.config['drift_velocity_liquid'],
                                            self.config['drift_time_gate'],
                                            self.config['electron_lifetime_liquid'],
                                            self.config['electron_extraction_yield'],
                                            self.rotation_matrix_anode,
                                            self.config['anode_pitch']
                                            )

        interactions['electrons'] = np.random.binomial(interactions['electrons'],
                                                       p=charge_yields
                                                       )
        del charge_yields
        # Compute number of photons for each interaction:
        photons_per_electron = create_s2_scintillation_photons(interactions, sc_gains)
        return photons_per_electron

    def get_photon_time_and_channels(self,
                                     interactions,
                                     photons_per_electron):
        """

        :param interactions:
        :param photons_per_electron:
        :return:
        """
        # Get total number of photons and electrons:
        nph = np.sum(interactions['s2_photons'])

        # TODO: Unify dtypes for S1 and S2-class!
        photon_dtype = [(('Unix time of the digitized photon pulse', 'time'), np.float64),
                        (('PMT channel', 'channel'), np.int16),
                        (('GEANT4 id this pulse belongs to', 'g4id'), np.int32),
                        (('Event number which identifies the interactions this event belongs to.', 'event_number'),
                         np.int32),
                        ]
        photons = np.zeros(nph, dtype=photon_dtype)

        # Compute photon time information:
        photon_timings(interactions,
                       photons,
                       photons_per_electron,
                       self.config['electron_trapping_time'],
                       self.config['diffusion_constant_liquid'],
                       self.config['drift_velocity_liquid'],
                       self.x_grid,
                       self.n_grid,
                       self.i_grid,
                       self.resource.s2_luminescence['t'],
                       self.config['s2_time_spread'],
                       self.timing_dict
                       )

        # Distributing photons over channels:
        # Load PMT probability map for a given interaction position:
        points = np.array([interactions['x'], interactions['y']]).T
        pattern = self.resource.s2_pattern_map(points)
        # Normalize map:
        norm = np.sum(pattern, axis=1)
        pattern = (pattern.T / norm).T

        # Assign channel values to the photons:
        # TODO remove hardcoded channel values.
        get_channels(interactions, photons, pattern, np.arange(494, dtype=np.int16))

        return photons


@numba.njit()
def distance_to_wire(x, y, rotation_matrix, pitch):
    """
    Rotating x and y to coordinate system where y-axis is aligned to
    anode wires. Compute distance to nearest wire.

    :param x:
    :param y:
    :param rotation_matrix:
    :param pitch:
    """
    rot_y = rotation_matrix[:, 1]
    distance = x * rot_y[0] + y * rot_y[1]
    # TODO: Ask why is this the shortest distance to any anode wire.
    # Have to ask and put comment.
    distance = (distance + pitch / 2) % pitch - pitch / 2

    return distance


@numba.njit(parallel=False)
def propagate_electrons(interactions,
                        drift_velocity,
                        drift_time_gate,
                        lifetime_liquid,
                        extraction_yield,
                        anode_rotation_matrix,
                        anode_wire_pitch=0.5
                        ):
    """
    Function which propagates electron clouds to the LXe surface.
    Computes the following parameters:

    #TODO: Finish doc-string.

    Updates instructions in place.

    :returns: numpy.array with charge_yields.
    """
    charge_yields = np.zeros(len(interactions))

    for i in numba.prange(len(interactions)):
        inter = interactions[i]

        # Compute center drift time of electron cloud (without diffusion):
        # TODO how to deal with events above gate?
        drift = -inter['z'] / drift_velocity + drift_time_gate
        drift = max(drift, 0)
        inter['drift'] = drift

        # Compute life time correction and charge yield:
        drift_correction = np.exp(-1 * drift / lifetime_liquid)
        charge_yields[i] = extraction_yield * drift_correction

        # Compute shortest distance to closest anode wire:
        inter['distance'] = distance_to_wire(inter['x'],
                                             inter['y'],
                                             anode_rotation_matrix,
                                             anode_wire_pitch
                                             )
    return charge_yields


@numba.njit
def create_s2_scintillation_photons(interactions, sc_gains):
    """
    Function which draws for each electron the number of generated
    photons adds information about total number of generated s2_photons
    to interactions.

    :returns: numpy array of photons generated per electron. Needed to
        compute timing and channel information per photon later.
    """
    n_photon_per_electron = np.zeros(np.sum(interactions['electrons']),
                                     dtype=np.int32)

    offset = 0
    for ind, inter in enumerate(interactions):
        n_elec = inter['electrons']
        photons = np.random.poisson(sc_gains[ind], n_elec)

        inter['s2_photons'] = np.sum(photons)
        n_photon_per_electron[offset:offset + n_elec] = photons
        offset += n_elec

    return n_photon_per_electron


@numba.njit
def photon_timings(interactions,
                   photons,
                   photons_per_electron,
                   electron_trapping_time,
                   diffusion_constant_liquid,
                   drift_velocity_liquid,
                   x_grid,
                   n_grid,
                   i_grid,
                   s2_luminescence_timing,
                   s2_time_spread,
                   timing_dict
                   ):
    """
    TODO Add doc-sting
    """
    offset = 0
    ph_i = 0
    elec_i = 0
    for inter in interactions:
        # Looping over interacrtions and extra informations from the
        # bulke movement of the electrons:
        # Adding time and drift time information:
        t_interaction = inter['time']
        drift_time = inter['drift']
        t_interaction += drift_time

        # Computing diffusion smearing for given interaction:
        diffusion_stdev = np.sqrt(2 * diffusion_constant_liquid * drift_time)
        diffusion_stdev /= drift_velocity_liquid

        # Compute pitch index for garfield luminescence timing:
        pitch_index = np.argmin(np.abs(inter['distance'] - x_grid))

        for i in range(inter['electrons']):
            # Looping over electrons and compute time properties on the
            # per-electron level.

            # Add diffusion per electron:
            t_electron = np.random.normal(0, diffusion_stdev)

            # Add trapping time of electron:
            t_electron += np.random.exponential(electron_trapping_time)

            # Get number of photons generated by this electron and
            # get time information on the per-photon level:
            nph = photons_per_electron[elec_i]
            t_photons = _propagate_photons(pitch_index,
                                           nph,
                                           n_grid,
                                           i_grid,
                                           s2_luminescence_timing,
                                           s2_time_spread,
                                           timing_dict
                                           )

            # Assign electron-bulke propagation times:
            photons['time'][offset:offset + nph] = t_interaction

            # Adding electron propagation-times:
            photons['time'][offset:offset + nph] += t_electron

            # Adding per-photon propagation times:
            photons['time'][offset:offset + nph] += t_photons

            # Updating electron and photon offset:
            elec_i += 1
            offset += nph


@numba.njit(parallel=False)
def _propagate_photons(pitch_index,
                       nph,
                       n_grid,
                       i_grid,
                       s2_luminescence_timing,
                       s2_time_spread,
                       deexcitation_timing):
    """
    TODO provide doc string and documentation
    """
    res = np.zeros(nph, dtype=np.float64)

    for i in numba.prange(nph):
        # Loop over photons and draw properties for each photon.
        # Drawing luminescence time from Garfield model:
        index = i_grid[pitch_index] + np.random.randint(n_grid[pitch_index])
        t = s2_luminescence_timing[index]

        # Add S2 time spread:
        t += np.random.normal(0, s2_time_spread)

        # Draw if singlet or triplet, draw de-excitation time from
        # singlet and triplet states:
        t += _singlet_triplet_delays(1,  # 1 == GXe
                                     deexcitation_timing['singlet_fraction_gas'],
                                     deexcitation_timing)
        res[i] = t

    return res


@numba.njit
def get_channels(interactions, photons, pattern, pmt_channel):
    """
    Function which assigns channels to photons as well as event-
    and g4id.

    """
    offset = 0
    for ind, inter in enumerate(interactions):
        n_ph = inter['s2_photons']
        channel = _rand_choice_nb(pmt_channel, pattern[ind], n_ph)
        ph = photons[offset:offset + n_ph]
        ph['channel'][:] = channel

        # Add information about the event- and g4id:
        ph['event_number'][:] = inter['event_number']
        ph['g4id'][:] = inter['g4id']
        offset += n_ph
