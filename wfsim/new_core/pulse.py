import logging

import numpy as np
from scipy.interpolate import interp1d
import numba
from numba.typed import List

from .common import sort_by_channel
import wfsim
from wfsim.load_resource import load_config

# TODO never worked with logging, read this up
log = logging.getLogger('SimulationCore')


class Pulse:
    def __init__(self, config):
        """
        TODO: Add doc string
        """
        self.config = config

        # TODO why is this needed?:
        self.config.update(getattr(self.config, self.__class__.__name__, {}))

        self.resource = load_config(config)

        # TODO also this function should get an return for easier testing
        # TODO add nveto compatibility?
        # TODO add high energy channels.
        self.init_pmt_current_templates()
        self.charge, self.pmt_spe_distribution = self.init_spe_scaling_factor_distributions()

    def __call__(self, times_and_channels,
                 add_baseline=True, add_noise=True, add_zle=True):
        """
        TODO: Add doc string
        """
        # Get Photon TTS values:
        TTS = np.random.normal(self.config['pmt_transit_time_mean'],
                               self.config['pmt_transit_time_spread'],
                               len(times_and_channels)
                               )
        times_and_channels['time'] = times_and_channels['time'] + TTS

        # Sort photon info:
        times_and_channels = sort_by_channel(times_and_channels)

        # Add gain and dpe field:
        # TODO additional fields needed? Should add truth information?
        photons = np.zeros(len(times_and_channels),
                           dtype=[(('Unix time of the digitized photon pulse', 'time'), np.float64),
                                  (('PMT channel', 'channel'), np.int16),
                                  (('Summed "gain" for PMT-signal [ADC x sample]', 'gain'), np.float32),
                                  (('Indicates if vuv-photon induced a DPE signal', 'dpe'), np.bool_),
                                  ])

        copy_photon_information(times_and_channels, photons)
        del times_and_channels

        # Check which vuv-photon triggers a DPE:
        # TODO: Can other sources of DPE emission be neglected?
        # TODO: nVETO does not have vuv-induced DPE emission!
        photons['dpe'] = np.random.binomial(1,
                                            p=self.config['p_double_pe_emision'],
                                            size=len(photons))

        # Add "gain" values to photons, for DPE we draw two times a gain value
        # and add them up.
        get_to_adc(photons, self.charge, self.pmt_spe_distribution)

        # Make pulses:
        # Two outputs since awkward arrays cannot be used,
        # First output stores information about each pulse
        # Second output stores pulses as numba.typed.List
        props, pulses = make_pulses(photons,
                                    self._pmt_current_templates,
                                    self.config.get('sample_duration'),
                                    self.resource.noise_data,
                                    config=self.config,
                                    add_baseline=add_baseline,
                                    add_noise=add_noise,
                                    add_zle=add_zle
                                    )
        return props, pulses

    def init_spe_scaling_factor_distributions(self, detector='tpc', truncate=True):
        """
        Function which reads in ADC x Sample SPE distribution.

        Note if a distraction extents into negative regions it is truncated.

        :param detector: Which detector should be used.
        :param truncate: If true SPE distributions are truncated in the
            negative regions.

        :returns: two numpy.arrays. First one containing the charge binning
            in units of ADC x sample and the second array contains the
            distributions for the individual PMTs.
        """
        # TODO: How to add the nveto?
        spe_shapes = self.resource.photon_area_distribution

        # Charge values in ADC x Sample:
        charge_binning = spe_shapes.loc[:, 'charge'].to_numpy()

        channel = self.config['channels_in_detector'][detector].astype('str')
        pmt_distributions = spe_shapes.loc[:, channel].to_numpy()

        negative_bins = np.argwhere(charge_binning < 0).flatten()
        if truncate and np.any(negative_bins):
            # If true let us truncate the distribution so we do not allow
            # negative charge values.

            # Get start and stop of negative bins and set all distributions here to
            # zero:
            start, stop = negative_bins.min(), negative_bins.max() + 1
            pmt_distributions[start:stop, :] = 0

            # Renormalize distirbutions:
            norm = pmt_distributions.sum(axis=0)

            # Set normalization to one if distributions sum up to zero.
            # Bit odd but prevents from 1/0-erros.
            norm = np.where(norm == 0, np.ones(len(norm)), norm)

            # Renomralize SPE distributions for the individual channels:
            pmt_distributions = pmt_distributions / norm

        return charge_binning, pmt_distributions.T

    def init_pmt_current_templates(self):
        """
        Create spe templates, for 10ns sample duration and 1ns rounding we have:
        _pmt_current_templates[i] : photon timing fall between [10*m+i, 10*m+i+1)
        (i, m are integers)
        """

        # Interpolate on cdf ensures that each spe pulse would sum up to 1 pe*sample duration^-1
        pe_pulse_function = interp1d(
            self.config.get('pe_pulse_ts'),
            np.cumsum(self.config.get('pe_pulse_ys')),
            bounds_error=False, fill_value=(0, 1))

        # Samples are always multiples of sample_duration
        sample_duration = self.config.get('sample_duration', 10)
        samples_before = self.config.get('samples_before_pulse_center', 2)
        samples_after = self.config.get('samples_after_pulse_center', 20)
        pmt_pulse_time_rounding = self.config.get('pmt_pulse_time_rounding', 1.0)

        # Let's fix this, so everything can be turned into int
        assert pmt_pulse_time_rounding == 1

        samples = np.linspace(-samples_before * sample_duration,
                              + samples_after * sample_duration,
                              1 + samples_before + samples_after)
        self._template_length = np.int(len(samples) - 1)

        templates = []
        for r in np.arange(0, sample_duration, pmt_pulse_time_rounding):
            pmt_current = np.diff(pe_pulse_function(samples - r)) / sample_duration  # pe / 10 ns
            # Normalize here to counter tiny rounding error from interpolation
            pmt_current *= (1 / sample_duration) / np.sum(pmt_current)  # pe / 10 ns
            templates.append(pmt_current)
        self._pmt_current_templates = np.array(templates)

        log.debug('Create spe waveform templates with %s ns resolution' % pmt_pulse_time_rounding)


@numba.njit(nogil=True, cache=True)
def copy_photon_information(times_and_channels, photons):
    """
    Copies photon information from one to another array.
    """
    for i in range(len(times_and_channels)):
        tc = times_and_channels[i]
        ph = photons[i]
        ph['time'] = tc['time']
        ph['channel'] = tc['channel']


@numba.njit(nogil=True, cache=True)
def get_to_adc(photons, charge, spe_adc_dis):
    """
    Function which adds gain and information about DPE signal to photons
    array. Requires photons to be sorted by channel. Writes result direct
    to input array.

    Note:
        We cannot use common._rand_choice_nb since it becomes very
        slow for a large number of photons.

    TODO: Finish doc string
    :param photons: numpy.structured array with the fields ....
    :param charge: Chrage binning of SPE-ADC x Sample distributions. The
        bins must be in units of [ADC x sample].
    :param spe_adc_dis:
    """

    prev_ch = photons[0]['channel']
    prob = spe_adc_dis[prev_ch]
    prob = np.cumsum(prob)
    for p in photons:
        ch = p['channel']

        if ch != prev_ch:
            prob = spe_adc_dis[prev_ch]
            prob = np.cumsum(prob)
            #TODO add linear scaling factor according to gain.

        index = np.searchsorted(prob, np.random.random(), side="right")
        gain = charge[index]

        if p['dpe']:
            # LXe vuv-photon induced DPE emission:
            index = np.searchsorted(prob, np.random.random(), side="right")
            gain += charge[index]

        p['gain'] = gain


def make_pulses(times_and_channels,
                spe_templates,
                dt,
                noise_data,
                config,
                add_noise=True,
                add_baseline=True,
                add_zle=True,
                ):
    #TODO: Add doc-string
    if add_zle and not add_baseline:
        raise ValueError('ZLE only works together with baseline! '
                         'Plase set "add_baseline" to True.')

    # TODO: Move dtpys into a new file?
    pulse_properties = np.zeros(len(times_and_channels),
                                dtype=[(('Unix time of the digitized photon pulse',
                                         'time'), np.int64),
                                       (('PMT channel',
                                         'channel'), np.int16),
                                       (('Number of photons in pulse',
                                         'n_photons'), np.int32),
                                       (('Number of vuv-induced DPE photons in pulse',
                                         'n_dpe_photons'), np.int32),
                                       (('Pulse duration in sample',
                                         'length'), np.int32)
                                       ]

                                )

    pulse_properties, p = _make_pulses(times_and_channels,
                                       spe_templates,
                                       dt,
                                       pulse_properties,
                                       noise_data,
                                       pre_trigger_window=config['pre_trigger'],
                                       post_trigger_window=config['post_trigger'],
                                       add_noise=add_noise,
                                       add_baseline=add_baseline,
                                       add_zle=add_zle
                                       )
    return pulse_properties, p


@numba.njit(nogil=True, cache=True)
def _make_pulses(times_and_channels,
                 spe_templates,
                 dt,
                 pulse_properties,
                 noise_data,
                 pre_trigger_window=50,
                 post_trigger_window=50,
                 add_noise=True,
                 add_baseline=True,
                 add_zle=True
                 ):
    """
    Function which creates PMT pulses based on the photon information
    supplied.

    TODO: Finish doc string
    :param times_and_channels: numpy structure array storing the "time"
        [ns] and "channel" informations of the photons.
    :param spe_templates: numpy.array of normalized SPE templates. The
        array must have the shape
    :param dt: Sampling rate/sample length of the digitzer in ns.
        E.g. 10 ns for nT TPC.
    :param pulse_properties:
    :param noise_data: Flat numpy.array containing some noise data.
    :param pre_trigger_window: Integer, digitizer pre-trigger window in
        sample.
    :param post_trigger_window: Same as pre-trigger
    :param add_noise: Boolean, if True noise is added to the pulse.
    :param add_baseline: Boolean, if True, add baseline invert pulse
        and truncates pulse according to the specified dynamic range.
    :param add_zle: Boolean, if True applies digitizer self-trigger to
        the pulse, chops pulses a bit.

    :returns: pulse_properties array, numba.typed.List containing the
        pulses.
    """
    pulses = List()

    # Get default length of the SPE template and
    # current channel:
    time_spe_signal = len(spe_templates[0]) * dt
    current_channel = times_and_channels[0]['channel']

    # Get pulse start for the first photon:
    pulse_start = times_and_channels[0]['time'] - pre_trigger_window*dt
    if pulse_start < 0:
        raise ValueError('Found a pulse with a negative start time. '
                         'This might be due to a too small time offset for '
                         'the first interaction!')

    pulse_end = times_and_channels[0]['time'] + time_spe_signal + post_trigger_window*dt

    n_pulses = 0
    photon_offset = 0
    n_photons_per_pulse = 1
    for phi in range(1, len(times_and_channels)):
        # Loop over photons, test if pulses overlap
        # If yes updated pulse end and number of photons in pulse
        # If not create pulse
        time = times_and_channels[phi]['time']
        photon_start = time - pre_trigger_window*dt

        ch = times_and_channels[phi]['channel']

        if ch == current_channel and (pulse_end + dt) >= photon_start:
            # Signal is in current interval, +dt needed due to 1 ns ->
            # 10 ns sampling conversion via floor-division.
            # >= to follow the same logic as holdoff in
            # wfsim.utils.find_intervals_below_threshold
            pulse_end = time + time_spe_signal + post_trigger_window*dt
            n_photons_per_pulse += 1

        if (ch != current_channel
                or (pulse_end + dt) < photon_start
                or phi == (len(times_and_channels) - 1)):
            # Either we run out of Photons, changed the channel, or Signal starts a new
            # pulse either way we have to create the pulse and store it:
            # ATTENTION: Times regarding the Pulse are now in SAMPLE
            # not ns! We convert it back when we write the result.
            pulse_start //= dt
            pulse_end //= dt

            len_pulse = int(pulse_end - pulse_start + 1)
            pulse = np.zeros(len_pulse, dtype=np.float64)

            # Getting photon information for current pulse:
            photon_properties = times_and_channels[photon_offset:photon_offset + n_photons_per_pulse]

            # Populating pulse buffer with data:
            add_current(photon_properties['time'],
                        photon_properties['gain'] * dt,
                        pulse_start,
                        dt=dt,
                        pmt_current_templates=spe_templates,
                        pulse_current=pulse
                        )

            if add_noise:
                # Adding noise:
                _add_noise(pulse, noise_data)

            if add_baseline:
                # Adding baseline, invert and & truncate at 0 and max bit:
                _digitize_signal(pulse,
                                 baseline=16000,  # TODO add baseline value
                                 saturation_value=2**14)

                # Add ZLE:
            # TODO: Can we simplify the ZLE and return part? Has grown
            # quite large.
            if add_zle:
                # Adding trigger behavior of the digitzers,
                # record only windows which are pre_trigger +
                # above_threshold + post_trigger long, if pulse has
                # gaps below threshold > pre- + post-trigger split
                # pulse.
                # TODO: Add threshold per channel
                intervals = _zle(pulse,
                                 16000 - 20 - 1,
                                 pre_trigger_window,
                                 post_trigger_window)
            else:
                intervals = np.array([[pre_trigger_window,
                                       len(pulse) - post_trigger_window]])

            if len(intervals) > 1:
                # Only needed if we have to split a pulse, if not needed
                # use cheaper part below:
                for le, re in intervals:
                    le -= pre_trigger_window
                    re += post_trigger_window
                    pulses.append(pulse[le:re])

                    # Now pulse properties:
                    # Compute start and end of the pulse in ns:
                    start = (pulse_start + le) * dt
                    end = (pulse_start + re) * dt

                    # Test which photons are inside this pulse:
                    mask_inside_pulse = photon_properties['time'] < end
                    mask_inside_pulse &= (start <= photon_properties['time'])

                    # Count how many photons are inside pulse:
                    pulse_properties[n_pulses]['n_photons'] = np.sum(mask_inside_pulse)
                    # Now add DPE infromation:
                    mask_inside_pulse &= photon_properties[mask_inside_pulse]['dpe']
                    pulse_properties[n_pulses]['n_dpe_photons'] = np.sum(mask_inside_pulse)

                    pulse_properties[n_pulses]['time'] = start
                    pulse_properties[n_pulses]['channel'] = current_channel
                    pulse_properties[n_pulses]['length'] = re - le

                    n_pulses += 1

            if len(intervals) == 1:
                # Simple case, of just a single interval. Store pulse and pulse properties:
                le, re = intervals[0]
                le -= pre_trigger_window
                re += post_trigger_window
                pulses.append(pulse[le:re])

                pulse_properties[n_pulses]['time'] = (pulse_start + le) * dt
                pulse_properties[n_pulses]['channel'] = current_channel
                pulse_properties[n_pulses]['n_photons'] = n_photons_per_pulse
                pulse_properties[n_pulses]['n_dpe_photons'] = np.sum(photon_properties['dpe'])
                pulse_properties[n_pulses]['length'] = re - le
                n_pulses += 1

            # Update values for next pulse:
            current_channel = ch
            pulse_start = photon_start
            pulse_end = time + time_spe_signal + post_trigger_window*dt
            photon_offset += n_photons_per_pulse
            n_photons_per_pulse = 1

    return pulse_properties[:n_pulses], pulses


# Taken as it is, but no other way then copying it since in class.
@numba.njit(nogil=True, cache=True)
def add_current(photon_timings,
                photon_gains,
                pulse_left,
                dt,
                pmt_current_templates,
                pulse_current):
    #         """
    #         Simulate single channel waveform given the photon timings
    #         photon_timing         - dim-1 integer array of photon timings in unit of ns
    #         photon_gain           - dim-1 float array of ph. 2 el. gain individual photons
    #         pulse_left            - left of the pulse in unit of 10 ns
    #         dt                    - mostly it is 10 ns
    #         pmt_current_templates - list of spe templates of different reminders
    #         pulse_current         - waveform
    #         """
    if not len(photon_timings):
        return

    template_length = len(pmt_current_templates[0])
    i_photons = np.argsort(photon_timings)
    # Convert photon_timings to int outside this function
    # photon_timings = photon_timings // 1

    gain_total = 0
    tmp_photon_timing = photon_timings[i_photons[0]]
    for i in i_photons:
        if photon_timings[i] > tmp_photon_timing:
            start = int(tmp_photon_timing // dt) - pulse_left
            reminder = int(tmp_photon_timing % dt)
            pulse_current[start:start + template_length] += \
                pmt_current_templates[reminder] * gain_total

            gain_total = photon_gains[i]
            tmp_photon_timing = photon_timings[i]
        else:
            gain_total += photon_gains[i]
    else:
        start = int(tmp_photon_timing // dt) - pulse_left
        reminder = int(tmp_photon_timing % dt)
        pulse_current[start:start + template_length] += \
            pmt_current_templates[reminder] * gain_total


@numba.njit(nogil=True, cache=True)
def _add_noise(data, noise_data):
    """
    Helper function to add noise data. Function, in case we would like
    to make this more complex.
    """
    #TODO: add noise per channel?
    length_noise = len(noise_data)
    id_t = np.random.randint(0, length_noise - len(data))
    data[:] += noise_data[id_t:length_noise + id_t]


@numba.njit(nogil=True, cache=True, parallel=False)
def _digitize_signal(data, baseline, saturation_value=2**14):
    """
    Function which adds baseline, truncates float data to integers and
    and applies saturation.
    """
    for i in numba.prange(len(data)):
        d = data[i]
        d = baseline - d
        d = d // 1
        # Digitizer can saturate in two directions:
        d = min(saturation_value, d)
        data[i] = max(0, d)


@numba.njit(nogil=True, cache=True)
def _zle(pulse, threshold, pre_trigger, post_trigger):
    """
    Function which estimates the left and right indices for which the
    pulses crosses the digitizer threshold.

    :param pulse: numpy.array containing the complete pulse.
    :param threshold: Threshold to be applied. Please note that the
        threshold must be an absolute value with respect to the inverted
        and baseline subtracted pulse. E.g. baseline - 20 ADC - 1.
    :param pre_trigger: Number of SAMPLES for pre-trigger.
    :param post_trigger: Number of SAMPLES for post-trigger.

    :returns: numpy.array of the shape n x 2 containing the start and
        end of the pulse intervals.
    """
    intervals = np.zeros((len(pulse)//2, 2), dtype=np.int64)

    # Test pulse for values above threshold:
    n_intervals = wfsim.utils.find_intervals_below_threshold(pulse,
                                                             threshold,
                                                             pre_trigger + post_trigger,
                                                             intervals
                                                             )
    return intervals[:n_intervals]