import numba
import numpy as np


@numba.njit(nogil=True, cache=True)
def _rand_choice_nb(arr, prob):
    """
    Function which mimis the behavior of np.random.choice including the "p" option.

    :param arr: A 1D numpy array of values to sample from.
    :param prob: A 1D numpy array of probabilities for the given samples.
        The sum of the probabilities must be properly normalized!
    :return: A random sample from the given array with a given probability.
    """
    # Function which mimics np.random.choic stolen from https://github.com/numba/numba/issues/2539
    return arr[np.searchsorted(np.cumsum(prob), np.random.random(), side="right")]


@numba.njit(nogil=True, cache=True)
def sort_by_channel(x):
    """
    Sorts array first by channel than by time. Idea taken from
    strax.sort_by_time. Is a factor 5 faster as np.sort.

    Assumes, that the time range does not span more than 11 days!.
    """
    if len(x) == 0:
        # Nothing to do, and .min() on empty array doesn't work, so:
        return x

    if x['time'].max() - x['time'].min() > 10**15:
        raise ValueError('Time cannot span more than 10**15 ns!')

    # I couldn't get fast argsort on multiple keys to work in numba
    # So, let's make a single key...
    sort_key = (x['channel'] - x['channel'].min()) * 10**15 + (x['time'] - x['time'].min())
    sort_i = np.argsort(sort_key)
    return x[sort_i]