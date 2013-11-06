# The Hazard Library
# Copyright (C) 2012 GEM Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
Module :mod:`openquake.hazardlib.tom` contains implementations of probability
density functions for earthquake temporal occurrence modeling.
"""
import abc
import math

import numpy
import scipy.stats

from openquake.hazardlib.slots import with_slots

@with_slots
class BaseTOM(object):
    """
    Base class for a temporal occurrence model, that is a probability density
    function allowing calculation of probability of earthquake rupture
    occurrence in a time span.

    :param time_span:
        The time interval for which probabilities are computed, in years.
    :raises ValueError:
        If ``time_span`` is not positive.
    """
    __metaclass__ = abc.ABCMeta

    __slots__ = ['time_span']

    def __init__(self, time_span):
        if time_span <= 0:
            raise ValueError('time_span must be positive')
        self.time_span = time_span

    @abc.abstractmethod
    def get_probability_no_exceedance(self, occurrence_rate, poes):
        """
        Compute and return, for a number of ground motion levels and sites,
        the probability that a rupture with annual occurrence rate given by
        ``occurrence_rate`` and able to cause ground motion values higher than
        a given level at a site with probability ``poes``, does not cause any
        exceedance in the time window specified by the ``time_span`` parameter
        given in the constructor.

        Method must be implemented by subclasses.

        :param occurrence_rate:
            The average number of events per year.
        :param poes:
            2D numpy array containing conditional probabilities the the a
            rupture occurrence causes a ground shaking value exceeding a
            ground motion level at a site. First dimension represent sites,
            second dimension intensity measure levels. ``poes`` can be obtained
            calling the :meth:`method
            <openquake.hazardlib.gsim.base.GroundShakingIntensityModel.get_poes>`.
        :return:
            2D numpy array containing probabilities of no exceedance. First
            dimension represents sites, second dimensions intensity measure
            levels.
        """

@with_slots
class BrownianPassageTimeTOM(BaseTOM):
    """
    Temporal occurrence model based on Brownian Passage Time distribution, as
    described in:

    A Brownian Model for Recurrent Earthquakes, by Mark V. Matthewes, William L.
    Ellsworth, and Paul. A. Reasenberg, Bullettin of the Seismological Society
    of America, Vol. 92, No.6, pages 2233-2250, 2002.

    :param time_span:
        Float, the time interval probability are computed for, in years.
    :param reference_time:
        Float, the time from which ``time_span`` begins, in years
    :param time_last_event:
        Float, time of last event occurred in the source, in years
    :param alpha:
        Float, aperiodicity factor, that is the coefficient of variation of the
        source recurrence time.
    :raises ValueError:
        If ``time_span`` is not positive, if the elapsed time (
        ``reference_time`` - ``time_last_event``) is not positive, if ``alpha``
        is not positive
    """
    __slots__ = BaseTOM.__slots__ + 'elapsed_time alpha'.split()

    def __init__(self, time_span, reference_time, time_last_event, alpha):
        elapsed_time = reference_time - time_last_event
        if elapsed_time <= 0:
            raise ValueError('''elapsed time (reference time minus time of last
            event) must be positive''')
        if alpha <= 0:
            raise ValueError('aperiodicity factor must be positive')
        super(BrownianPassageTimeTOM, self).__init__(time_span)
        self.elapsed_time = elapsed_time
        self.alpha = alpha

    def get_probability_no_exceedance(self, occurrence_rate, poes):
        """
        First compute probability of rupture to occurre once ``p1`` in the
        time span specified in the constructor. This is done implementing
        equation 17 page 2238 of Matthewes et al. 2002; that is the probability
        of the rupture to occur in the time interval between ``elapsed_time``
        and ``elapsed_time + time_span`` is computed as ::

        (F(elapsed_time + time_span) - F(elapsed_time)) / (1 - F(elapsed_time))

        where ``F`` is the cumulative distribution function for the Brownian
        Passage Time distribution.

        Given that ``time span`` is in general much smaller than event
        recurrence time, the probability of having more than one rupture in
        ``time_span`` is assumed 0.

        By knowing the probability of observing one occurrence ``p1``, and the
        probability of zero occurrences ``p0 = 1 - p1``, the probability of no
        exceedance is computed as ::

            p0 + p1 * (1 - poes)

        which follows from the total probability theorem.
        """
        mu = 1. / occurrence_rate

        num = self._cdf(self.elapsed_time + self.time_span, mu) - \
              self._cdf(self.elapsed_time, mu)
        den = 1 - self._cdf(self.elapsed_time, mu)

        p1 = num / den
        p0 = 1 - p1

        return p0 + p1 * (1 - poes)

    def _combine_variables(self, t, mu):
        """
        Implements equation 14 page 2237 of Matthewes et al. 2002.
        """
        u1 = (1. / self.alpha) * (math.sqrt(t / mu) - math.sqrt(mu / t))
        u2 = (1. / self.alpha) * (math.sqrt(t / mu) + math.sqrt(mu / t))

        return u1, u2

    def _cdf(self, t, mu):
        """
        Compute and return cumulative distribution function value.

        Implements equation 15 page 2237 of Matthewes et al. 2002.
        """
        cdf = scipy.stats.norm().cdf
        u1, u2 = self._combine_variables(t, mu)

        return cdf(u1) + math.exp(2 / self.alpha ** 2) * cdf(- u2)


@with_slots
class PoissonTOM(BaseTOM):
    """
    Poissonian temporal occurrence model.
    """

    def get_probability_one_or_more_occurrences(self, occurrence_rate):
        """
        Calculate and return the probability of event to happen one or more
        times within the time range defined by constructor's ``time_span``
        parameter value.

        Calculates probability as ``1 - e ** (-occurrence_rate*time_span)``.

        :param occurrence_rate:
            The average number of events per year.
        :return:
            Float value between 0 and 1 inclusive.
        """
        return 1 - math.exp(- occurrence_rate * self.time_span)

    def get_probability_one_occurrence(self, occurrence_rate):
        """
        Calculate and return the probability of event to occur once
        within the time range defined by the constructor's ``time_span``
        parameter value.
        """
        return scipy.stats.poisson(occurrence_rate * self.time_span).pmf(1)

    def sample_number_of_occurrences(self, occurrence_rate):
        """
        Draw a random sample from the distribution and return a number
        of events to occur.

        Method uses numpy random generator, which needs to be seeded
        outside of this method in order to get reproducible results.

        :param occurrence_rate:
            The average number of events per year.
        :return:
            Sampled integer number of events to occur within model's
            time span.
        """
        return numpy.random.poisson(occurrence_rate * self.time_span)

    def get_probability_no_exceedance(self, occurrence_rate, poes):
        """
        See :meth:`superclass method
        <BaseTOM.get_probability_no_exceedance>`
        for spec of input and result values.

        The probability is computed using the following formula ::

            (1 - e ** (-occurrence_rate * time_span)) ** poes
        """
        p = self.get_probability_one_or_more_occurrences(occurrence_rate)

        return (1 - p) ** poes
