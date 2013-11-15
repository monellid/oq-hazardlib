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
:mod:`openquake.hazardlib.calc.stochastic` contains
:func:`stochastic_event_set_poissonian`.
"""
from openquake.hazardlib.tom import PoissonTOM
from openquake.hazardlib.calc import filters


def stochastic_event_set_poissonian(
        sources, time_span,
        sites=None,
        source_site_filter=filters.source_site_noop_filter,
        rupture_site_filter=filters.rupture_site_noop_filter):
    """
    The Poissonian Stochastic Event Set calculator generates a 'Stochastic
    Event Set' (that is a collection of earthquake ruptures) by randomly
    sampling a source model whose rupture follow a Poissonian temporal
    occurrence model. The Stochastic Event Set represent a possible
    *realization* of the seismicity as described by the source model,
    in the given time span.

    The calculator assumes
    :class:`Poissonian <openquake.hazardlib.tom.PoissonTOM>` temporal
    occurrence model.

    :param sources:
        An iterator of seismic sources objects (instances of subclasses
        of :class:`~openquake.hazardlib.source.base.BaseSeismicSource`).
    :param time_span:
        An investigation period for Poissonian temporal occurrence model,
        floating point number in years.
    :param sites:
        A list of sites to consider (or None)
    :param source_site_filter:
        The source filter to use (only meaningful is sites is not None)
    :param source_site_filter:
        The rupture filter to use (only meaningful is sites is not None)
    :returns:
        Generator of :class:`~openquake.hazardlib.source.rupture.Rupture`
        objects that are contained in an event set. Some ruptures can be
        missing from it, others can appear one or more times in a row.
    """
    tom = PoissonTOM(time_span)
    if sites is None:  # no filtering
        for source in sources:
            try:
                for rupture in source.iter_ruptures(tom):
                    for i in xrange(rupture.sample_number_of_occurrences()):
                        yield rupture
            except Exception, err:
                msg = 'An error occurred with source id=%s. Error: %s'
                msg %= (source.source_id, err.message)
                raise RuntimeError(msg)
        return
    # else apply filtering
    sources_sites = source_site_filter((source, sites) for source in sources)
    for source, r_sites in sources_sites:
        try:
            ruptures_sites = rupture_site_filter(
                (rupture, r_sites) for rupture in source.iter_ruptures(tom))
            for rupture, _sites in ruptures_sites:
                for i in xrange(rupture.sample_number_of_occurrences()):
                    yield rupture
        except Exception, err:
            msg = 'An error occurred with source id=%s. Error: %s'
            msg %= (source.source_id, err.message)
            raise RuntimeError(msg)
