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
:mod:`openquake.hazardlib.calc.disagg` contains
:func:`disaggregation_poissonian` as well as several aggregation functions for
extracting a specific PMF from the result of :func:`disaggregation_poissonian`.
"""
import numpy
import warnings

from openquake.hazardlib.calc import filters
from openquake.hazardlib.geo.geodetic import npoints_between
from openquake.hazardlib.geo.utils import get_longitudinal_extent
from openquake.hazardlib.geo.utils import get_spherical_bounding_box
from openquake.hazardlib.site import SiteCollection
from openquake.hazardlib.tom import PoissonTOM


def disaggregation_poissonian(
        sources, site, imt, iml, gsims, time_span, truncation_level,
        n_epsilons, mag_bin_width, dist_bin_width, coord_bin_width,
        source_site_filter=filters.source_site_noop_filter,
        rupture_site_filter=filters.rupture_site_noop_filter):
    """
    Compute "Disaggregation" matrix representing conditional probability of an
    intensity mesaure type ``imt`` exceeding, at least once, an intensity
    measure level ``iml`` at a geographical location ``site``, given rupture
    scenarios (belonging to a Poissonian source model) classified in terms of:

    - rupture magnitude
    - Joyner-Boore distance from rupture surface to site
    - longitude and latitude of the surface projection of a rupture's point
      closest to ``site``
    - epsilon: number of standard deviations by which an intensity measure
      level deviates from the median value predicted by a GSIM, given the
      rupture parameters
    - rupture tectonic region type

    In other words, the disaggregation matrix allows to compute the probability
    of each scenario with the specified properties (e.g., magnitude, or the
    magnitude and distance) to cause one or more exceedences of a given hazard
    level.

    For more detailed information about the disaggregation, see for instance
    "Disaggregation of Seismic Hazard", Paolo Bazzurro, C. Allin Cornell,
    Bulletin of the Seismological Society of America, Vol. 89, pp. 501-520,
    April 1999.

    :param sources:
        Seismic source model, as for
        :mod:`PSHA <openquake.hazardlib.calc.hazard_curve>` calculator it
        should be an iterator of seismic sources.
    :param site:
        :class:`~openquake.hazardlib.site.Site` of interest to calculate
        disaggregation matrix for.
    :param imt:
        Instance of :mod:`intensity measure type <openquake.hazardlib.imt>`
        class.
    :param iml:
        Intensity measure level. A float value in units of ``imt``.
    :param gsims:
        Tectonic region type to GSIM objects mapping.
    :param time_span:
        Investigation time span, in years.
    :param truncation_level:
        Float, number of standard deviations for truncation of the intensity
        distribution.
    :param n_epsilons:
        Integer number of epsilon histogram bins in the result matrix.
    :param mag_bin_width:
        Magnitude discretization step, width of one magnitude histogram bin.
    :param dist_bin_width:
        Distance histogram discretization step, in km.
    :param coord_bin_width:
        Longitude and latitude histograms discretization step,
        in decimal degrees.
    :param source_site_filter:
        Optional source-site filter function. See
        :mod:`openquake.hazardlib.calc.filters`.
    :param rupture_site_filter:
        Optional rupture-site filter function. See
        :mod:`openquake.hazardlib.calc.filters`.

    :returns:
        A tuple of two items. First is itself a tuple of bin edges information
        for (in specified order) magnitude, distance, longitude, latitude,
        epsilon and tectonic region types.

        Second item is 6d-array representing the full disaggregation matrix.
        Dimensions are in the same order as bin edges in the first item
        of the result tuple. The matrix can be used directly by pmf-extractor
        functions.
    """
    tom = PoissonTOM(time_span)

    bins_data = _collect_bins_data(sources, site, imt, iml, gsims, tom,
                                   truncation_level, n_epsilons,
                                   source_site_filter, rupture_site_filter)
    if all([len(x) == 0 for x in bins_data]):
        # No ruptures have contributed to the hazard level at this site.
        warnings.warn(
            'No ruptures have contributed to the hazard at site %s'
            % site,
            RuntimeWarning
        )
        return None, None

    bin_edges = _define_bins(bins_data, mag_bin_width, dist_bin_width,
                             coord_bin_width, truncation_level, n_epsilons)
    diss_matrix = _arrange_data_in_bins(bins_data, bin_edges)
    return bin_edges, diss_matrix


# DEPRECATED
def disaggregation(sources, site, imt, iml, gsims, tom,
                   truncation_level, n_epsilons,
                   mag_bin_width, dist_bin_width, coord_bin_width,
                   source_site_filter=filters.source_site_noop_filter,
                   rupture_site_filter=filters.rupture_site_noop_filter):
    """
    An implementation of the now-deprecated disaggregation calculation
    interface. Please use :func:`disaggregation_poissonian` instead.
    """
    warnings.warn(
        '`openquake.hazardlib.calc.disagg.disaggregation` is deprecated. '
        'Please use '
        '`openquake.hazardlib.calc.disagg.disaggregation_poissonian` instead',
        RuntimeWarning
    )

    time_span = tom.time_span
    return disaggregation_poissonian(
        sources, site, imt, iml, gsims, time_span, truncation_level,
        n_epsilons, mag_bin_width, dist_bin_width, coord_bin_width,
        source_site_filter=filters.source_site_noop_filter,
        rupture_site_filter=filters.rupture_site_noop_filter
    )


def _collect_bins_data(sources, site, imt, iml, gsims, tom,
                       truncation_level, n_epsilons,
                       source_site_filter, rupture_site_filter):
    """
    Extract values of magnitude, distance, closest point, tectonic region
    types and PoE distribution.

    This method processes the source model (generates ruptures) and collects
    all needed parameters to arrays. It also defines tectonic region type
    bins sequence.
    """
    mags = []
    dists = []
    lons = []
    lats = []
    tect_reg_types = []
    probs_one_or_more = []
    probs_exceed_given_rup = []
    src_idxs = []
    sitecol = SiteCollection([site])
    sitemesh = sitecol.mesh

    _next_trt_num = 0
    trt_nums = {}

    sources_sites = ((source, sitecol) for source in sources)
    # here we ignore filtered site collection because either it is the same
    # as the original one (with one site), or the source/rupture is filtered
    # out and doesn't show up in the filter's output
    for src_idx, (source, s_sites) in \
            enumerate(source_site_filter(sources_sites)):
        try:
            tect_reg = source.tectonic_region_type
            gsim = gsims[tect_reg]

            if not tect_reg in trt_nums:
                trt_nums[tect_reg] = _next_trt_num
                _next_trt_num += 1
            tect_reg = trt_nums[tect_reg]

            ruptures_sites = ((rupture, s_sites)
                              for rupture in source.iter_ruptures())
            for rupture, r_sites in rupture_site_filter(ruptures_sites):
                # extract rupture parameters of interest
                mags.append(rupture.mag)
                [jb_dist] = rupture.surface.get_joyner_boore_distance(sitemesh)
                dists.append(jb_dist)
                [closest_point] = rupture.surface.get_closest_points(sitemesh)
                lons.append(closest_point.longitude)
                lats.append(closest_point.latitude)
                tect_reg_types.append(tect_reg)

                # compute conditional probability of exceeding iml given
                # the current rupture, and different epsilon level, that is
                # ``P(IMT >= iml | rup, epsilon_bin)`` for each of epsilon bins
                sctx, rctx, dctx = gsim.make_contexts(sitecol, rupture)
                [poes_given_rup_eps] = gsim.disaggregate_poe(
                    sctx, rctx, dctx, imt, iml, truncation_level, n_epsilons
                )

                # compute probability of one or more rupture occurrences
                probs_one_or_more.append(
                    rupture.get_probability_one_or_more_occurrences()
                )

                # collect probability of exceedance given the rupture
                probs_exceed_given_rup.append(poes_given_rup_eps)

                # keep track of the source index, so that the probabilities can
                # be associated to each source
                src_idxs.append(src_idx)
        except Exception, err:
            msg = 'An error occurred with source id=%s. Error: %s'
            msg %= (source.source_id, err.message)
            raise RuntimeError(msg)

    mags = numpy.array(mags, float)
    dists = numpy.array(dists, float)
    lons = numpy.array(lons, float)
    lats = numpy.array(lats, float)
    tect_reg_types = numpy.array(tect_reg_types, int)
    probs_one_or_more = numpy.array(probs_one_or_more, float)
    probs_exceed_given_rup = numpy.array(probs_exceed_given_rup, float)
    src_idxs = numpy.array(src_idxs, int)

    trt_bins = [
        trt for (num, trt) in sorted((num, trt)
                                     for (trt, num) in trt_nums.items())
    ]

    return (mags, dists, lons, lats, tect_reg_types, trt_bins,
            probs_one_or_more, probs_exceed_given_rup, src_idxs)


def _define_bins(bins_data, mag_bin_width, dist_bin_width,
                 coord_bin_width, truncation_level, n_epsilons):
    """
    Define bin edges for disaggregation histograms.

    Given bins data as provided by :func:`_collect_bins_data`, this function
    finds edges of histograms, taking into account maximum and minimum values
    of magnitude, distance and coordinates as well as requested sizes/numbers
    of bins.
    """
    mags, dists, lons, lats, tect_reg_types, trt_bins, _, _, _ = bins_data

    mag_bins = mag_bin_width * numpy.arange(
        int(numpy.floor(mags.min() / mag_bin_width)),
        int(numpy.ceil(mags.max() / mag_bin_width) + 1)
    )

    dist_bins = dist_bin_width * numpy.arange(
        int(numpy.floor(dists.min() / dist_bin_width)),
        int(numpy.ceil(dists.max() / dist_bin_width) + 1)
    )

    west, east, north, south = get_spherical_bounding_box(lons, lats)
    west = numpy.floor(west / coord_bin_width) * coord_bin_width
    east = numpy.ceil(east / coord_bin_width) * coord_bin_width
    lon_extent = get_longitudinal_extent(west, east)
    lon_bins, _, _ = npoints_between(
        west, 0, 0, east, 0, 0,
        numpy.round(lon_extent / coord_bin_width) + 1
    )

    lat_bins = coord_bin_width * numpy.arange(
        int(numpy.floor(south / coord_bin_width)),
        int(numpy.ceil(north / coord_bin_width) + 1)
    )

    eps_bins = numpy.linspace(-truncation_level, truncation_level,
                              n_epsilons + 1)

    return mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, trt_bins


def _arrange_data_in_bins(bins_data, bin_edges):
    """
    Given bins data, as it comes from :func:`_collect_bins_data`, and bin edges
    from :func:`_define_bins`, create a normalized 6d disaggregation matrix.
    """
    (mags, dists, lons, lats, tect_reg_types, trt_bins, probs_one_or_more,
     probs_exceed_given_rup, src_idxs) = bins_data
    mag_bins, dist_bins, lon_bins, lat_bins, eps_bins, trt_bins = bin_edges
    shape = (len(mag_bins) - 1, len(dist_bins) - 1, len(lon_bins) - 1,
             len(lat_bins) - 1, len(eps_bins) - 1, len(trt_bins))
    diss_matrix = numpy.zeros(shape)

    src_indices = numpy.unique(src_idxs)

    for i_mag in xrange(len(mag_bins) - 1):
        mag_idx = mags <= mag_bins[i_mag + 1]
        if i_mag != 0:
            mag_idx &= mags > mag_bins[i_mag]

        for i_dist in xrange(len(dist_bins) - 1):
            dist_idx = dists <= dist_bins[i_dist + 1]
            if i_dist != 0:
                dist_idx &= dists > dist_bins[i_dist]

            for i_lon in xrange(len(lon_bins) - 1):
                extents = get_longitudinal_extent(lons, lon_bins[i_lon + 1])
                lon_idx = extents >= 0
                if i_lon != 0:
                    extents = get_longitudinal_extent(lon_bins[i_lon], lons)
                    lon_idx &= extents > 0

                for i_lat in xrange(len(lat_bins) - 1):
                    lat_idx = lats <= lat_bins[i_lat + 1]
                    if i_lat != 0:
                        lat_idx &= lats > lat_bins[i_lat]

                    for i_eps in xrange(len(eps_bins) - 1):

                        for i_trt in xrange(len(trt_bins)):
                            trt_idx = tect_reg_types == i_trt

                            diss_idx = (i_mag, i_dist, i_lon,
                                        i_lat, i_eps, i_trt)

                            prob_idx = (mag_idx & dist_idx & lon_idx
                                        & lat_idx & trt_idx)

                            poe = numpy.prod(
                                (1 - probs_one_or_more[prob_idx]) **
                                probs_exceed_given_rup[prob_idx, i_eps]
                            )
                            poe = 1 - poe

                            diss_matrix[diss_idx] = poe

    return diss_matrix


def mag_pmf(matrix):
    """
    Fold full disaggregation matrix to magnitude PMF.

    :returns:
        1d array, a histogram representing magnitude PMF.
    """
    nmags, ndists, nlons, nlats, neps, ntrts = matrix.shape
    mag_pmf = numpy.zeros(nmags)
    for i in xrange(nmags):
        mag_pmf[i] = numpy.prod(
            [1 - matrix[i][j][k][l][m][n]
             for j in xrange(ndists)
             for k in xrange(nlons)
             for l in xrange(nlats)
             for m in xrange(neps)
             for n in xrange(ntrts)]
        )
    return 1 - mag_pmf


def dist_pmf(matrix):
    """
    Fold full disaggregation matrix to distance PMF.

    :returns:
        1d array, a histogram representing distance PMF.
    """
    nmags, ndists, nlons, nlats, neps, ntrts = matrix.shape
    dist_pmf = numpy.zeros(ndists)
    for j in xrange(ndists):
        dist_pmf[j] = numpy.prod(
            [1 - matrix[i][j][k][l][m][n]
             for i in xrange(nmags)
             for k in xrange(nlons)
             for l in xrange(nlats)
             for m in xrange(neps)
             for n in xrange(ntrts)]
        )
    return 1 - dist_pmf


def trt_pmf(matrix):
    """
    Fold full disaggregation matrix to tectonic region type PMF.

    :returns:
        1d array, a histogram representing tectonic region type PMF.
    """
    nmags, ndists, nlons, nlats, neps, ntrts = matrix.shape
    trt_pmf = numpy.zeros(ntrts)
    for n in xrange(ntrts):
        trt_pmf[n] = numpy.prod(
            [1 - matrix[i][j][k][l][m][n]
             for i in xrange(nmags)
             for j in xrange(ndists)
             for k in xrange(nlons)
             for l in xrange(nlats)
             for m in xrange(neps)]
        )
    return 1 - trt_pmf


def mag_dist_pmf(matrix):
    """
    Fold full disaggregation matrix to magnitude / distance PMF.

    :returns:
        2d array. First dimension represents magnitude histogram bins,
        second one -- distance histogram bins.
    """
    nmags, ndists, nlons, nlats, neps, ntrts = matrix.shape
    mag_dist_pmf = numpy.zeros((nmags, ndists))
    for i in xrange(nmags):
        for j in xrange(ndists):
            mag_dist_pmf[i][j] = numpy.prod(
                [1 - matrix[i][j][k][l][m][n]
                 for k in xrange(nlons)
                 for l in xrange(nlats)
                 for m in xrange(neps)
                 for n in xrange(ntrts)]
            )
    return 1 - mag_dist_pmf


def mag_dist_eps_pmf(matrix):
    """
    Fold full disaggregation matrix to magnitude / distance / epsilon PMF.

    :returns:
        3d array. First dimension represents magnitude histogram bins,
        second one -- distance histogram bins, third one -- epsilon
        histogram bins.
    """
    nmags, ndists, nlons, nlats, neps, ntrts = matrix.shape
    mag_dist_eps_pmf = numpy.zeros((nmags, ndists, neps))
    for i in xrange(nmags):
        for j in xrange(ndists):
            for m in xrange(neps):
                mag_dist_eps_pmf[i][j][m] = numpy.prod(
                    [1 - matrix[i][j][k][l][m][n]
                     for k in xrange(nlons)
                     for l in xrange(nlats)
                     for n in xrange(ntrts)]
                )
    return 1 - mag_dist_eps_pmf


def lon_lat_pmf(matrix):
    """
    Fold full disaggregation matrix to longitude / latitude PMF.

    :returns:
        2d array. First dimension represents longitude histogram bins,
        second one -- latitude histogram bins.
    """
    nmags, ndists, nlons, nlats, neps, ntrts = matrix.shape
    lon_lat_pmf = numpy.zeros((nlons, nlats))
    for k in xrange(nlons):
        for l in xrange(nlats):
            lon_lat_pmf[k][l] = numpy.prod(
                [1 - matrix[i][j][k][l][m][n]
                 for i in xrange(nmags)
                 for j in xrange(ndists)
                 for m in xrange(neps)
                 for n in xrange(ntrts)]
            )
    return 1 - lon_lat_pmf


def mag_lon_lat_pmf(matrix):
    """
    Fold full disaggregation matrix to magnitude / longitude / latitude PMF.

    :returns:
        3d array. First dimension represents magnitude histogram bins,
        second one -- longitude histogram bins, third one -- latitude
        histogram bins.
    """
    nmags, ndists, nlons, nlats, neps, ntrts = matrix.shape
    mag_lon_lat_pmf = numpy.zeros((nmags, nlons, nlats))
    for i in xrange(nmags):
        for k in xrange(nlons):
            for l in xrange(nlats):
                mag_lon_lat_pmf[i][k][l] = numpy.prod(
                    [1 - matrix[i][j][k][l][m][n]
                     for j in xrange(ndists)
                     for m in xrange(neps)
                     for n in xrange(ntrts)]
                )
    return 1 - mag_lon_lat_pmf


def lon_lat_trt_pmf(matrix):
    """
    Fold full disaggregation matrix to longitude / latitude / tectonic region
    type PMF.

    :returns:
        3d array. Dimension represent longitude, latitude and tectonic region
        type histogram bins respectively.
    """
    nmags, ndists, nlons, nlats, neps, ntrts = matrix.shape
    lon_lat_trt_pmf = numpy.zeros((nlons, nlats, ntrts))
    for k in xrange(nlons):
        for l in xrange(nlats):
            for n in xrange(ntrts):
                lon_lat_trt_pmf[k][l][n] = numpy.prod(
                    [1 - matrix[i][j][k][l][m][n]
                     for i in xrange(nmags)
                     for j in xrange(ndists)
                     for m in xrange(neps)]
                )
    return 1 - lon_lat_trt_pmf
