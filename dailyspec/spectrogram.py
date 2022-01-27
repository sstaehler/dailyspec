#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
Plot spectrograms in two different frequency channels

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2018
:license:
    None
'''

import argparse
from datetime import timedelta, datetime

import matplotlib.dates as mdates
import matplotlib.mlab as mlab
import matplotlib.pyplot
import matplotlib.pyplot as plt
import matplotlib.style as mstyle
import numpy as np
import obspy
from matplotlib.patches import Polygon
from obspy import UTCDateTime as utct
from obspy.signal.tf_misfit import cwt
from obspy.signal.util import next_pow_2

plt.rcParams['agg.path.chunksize'] = 1000
mstyle.use('seaborn-colorblind')

def plot_cwf(tr, fmin=1. / 50, fmax=1. / 2, w0=8):
    npts = tr.stats.npts
    dt = tr.stats.delta

    scalogram = abs(cwt(tr.data, dt, w0=w0, nf=200,
                        fmin=fmin, fmax=fmax))

    t = np.linspace(0, dt * npts, npts)
    f = np.logspace(np.log10(fmin),
                    np.log10(fmax),
                    scalogram.shape[0])
    return scalogram ** 2, f, t


def mark_event(ax_low: matplotlib.pyplot.Axes,
               ax_up: matplotlib.pyplot.Axes,
               fig: matplotlib.pyplot.Figure,
               catalog: obspy.Catalog,
               stats: dict,
               starttime=None, endtime=None,
               phase_list=('3.5kmps', 'PKIKP')):
    from obspy.taup import TauPyModel
    from obspy.geodetics import locations2degrees
    from matplotlib.dates import date2num
    from matplotlib.transforms import blended_transform_factory

    model = TauPyModel('iasp91')
    cat_trim = catalog.filter("magnitude >= 5.0",
                              "time >= %s" % (starttime - 3600),
                              "time < %s" % (endtime - 120))
    nevents = len(cat_trim)
    for i, event in enumerate(cat_trim):
        ievent = nevents - i
        origin = event.preferred_origin()
        distance = locations2degrees(
            lat1=origin.latitude,
            long1=origin.longitude,
            lat2=stats.latitude,
            long2=stats.longitude)

        trans = blended_transform_factory(ax_up.transData, fig.transFigure)
        ax_low.axvline(x=origin.time.datetime, c='white', lw=0.6, ls='dashed')
        ax_up.axvline(x=origin.time.datetime, c='white', lw=0.6, ls='dashed')

        ncols = 5
        x_event_label = 0.07 + 0.15 * ((ievent - 1) % ncols)
        y_event_label = 0.09 - 0.03 * ((ievent - 1) // ncols)

        event_label = '{1:s}\nM{2:3.1f} - {3:3d} deg'.format(
            ievent,
            event.event_descriptions[0]['text'],
            event.preferred_magnitude().mag,
            int(distance))
        bbox = dict(boxstyle="round", fc='C%d' % ievent)
        ax_up.annotate(text=event_label,
                       xy=(x_event_label, y_event_label),
                       xycoords=fig.transFigure,
                       textcoords=fig.transFigure,
                       ha='left', va='center',
                       fontsize=8)
        ax_up.annotate(text=str(ievent),
                       xy=(x_event_label - 0.005, y_event_label),
                       xycoords=fig.transFigure,
                       textcoords=fig.transFigure,
                       ha='right', va='center',
                       bbox=bbox, fontsize=10)

        ax_up.annotate(
            text=str(ievent),
            xy=(date2num(origin.time.datetime), 0.83),
            xycoords=trans, #ax.get_xaxis_transform(),
            textcoords=trans, #ax.get_xaxis_transform(),
            ha='center', va='bottom',
            fontsize=8,
            #rotation=270.,
            annotation_clip=True,
            bbox=bbox)

        if 'longitude' in stats:
            arrivals = model.get_travel_times_geo(
                source_depth_in_km=origin.depth*1e-3,
                source_latitude_in_deg=origin.latitude,
                source_longitude_in_deg=origin.longitude,
                receiver_latitude_in_deg=stats.latitude,
                receiver_longitude_in_deg=stats.longitude,
                phase_list=phase_list)
            for arrival in arrivals:
                t = arrival.time
                t_arrival = (origin.time + t)
                if 180. > arrival.purist_distance > 100. and arrival.name in ['PP', 'SS', '3.5kmps']:
                    ax_low.axvline(x=t_arrival.datetime, c='lightgrey', lw=0.5)
                    ax_up.axvline(x=t_arrival.datetime, c='lightgrey', lw=0.5)
                    ax_up.annotate(xy=(date2num(t_arrival.datetime), 0.83),
                                   xycoords=trans,
                                   textcoords=trans,
                                   fontsize=8,
                                   ha='center', va='bottom',
                                   #text=arrival.name)
                                   text='R1')
                    # ax.annotate(
                    #     text='', #f'display = ({t_arrival_t:.1f}, {ydisplay:.1f}), {arrival.name}',
                    #     xy=(t_arrival_t, -5), xycoords='axes pixels',
                    #     xytext=(t_origin_t, -20), textcoords='axes pixels',
                    #     ha='left', va='top',
                    #     rotation=270.,
                    #     bbox=bbox, arrowprops=arrowprops)
                elif arrival.purist_distance < 100. and arrival.name in ['P', 'S', '3.5kmps']:
                    ax_low.axvline(x=t_arrival.datetime, c='lightgrey', lw=0.5)
                    ax_up.axvline(x=t_arrival.datetime, c='lightgrey', lw=0.5)
                    ax_up.annotate(xy=(date2num(t_arrival.datetime), 0.83),
                                   xycoords=trans,
                                   textcoords=trans,
                                   fontsize=8,
                                   ha='center', va='bottom',
                                   text='R1')
                elif 180 < arrival.purist_distance < 270. and arrival.name in ['P', 'S', '3.5kmps']:
                    ax_low.axvline(x=t_arrival.datetime, c='lightgrey', lw=0.5)
                    ax_up.axvline(x=t_arrival.datetime, c='lightgrey', lw=0.5)
                    ax_up.annotate(xy=(date2num(t_arrival.datetime), 0.83),
                                   xycoords=trans,
                                   textcoords=trans,
                                   fontsize=8,
                                   ha='center', va='bottom',
                                   text='R2')
                                   # text=arrival.name)
                    # ax.annotate(
                    #     text='', #f'display = ({t_arrival_t:.1f}, {ydisplay:.1f}), {arrival.name}',
                    #     xy=(t_arrival_t, -5), xycoords='axes pixels',
                    #     xytext=(t_origin_t, -20), textcoords='axes pixels',
                    #     ha='left', va='top',
                    #     rotation=270.,
                    #     bbox=bbox, arrowprops=arrowprops)


def calc_specgram_dual(st_LF, st_HF,
                       winlen_sec_LF=1800,
                       winlen_sec_HF=10.0,
                       fmax=None,
                       overlap=0.5, kind='spec',
                       tstart=None, tend=None,
                       vmin=None, vmax=None, log=True,
                       ratio_LF_spec=0.6,
                       catalog=None,
                       show=False,
                       fnam=None,
                       noise=None):
    """
    Plot a dual spectrogram of a logarithmic low-frequency part and a linear high-frequency part above 1 Hz.
    :param st_LF: obspy.Stream
    :param st_HF: obspy.Stream
    :param winlen_sec_LF: float
        Window length for the LF spectrogram in seconds (and also lower frequency limit.)
        If kind=='cwt', it only controls the lower frequency limit of plot
    :param winlen_sec_HF: float
        Window length for the HF spectrogram.
    :param fmax: float
        Upper frequency limit of HF spectrogram.
    :param overlap: float
        Percentage of overlap (between 0 and 1) for spectrogram computation, if kind=='spec'
    :param kind: str
        'spec': compute classic spectrogram
        'cwt': compute continuous wavelet transform
    :param tstart: str
        Starting time for plot (lower limit of X-axis) in format that obspy.UTCTime understands
    :param tend: str
        End time for plot (lower limit of X-axis) in format that obspy.UTCTime understands
    :param vmin: float
        Lower limit of colorbar
    :param vmax: float
        Upper limit of colorbar
    :param log: bool
        Plot spectrogram logarithmic (default) or linear
    :param ratio_LF_spec: float
        How much of the height does the LF spectogram get?
    :param catalog: obspy.Catalog
        Catalog object to mark event times
    :param show: bool
        Show interactive matplotlib figure (default: False)
    :param fnam: str
        File name to save spectrogram into.
    :param noise:
        Plot Earth noise model on the right.
    """
    if tstart is None:
        tstart = st_LF[0].stats.starttime
    else:
        tstart = utct(tstart)
    if tend is None:
        tend = st_LF[0].stats.endtime
    else:
        tend = utct(tend)

    # Some input checking
    freq_HF = st_HF[0].stats.sampling_rate
    if freq_HF <= 2.0:
        raise KeyError('Sampling rate of HF data must be > 2sps, is %4.1fsps' %
                       freq_HF)
    winlen_LF = int(winlen_sec_LF * st_LF[0].stats.sampling_rate)
    winlen_HF = int(winlen_sec_HF * st_HF[0].stats.sampling_rate)

    fmin_LF = 2. / winlen_sec_LF
    fmax_LF = 1.0

    fmin_HF = 1.0
    if fmax is None:
        fmax_HF = freq_HF / 2.
    else:
        fmax_HF = fmax

    st_LF.filter('highpass', freq=fmin_LF * 0.9,
                 zerophase=True, corners=6)
    st_HF.filter('bandpass', freqmin=fmin_HF * 0.5, freqmax=fmax_HF,
                 zerophase=True, corners=6)

    fig, ax_cb, ax_psd_HF, ax_psd_LF, \
    ax_seis_HF, ax_seis_LF, \
    ax_spec_HF, ax_spec_LF, = create_axes(ratio_LF_height=ratio_LF_spec)

    for tr in st_LF:
        t = np.arange(utct(tr.stats.starttime).datetime,
                      utct(tr.stats.endtime + tr.stats.delta).datetime,
                      timedelta(seconds=tr.stats.delta)).astype(datetime)
        ax_seis_LF.plot(t, tr.data * 1e6,
                        'navy', lw=0.3)
    for tr in st_HF:
        t = np.arange(utct(tr.stats.starttime).datetime,
                      utct(tr.stats.endtime + tr.stats.delta).datetime,
                      timedelta(seconds=tr.stats.delta)).astype(datetime)
        ax_seis_HF.plot(t, tr.data * 1e6,
                        'darkred', lw=0.3)

    for st, ax_spec, ax_psd, flim, winlen, cmap, vlim in \
            zip([st_LF, st_HF],
                [ax_spec_LF, ax_spec_HF],
                [ax_psd_LF, ax_psd_HF],
                [(fmin_LF, fmax_LF),
                 (fmin_HF, fmax_HF)],
                [winlen_LF, winlen_HF],
                ['inferno', 'inferno'],
                [(vmin, vmax), (vmin, vmax)]):

        for tr in st:
            # print(winlen / tr.stats.sampling_rate, tr.stats.sampling_rate)
            if (kind == 'cwt' and winlen / tr.stats.sampling_rate < 20.) or \
                    (kind == 'spec'):  # and winlen < 600 * tr.stats.sampling_rate):
                p, f, t = mlab.specgram(tr.data, NFFT=winlen,
                                        Fs=tr.stats.sampling_rate,
                                        noverlap=int(winlen * overlap),
                                        pad_to=next_pow_2(winlen) * 4)
            else:
                p, f, t = plot_cwf(tr, w0=12,
                                   fmin=flim[0],
                                   fmax=flim[1])
            if len(f) > 200:
                df = 4
            else:
                df = 2
            dt = 8
            p = p[::df, ::dt]
            f = f[::df]
            t = t[::dt]

            t += float(tr.stats.starttime)
            delta = t[1] - t[0]
            t = np.arange(utct(t[0]).datetime,
                          utct(t[-1] + delta * 0.9).datetime,
                          timedelta(seconds=delta)).astype(datetime)

            bol = np.array((f >= flim[0] * 0.9, f <= flim[1])).all(axis=0)

            vmin = vlim[0]
            vmax = vlim[1]
            if vmin is None:
                vmin = np.percentile(10 * np.log10(p[bol, :]), q=1, axis=None)
            if vmax is None:
                vmax = np.percentile(10 * np.log10(p[bol, :]), q=90, axis=None)

            if log:
                ax_spec.pcolormesh(t, f[bol], 10 * np.log10(p[bol, :]),
                                   vmin=vmin, vmax=vmax,
                                   shading='nearest',
                                   cmap=cmap)
            else:
                ax_spec.pcolormesh(t, f[bol], p[bol, :],
                                   vmin=vmin, vmax=vmax,
                                   shading='nearest',
                                   cmap=cmap)

            plot_psd(ax_psd, bol, f, p)

        ax_psd.set_xlim(vmin, vmax)

        if noise == 'Earth':
            plt_earth_noise(ax_psd)

    format_axes(ax_cb, ax_psd_HF, ax_psd_LF, ax_seis_HF, ax_seis_LF,
                ax_spec_HF, ax_spec_LF, fmax_HF,
                fmax_LF, fmin_HF, fmin_LF, st_HF, st_LF,
                tstart, tend
                )
    if catalog is not None:
        mark_event(ax_low=ax_spec_LF,
                   ax_up=ax_spec_HF,
                   fig=fig,
                   catalog=catalog,
                   stats=st_LF[0].stats,
                   starttime=tstart,
                   endtime=tend)

    if fnam:
        plt.savefig(fnam, dpi=240)
        if show:
            plt.show()
    else:
        plt.show()
    plt.close('all')


def plt_earth_noise(ax_psd):
    nhnm = obspy.signal.spectral_estimation.get_nhnm()
    nlnm = obspy.signal.spectral_estimation.get_nlnm()
    ix = nlnm[1]
    iy = 1. / nlnm[0]
    verts = [(-300, 1. / nlnm[0][0])] + list(zip(ix, iy)) + [
        (-300, 1. / nlnm[0][-1])]
    poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
    ax_psd.add_patch(poly)
    ix = nhnm[1]
    iy = 1. / nhnm[0]
    verts = [(0, 1. / nhnm[0][0])] + list(zip(ix, iy)) + [(0, 1. / nhnm[0][-1])]
    poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
    ax_psd.add_patch(poly)
    ax_psd.plot(nhnm[1], 1. / nhnm[0], color='darkgrey', linestyle='dashed')
    ax_psd.plot(nlnm[1], 1. / nlnm[0], color='darkgrey', linestyle='dashed')


def plot_psd(ax_psd, bol, f, p):
    median = np.percentile(p[bol, :], axis=1, q=50)
    perc_95 = np.percentile(p[bol, :], axis=1, q=95)
    perc_05 = np.percentile(p[bol, :], axis=1, q=5)
    perc_99 = np.percentile(p[bol, :], axis=1, q=99)
    perc_01 = np.percentile(p[bol, :], axis=1, q=1)
    ax_psd.plot(10 * np.log10(median), f[bol],
                color='indigo')
    ax_psd.plot(10 * np.log10(perc_95), f[bol],
                color='darkgrey', linestyle='dashed')
    ax_psd.plot(10 * np.log10(perc_05), f[bol],
                color='darkgrey', linestyle='dashed')
    ax_psd.plot(10 * np.log10(perc_99), f[bol],
                color='indigo', linestyle='dotted')
    ax_psd.plot(10 * np.log10(perc_01), f[bol],
                color='indigo', linestyle='dotted')


def format_axes(ax_cb, ax_psd_HF, ax_psd_LF, ax_seis_HF, ax_seis_LF,
                ax_spec_HF, ax_spec_LF, fmax_HF,
                fmax_LF, fmin_HF, fmin_LF, st_HF, st_LF,
                tstart, tend
                ):
    ax_seis_LF.set_ylim(st_LF[0].std() * 1e7 * np.asarray([-3., 1.]))
    ax_seis_HF.set_ylim(st_HF[0].std() * 1e7 * np.asarray([-2., 2.]))
    ax_seis_LF.set_ylabel('%s [um/s²] \n< 1 Hz' % st_LF[0].stats.channel,
                          color='navy')
    ax_seis_HF.set_ylabel('%s [um/s] \n> 1 Hz' % st_HF[0].stats.channel,
                          color='darkred')
    ax_seis_LF.tick_params('y', colors='navy')
    ax_seis_HF.tick_params('y', colors='darkred')
    ax_spec_HF.set_ylim(fmin_HF, fmax_HF)
    ax_spec_HF.set_ylabel('frequency / Hz', fontsize=12)
    ax_spec_LF.set_yscale('log')
    ax_spec_LF.set_ylabel('period / seconds', fontsize=12)
    # This needs to be done for both axes, otherwise the PSD and the Spec axis
    # plot weird yticks on top of each other
    for ax in [ax_spec_LF, ax_psd_LF]:
        ax.set_ylim(fmin_LF, fmax_LF)
        tickvals = np.asarray(
            [1. / 2, 1. / 5, 1. / 10, 1. / 20, 1. / 50, 1. / 100,
             1. / 200., 1. / 500, 1. / 1000., 1. / 2000, 1. / 5000, 1. / 10000])
        ax.set_yticks(tickvals)  # [tickvals >= fmin_LF])
        ax.set_yticklabels(['2', '5', '10', '20', '50', '100', '200', '500',
                            '1000', '2000', '5000', '10000'])
        ax.set_yticklabels([], minor=True)
    for ax_psd in [ax_psd_HF, ax_psd_LF]:
        ax_psd.yaxis.set_label_position("right")
        ax_psd.yaxis.set_ticks_position("right")
    ax_psd_LF.set_ylim(fmin_LF, fmax_LF)

    ax_psd_LF.set_xlabel('PSD (m/s²)²/Hz [dB]')
    ax_psd_LF.set_ylabel('period / seconds', fontsize=12)
    ax_psd_HF.set_ylabel('frequency / Hz', fontsize=12)
    ax_psd_HF.set_xticks([])
    # make unnecessary labels disappear
    for ax in [ax_spec_HF, ax_seis_LF]:
        plt.setp(ax.get_xticklabels(), visible=False)

    for ax in [ax_spec_HF, ax_spec_LF, ax_seis_LF]:
        ax.set_xlim(tstart.datetime, tend.datetime)

    # Axis with colorbar
    mappable = ax_spec_HF.collections[0]
    plt.colorbar(mappable=mappable, cax=ax_cb)
    ax_cb.set_ylabel('PSD (m/s²)²/Hz')

    locator = mdates.AutoDateLocator(minticks=6, maxticks=9)
    formatter = mdates.ConciseDateFormatter(locator)
    ax_spec_LF.xaxis.set_major_locator(locator)
    ax_spec_LF.xaxis.set_major_formatter(formatter)
    # hours = mdates.HourLocator()
    # mins = mdates.MinuteLocator(interval=10)
    # # format the ticks
    # for ax in (ax_spec_HF, ax_spec_LF):
    #     ax.xaxis.set_major_locator(hours)
    #     ax.xaxis.set_minor_locator(mins)


def create_axes(ratio_LF_height=0.6):
    """
    Create necessary axis objects
    :param ratio_LF_height: float
        Ratio of LF spectrograms height to total spectrogram height
    :return: matplotlib.Axis
    """
    fig = plt.figure(figsize=(16, 9))
    # [left bottom width height]

    h_spec_total = 0.7
    h_base = 0.13

    h_spec_LF = h_spec_total * ratio_LF_height
    h_spec_HF = h_spec_total - h_spec_LF
    h_seis = 0.15  # 0.2
    w_base = 0.06
    w_spec = 0.8
    w_psd = 0.1
    ax_seis_LF = fig.add_axes([w_base, h_base + h_spec_total,
                               w_spec, h_seis],
                              label='seismogram LF')
    ax_seis_HF = ax_seis_LF.twinx()
    ax_spec_LF = fig.add_axes([w_base, h_base, w_spec, h_spec_LF],
                              sharex=ax_seis_LF,
                              label='spectrogram LF')
    ax_spec_HF = fig.add_axes([w_base, h_base + h_spec_LF, w_spec, h_spec_HF],
                              sharex=ax_seis_LF,
                              label='spectrogram HF')
    ax_psd_LF = fig.add_axes([w_base + w_spec, h_base, w_psd, h_spec_LF],
                             sharey=ax_spec_LF,
                             label='PSD LF')
    ax_psd_HF = fig.add_axes([w_base + w_spec, h_base + h_spec_LF,
                              0.1, h_spec_HF],
                             sharey=ax_spec_HF,
                             label='PSD HF')
    # Colorbar axis
    ax_cb = fig.add_axes([w_base + w_spec + w_psd / 2.,
                          h_base + h_spec_HF + h_spec_LF + h_seis * 0.1,
                          w_psd * 0.2,
                          h_seis * 0.8],
                         label='colorbar')
    return fig, ax_cb, ax_psd_HF, ax_psd_LF, ax_seis_HF, ax_seis_LF, \
           ax_spec_HF, ax_spec_LF


def def_args():
    helptext = 'Calculate spectograms to quickly check content of hours ' \
               'to days of seismic data. ' \
               'The code assumes that the input files are instrument  ' \
               'corrected to acceleration. ' \
               'If this is not the case, adapt the value range (dBmin, dBmax)'
    parser = argparse.ArgumentParser(helptext)
    parser.add_argument('--lp', nargs='+', required=True,
                        help='Data file(s) containing the long period data')
    parser.add_argument('--hf', nargs='+', required=True,
                        help='Data file(s) containing the high frequency data')
    parser.add_argument('--p', nargs='+', required=True,
                        help='Data file(s) containing the pressure time series')
    parser.add_argument('--dBmin', type=float, default=-180,
                        help='Minimum value for spectrograms (in dB)')
    parser.add_argument('--dBmax', type=float, default=-100,
                        help='Maximum value for spectrograms (in dB)')
    parser.add_argument('--tstart', default=None,
                        help='Start time for spectrogram' +
                             '(in any format that datetime understands)')
    parser.add_argument('--tend', default=None,
                        help='End time for spectrogram')
    parser.add_argument('--winlen', default=100., type=float,
                        help='Window length for long-period spectrogram (in '
                             'seconds)')
    parser.add_argument('--no_noise', default=False, action='store_true',
                        help='Omit plotting the NLNM/NHNM curves')
    parser.add_argument('--fmax', default=5.0, type=float,
                        help='Omit plotting the NLNM/NHNM curves')
    parser.add_argument('-i', '--interactive', default=False,
                        action='store_true',
                        help='Open a matplotlib window instead of saving to '
                             'disk')

    parser.add_argument('--kind', default='spec',
                        help='Calculate spectrogram (spec) or continuous '
                             'wavelet transform (cwt, much slower)? Default: '
                             'spec')
    return parser.parse_args()


if __name__ == '__main__':
    args = def_args()

    starttime = None
    endtime = None
    if args.tstart:
        starttime = obspy.UTCDateTime(args.tstart)
    if args.tstart:
        endtime = obspy.UTCDateTime(args.tend)
    print(starttime)
    print(endtime)

    st_LF = obspy.Stream()
    for file in args.lp:
        st_LF += obspy.read(file, starttime=starttime - 120.,
                            endtime=endtime + 120.)
    st_HF = obspy.Stream()
    for file in args.hf:
        st_HF += obspy.read(file, starttime=starttime - 120.,
                            endtime=endtime + 120.)

    if st_LF[0].stats.sampling_rate > 5.0:
        dec_fac = int(st_LF[0].stats.sampling_rate / 5.0)
        st_LF.decimate(dec_fac)

    for st in [st_HF, st_LF]:
        st.detrend()
        st.filter('highpass', freq=1. / args.winlen)
        st.trim(starttime=starttime,
                endtime=endtime)

    if args.interactive:
        fnam_out = None
    else:
        fnam_templ_1 = "spectrogram_{network}.{location}.{station}." \
                       "{channel}_vs_"
        fnam_templ_2 = "{network}.{location}.{station}.{channel}-" \
                       "{starttime.year}.{starttime.julday}.png"
        fnam_out = fnam_templ_1.format(**st_LF[0].stats) + \
                   fnam_templ_2.format(**st_HF[0].stats)
    if args.no_noise:
        noise = None
    else:
        noise = 'Earth'
    calc_specgram_dual(st_LF, st_HF,
                       winlen_sec_LF=args.winlen,
                       winlen_sec_HF=10.,
                       vmin=args.dBmin,
                       vmax=args.dBmax,
                       fmax=args.fmax,
                       noise=noise,
                       fnam=fnam_out,
                       kind=args.kind)
