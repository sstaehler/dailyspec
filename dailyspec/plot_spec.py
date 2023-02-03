#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
"""
from argparse import ArgumentParser

def define_arguments():
    helptext = 'Plot spectogram of data'
    parser = ArgumentParser(description=helptext)

    helptext = 'Data files'
    parser.add_argument('-d', '--data_files', nargs='+', help=helptext)

    helptext = 'Inventory file'
    parser.add_argument('-i', '--inventory_file', help=helptext, default=None)

    helptext = 'Catalog file'
    parser.add_argument('-c', '--catalog_file', help=helptext, default=None)

    helptext = 'Maximum frequency for plot'
    parser.add_argument('--fmax', default=50., type=float, help=helptext)

    parser.add_argument('--dBmin', type=float, default=-180,
                        help='Minimum value for spectrograms (in dB)')
    parser.add_argument('--dBmax', type=float, default=-100,
                        help='Maximum value for spectrograms (in dB)')
    parser.add_argument('--tstart', default=None,
                        help='Start time for spectrogram' +
                             '(in any format that datetime understands)')
    parser.add_argument('--tend', default=None,
                        help='End time for spectrogram')
    parser.add_argument('--plot_ratio', default=0.6, type=float,
                        help='Window length for long-period spectrogram (in '
                             'seconds)')
    parser.add_argument('--winlen', default=100., type=float,
                        help='Window length for long-period spectrogram (in '
                             'seconds)')
    parser.add_argument('--w0', default=10, type=int,
                        help='Tradeoff between time and frequency resolution in CWT' +
                             'Lower numbers: better time resolution\n' +
                             'Higher numbers: better freq resolution')
    parser.add_argument('--figsize', nargs=2, default=(16, 9), type=float,
                        help='Size of the produced figure in Inches. Default is 16x9, which is good for high' +
                             'resolution screen display.')
    parser.add_argument('--winlen_HF', default=4., type=float,
                        help='Window length for high-frequency spectrogram (in '
                             'seconds)')
    parser.add_argument('--no_noise', default=False, action='store_true',
                        help='Omit plotting the NLNM/NHNM curves')
    parser.add_argument('--interactive', default=False,
                        action='store_true',
                        help='Open a matplotlib window instead of saving to '
                             'disk')

    parser.add_argument('--unit', default='VEL',
                        help='Unit of input data. Options: ACC, VEL, DIS. Plot is in acceleration.')

    parser.add_argument('--kind', default='spec',
                        help='Calculate spectrogram (spec) or continuous '
                             'wavelet transfort (cwt, much slower)? Default: '
                             'spec')

    args = parser.parse_args()

    return args


def main():
    args = define_arguments()

    from .spectrogram import calc_specgram_dual
    import obspy

    st = obspy.Stream()
    for file in args.data_files:
        st += obspy.read(file)
    st.merge(method=1, fill_value='interpolate')
    samp_rate_original = st[0].stats.sampling_rate
    if samp_rate_original > args.fmax * 10 and samp_rate_original % 5 == 0.:
        st.decimate(5)
    while st[0].stats.sampling_rate > 4. * args.fmax:
        st.decimate(2)

    if args.tstart is not None:
        t0 = obspy.UTCDateTime(args.tstart) - args.winlen*1.5
        t1 = obspy.UTCDateTime(args.tend) + args.winlen*1.5
        st.trim(t0, t1)
    
    if args.inventory_file is not None:
        inv = obspy.read_inventory(args.inventory_file)
        for tr in st:
            coords = inv.get_coordinates(tr.get_id())
            tr.stats.latitude = coords['latitude']
            tr.stats.longitude = coords['longitude']
            tr.stats.elevation = coords['elevation']
        st.remove_response(inventory=inv, output='ACC')
    else:
        if args.unit=='VEL':
            st.differentiate()
        if args.unit=='DIS':
            st.differentiate()
            st.differentiate()

    if args.catalog_file is not None:
        cat = obspy.read_events(args.catalog_file)
    else:
        cat = None

    # The computation of the LF spectrograms with long time windows or even CWT
    # can be REALLY slow, thus, decimate it to anything larger 2.5 Hz
    st_LF = st.copy()
    while st_LF[0].stats.sampling_rate > 4.:
        st_LF.decimate(2)

    #if not args.fnam:
    fnam = "spec_{network}.{station}.{location}.{channel}.png".format(**st_LF[0].stats)

    calc_specgram_dual(st_LF=st_LF,
                       st_HF=st.copy(),
                       fnam=fnam, show=args.interactive,
                       kind=args.kind,
                       fmax=args.fmax,
                       vmin=args.dBmin, vmax=args.dBmax,
                       tstart=args.tstart, tend=args.tend,
                       noise='Earth',
                       overlap=0.8,
                       w0=args.w0,
                       ratio_LF_spec=args.plot_ratio,
                       catalog=cat,
                       figsize=args.figsize,
                       winlen_sec_HF=args.winlen_HF,
                       winlen_sec_LF=args.winlen)


if __name__ == '__main__':
    main()
