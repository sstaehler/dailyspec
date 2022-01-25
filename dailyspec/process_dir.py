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
    parser.add_argument('-d', '--directory', nargs='+', help=helptext)

    helptext = 'Inventory file'
    parser.add_argument('-i', '--inventory_file', help=helptext)

    helptext = 'Channel name (SEED snippet, e.g. CH.ROTHE..HHZ)'
    parser.add_argument('-s', '--seed_id', help=helptext)

    helptext = 'Catalog file'
    parser.add_argument('-c', '--catalog_file', help=helptext, default=None)

    helptext = 'Maximum frequency for plot'
    parser.add_argument('--fmax', default=50., type=float, help=helptext)

    parser.add_argument('--dBmin', type=float, default=-180,
                        help='Minimum value for spectrograms (in dB)')
    parser.add_argument('--dBmax', type=float, default=-100,
                        help='Maximum value for spectrograms (in dB)')
    parser.add_argument('--year', default=2022, type=int,
                        help='Start time for spectrogram' +
                             '(in any format that datetime understands)')
    parser.add_argument('--jday_start', default=1, type=int,
                        help='Start time for spectrogram' +
                             '(in any format that datetime understands)')
    parser.add_argument('--jday_end', default=366, type=int,
                        help='End time for spectrogram')
    parser.add_argument('--plot_ratio', default=0.6, type=float,
                        help='Window length for long-period spectrogram (in '
                             'seconds)')
    parser.add_argument('--winlen', default=100., type=float,
                        help='Window length for long-period spectrogram (in '
                             'seconds)')
    parser.add_argument('--no_noise', default=False, action='store_true',
                        help='Omit plotting the NLNM/NHNM curves')
    parser.add_argument('--interactive', default=False,
                        action='store_true',
                        help='Open a matplotlib window instead of saving to '
                             'disk')

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
    from os.path import join as pjoin
    from tqdm import tqdm

    network, station, location, channel = args.seed_id.split('.')

    for iday in tqdm(range(args.jday_start, args.jday_end)):
        fnam = pjoin(network, station, f'{channel}.D',
                     f'{network}.{station}.{location}.{channel}.D.{args.year:4d}.{iday:03d}')

        st = obspy.read(fnam)
        st.merge(method=1, fill_value='interpolate')
        samp_rate_original = st[0].stats.sampling_rate
        if samp_rate_original > args.fmax * 10 and samp_rate_original % 5 == 0.:
            st.decimate(5)
        while st[0].stats.sampling_rate > 4. * args.fmax:
            st.decimate(2)

        inv = obspy.read_inventory(args.inventory_file)
        if args.catalog_file is not None:
            cat = obspy.read_events(args.catalog_file)
        else:
            cat = None

        for tr in st:
            coords = inv.get_coordinates(tr.get_id())
            tr.stats.latitude = coords['latitude']
            tr.stats.longitude = coords['longitude']
            tr.stats.elevation = coords['elevation']

        st.remove_response(inventory=inv, output='ACC')

        # The computation of the LF spectrograms with long time windows or even CWT
        # can be REALLY slow, thus, decimate it to anything larger 2.5 Hz
        st_LF = st.copy()
        while st_LF[0].stats.sampling_rate > 4.:
            # print('LF samp rate ', st_LF[0].stats.sampling_rate, ' decimating')
            st_LF.decimate(2)

        fnam = f"spec_{network}.{station}.{location}.{channel}_{args.year}.{iday}.png"
        calc_specgram_dual(st_LF=st_LF,
                           st_HF=st.copy(),
                           fnam=fnam, show=args.interactive,
                           kind=args.kind,
                           fmax=args.fmax,
                           vmin=args.dBmin, vmax=args.dBmax,
                           tstart=args.tstart, tend=args.tend,
                           noise='Earth',
                           overlap=0.8,
                           ratio_LF_spec=args.plot_ratio,
                           catalog=cat,
                           winlen_sec_HF=4,
                           winlen_sec_LF=args.winlen)


if __name__ == '__main__':
    main()
