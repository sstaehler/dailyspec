#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
:license:
    None
"""

def define_arguments():
    from argparse import ArgumentParser
    helptext = 'Plot spectogram of data'
    parser = ArgumentParser(description=helptext)

    helptext = 'Data files'
    parser.add_argument('-d', '--directory', help=helptext)

    helptext = 'Inventory file'
    parser.add_argument('-i', '--inventory_file', help=helptext)

    helptext = 'Channel name (SEED snippet, e.g. CH.ROTHE..HHZ)'
    parser.add_argument('-s', '--seed_id', help=helptext)

    helptext = 'Text file with list of channel names (SEED snippet, e.g. CH.ROTHE..HHZ)'
    parser.add_argument('-l', '--list_seed_ids', help=helptext)

    helptext = 'Catalog file'
    parser.add_argument('-c', '--catalog_file', help=helptext, default=None)

    helptext = 'Catalog file'
    parser.add_argument('-k', '--kind', help=helptext, default='spec')

    helptext = 'Maximum frequency for plot'
    parser.add_argument('--fmax', default=50., type=float, help=helptext)

    helptext = 'Maximum frequency for plot'
    parser.add_argument('--winlen', default=50., type=float, help=helptext)

    parser.add_argument('-r', '--plot_ratio', type=float, default=0.5,
                        help='Ratio of height to use for the low '+
                             'frequency spectrogram')
    parser.add_argument('--dBmin', type=float, default=-180,
                        help='Minimum value for spectrograms (in dB)')
    parser.add_argument('--dBmax', type=float, default=-100,
                        help='Maximum value for spectrograms (in dB)')
    parser.add_argument('--year', default=2022, type=int,
                        help='Start time for spectrogram' +
                             '(in any format that datetime understands)')
    parser.add_argument('--jday_start', default=1, type=int,
                        help='First Julian day to process')
    parser.add_argument('--jday_end', default=366, type=int,
                        help='Last Julian day to process')
    args = parser.parse_args()

    return args


def main():
    args = define_arguments()
    if args.list_seed_ids is None and args.seed_id is None:
        raise ValueError('Either seed_id or list_seed_ids needs to be set')

    from .spectrogram import calc_specgram_dual
    from .produce_cycler import produce_cycler_per_channel, produce_cycler_per_day
    import obspy
    from os.path import join as pjoin
    from os.path import exists as pexists
    from os import makedirs as mkdir
    from tqdm import tqdm

    if args.list_seed_ids is not None:
        with open(args.list_seed_ids, 'r') as f:
            seed_ids = f.read().splitlines()
    else:
        seed_ids = [args.seed_id]

    for seed_id in seed_ids:
        dir_out = pjoin('by_channel', f'fig_{seed_id}')
        mkdir(dir_out, exist_ok=True)
        network, station, location, channel = seed_id.split('.')

        iday_start = 366
        iday_end = 0
        iday_end_all_channels = 0
        for iday in tqdm(range(args.jday_start, args.jday_end)):
            fnam_mseed = pjoin(args.directory, f'{args.year:4d}', network, station,
                               f'{channel}.D',
                               f'{network}.{station}.{location}.{channel}.D.{args.year:4d}.{iday:03d}')
            fnam_out = pjoin(dir_out,
                             f"spec_{network}.{station}.{location}.{channel}_{args.year}.{iday:03d}.png")

            if pexists(fnam_mseed) and not pexists(fnam_out):
                st = obspy.read(fnam_mseed)
                st.merge(method=1, fill_value='interpolate')

                inv = obspy.read_inventory(args.inventory_file)
                if args.catalog_file is not None:
                    cat = obspy.read_events(args.catalog_file)
                else:
                    cat = None

                if args.inventory_file is not None:
                    for tr in st:
                        coords = inv.get_coordinates(tr.get_id())
                        tr.stats.latitude = coords['latitude']
                        tr.stats.longitude = coords['longitude']
                        tr.stats.elevation = coords['elevation']

                    try:
                        st.remove_response(inventory=inv, output='ACC',
                                pre_filt=(1./args.winlen * 0.5,
                                          1./args.winlen,
                                          args.fmax*0.95,
                                          args.fmax))
                    except ValueError:
                        print(f'Problem with response for station {tr.stats.station}')
                else:
                    st.differentiate()

                # samp_rate_original = st[0].stats.sampling_rate
                # if samp_rate_original > args.fmax * 10 and samp_rate_original % 5 == 0.:
                #     st.decimate(5)
                # while st[0].stats.sampling_rate > 4. * args.fmax:
                #     st.decimate(2)
                st.interpolate(sampling_rate=args.fmax * 2.,
                               method='nearest')
                # The computation of the LF spectrograms with long time windows or even CWT
                # can be REALLY slow, thus, decimate it to anything larger 2.5 Hz
                st_LF = st.copy()
                st_LF.filter('lowpass', freq=0.95, corners=16)
                # while st_LF[0].stats.sampling_rate > 4.:
                #     # print('LF samp rate ', st_LF[0].stats.sampling_rate, ' decimating')
                #     st_LF.decimate(2)
                st_LF.interpolate(sampling_rate=2.,
                               method='nearest')

                tstart = float(obspy.UTCDateTime(f'{args.year:4d}0101T00:00:00Z')) \
                         + 86400. * (iday - 1) - 20.
                tend = float(obspy.UTCDateTime(f'{args.year:4d}0101T00:00:00Z')) \
                       + 86400. * iday + 20.
                calc_specgram_dual(st_LF=st_LF,
                                   st_HF=st.copy(),
                                   fnam=fnam_out,
                                   kind=args.kind,
                                   fmax=args.fmax,
                                   tstart=tstart, tend=tend,
                                   vmin=args.dBmin, vmax=args.dBmax,
                                   noise='Earth',
                                   overlap=0.8,
                                   ratio_LF_spec=args.plot_ratio,
                                   catalog=cat,
                                   winlen_sec_HF=4,
                                   winlen_sec_LF=args.winlen)
            if pexists(fnam_out):
                iday_start = min(iday, iday_start)
                iday_end = max(iday, iday_end)

        produce_cycler_per_channel(year=args.year,
                                   jday_start=iday_start,
                                   jday_end=iday_end,
                                   seed_id=seed_id,
                                   fnam_out=pjoin(dir_out, 'overview.html'))
        iday_end_all_channels = max(iday_end_all_channels, iday_end)

    produce_cycler_per_day(year=args.year,
                           jday_start=iday_start,
                           jday_end=iday_end_all_channels,
                           seed_ids=seed_ids,
                           dir_image='../../by_channel',
                           dir_out='./by_day')

if __name__ == '__main__':
    main()
