#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

:copyright:
    Simon Stähler (mail@simonstaehler.com), 2019
    Savas Ceylan
:license:
    None
"""
from argparse import ArgumentParser

def main():
    args = define_arguments()

    return args


def define_arguments():
    helptext = 'Plot spectogram of data'
    parser = ArgumentParser(description=helptext)

    helptext = 'Channel name (SEED snippet, e.g. CH.ROTHE..HHZ)'
    parser.add_argument('-s', '--seed_id', help=helptext)
    parser.add_argument('--year', default=2022, type=int,
                        help='Start time for spectrogram' +
                             '(in any format that datetime understands)')
    parser.add_argument('--jday_start', default=1, type=int,
                        help='Start time for spectrogram' +
                             '(in any format that datetime understands)')
    parser.add_argument('--jday_end', default=366, type=int,
                        help='End time for spectrogram')
    args = parser.parse_args()

    return args

def produce_cycler_per_day(year, jday_start, jday_end, seed_ids,
        dir_image, dir_out):
    from os.path import join as pjoin
    import os.path as path
    from os import makedirs as mkdir

    mydir = path.dirname(path.abspath(__file__))
    with open(pjoin(mydir, 'data/string_head.txt'), 'r') as f:
        str_head = f.read()
    with open(pjoin(mydir, 'data/string_bottom.txt'), 'r') as f:
        str_bottom = f.read()
    for iday in range(jday_start, jday_end):
        str_all = str_head
        for seed_id in seed_ids:
            network, station, location, channel = seed_id.split('.')
            fnam = pjoin(dir_image, f'fig_{seed_id}',
                    f"spec_{network}.{station}.{location}.{channel}_{year}.{iday:03d}.png")
            str_fnam = f'<div class="mySlides img-magnifier-container">\n' \
                       f'<H1>{seed_id} {year}.{iday:03d}</H1>\n' \
                       f' <img id="myimage" src="{fnam}" style="width:100%">\n' \
                       f'</div>\n\n'
            str_all += str_fnam
        str_all += str_bottom
        dir_out_sol = pjoin(dir_out, f'{year:04d}_jday_{iday:03d}')
        mkdir(dir_out_sol, exist_ok=True)
        with open(pjoin(dir_out_sol, 'index.html'), 'w') as f:
            f.write(str_all)


def produce_cycler_per_channel(year, jday_start, jday_end, seed_id, fnam_out):
    import os.path as path

    mydir = path.dirname(path.abspath(__file__))
    with open(path.join(mydir, 'data/string_head.txt'), 'r') as f:
        str_head = f.read()
    with open(path.join(mydir, 'data/string_bottom.txt'), 'r') as f:
        str_bottom = f.read()
    str_all = str_head
    for iday in range(jday_start, jday_end):
        network, station, location, channel = seed_id.split('.')
        fnam = f"spec_{network}.{station}.{location}.{channel}_{year}.{iday:03d}.png"
        str_fnam = f'<div class="mySlides img-magnifier-container">\n' \
                   f'<H1>{seed_id} {year}.{iday:03d}</H1>\n' \
                   f' <img id="myimage" src="{fnam}" style="width:100%">\n' \
                   f'</div>\n\n'
        str_all += str_fnam
    str_all += str_bottom

    with open(fnam_out, 'w') as f:
        f.write(str_all)


def main():
    args = define_arguments()
    produce_cycler(args.year, args.jday_start, args.jday_end, args.seed_id,
                   fnam_out=f'overview_{args.seed_id}.html')


if __name__=='__main__':
    main()
