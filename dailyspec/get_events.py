import obspy
from obspy.clients.fdsn import Client

def define_arguments():
    from argparse import ArgumentParser
    helptext = 'Get Events around center location, with staggered minimum magnitude'
    parser = ArgumentParser(description=helptext)

    helptext = 'Inventory file'
    parser.add_argument('-i', '--inventory_file', default=None, help=helptext)

    helptext = 'Channel name for station (SEED snippet, e.g. CH.ROTHE..HHZ)'
    parser.add_argument('-s', '--seed_id', help=helptext)

    helptext = 'Station coordinates (lat, lon)'
    parser.add_argument('--coords', nargs=2, type=float, help=helptext, default=None)

    helptext = 'FDSN Client (default: IRIS)'
    parser.add_argument('-c', '--client', help=helptext, default='IRIS')

    helptext = 'Starttime'
    parser.add_argument('--tstart', help=helptext, required=True)

    helptext = 'Endtime (Default: Now)'
    parser.add_argument('--tend', help=helptext, default=None, type=str)

    helptext = 'Distance bins (maximum distance in degree)'
    parser.add_argument('--dists', help=helptext, nargs='+', type=float,
                        default=(5., 30., 180.))

    helptext = 'Minimum magnitude for each distance bin'
    parser.add_argument('--min_mags', help=helptext, nargs='+', type=float,
                        default=(2., 4., 6.5))

    return parser.parse_args()


def main():
    args = define_arguments()
    if args.coords:
        lat = args.coords[0]
        lon = args.coords[1]
    else:
        inv = obspy.read_inventory(args.inventory_file)
        coords = inv.get_coordinates(args.seed_id)
        lat = coords['latitude']
        lon = coords['longitude']

    # Get station coordinates
    client = Client(args.client)
    t0 = obspy.UTCDateTime(args.tstart)
    if args.tend is None:
        t1 = None
    else:
        t1 = obspy.UTCDateTime(args.tend)
    print(f'Download all events > {args.min_mags[0]:3.1f}, closer {args.dists[0]:.1f} deg')
    cat = client.get_events(starttime=t0,
                            endtime=t1,
                            minmagnitude=args.min_mags[0],
                            latitude=lat, longitude=lon,
                            maxradius=args.dists[0])

    for ibin in range(1, len(args.dists)):
        print(f'Download all events > {args.min_mags[ibin]:3.1f}, closer {args.dists[ibin]:.1f} deg')
        cat += client.get_events(starttime=t0,
                                 endtime=t1,
                                 minmagnitude=args.min_mags[ibin],
                                 latitude=lat, longitude=lon,
                                 maxradius=args.dists[ibin])

    cat.write('events.xml', format='QUAKEML')
    fig = cat.plot(show=False, resolution='i', label=None, color='date')
    fig.savefig('events.png', dpi=300)


if __name__ == '__main__':
    main()
