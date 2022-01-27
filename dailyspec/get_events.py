import obspy
from obspy import UTCDateTime as utct
from obspy.clients.fdsn import Client

client = Client("ETH")
t0 = obspy.UTCDateTime('2022-01-01T00:00:00Z')
cat = client.get_events(starttime=t0, minmagnitude=2.0)

client = Client("IRIS")
t0 = obspy.UTCDateTime('2022-01-01T00:00:00Z')

cat += client.get_events(starttime=t0, minmagnitude=4.5)

cat.write('events.xml', format='QUAKEML')

