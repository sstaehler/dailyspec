# dailyspec
Plot clean spectrograms of longer seismic time series


## Example:
This command 

    python -m dailyspec.plot_spec \
    --w0 24 -d ./S1022a/XB.ELYSE.02.BHZ \       
    --winlen 500  --fmax 10 --plot_ratio 0.6  --kind cwt \  
    --tstart 2022-05-04T22:00 --tend 2022-05-05T08:00 -h

will give you the spectrogram of the S1222a magnitude 5 marsquake, which was featured in NASA's press release on the event.
![spectrogram of a marsquake](https://mars.nasa.gov/system/news_items/main_images/9185_1-PIA25044-web.jpg)
https://mars.nasa.gov/news/9185/nasas-insight-records-monster-quake-on-mars/?site=insight

(the seismogram data will be released on October 1st 2022)
