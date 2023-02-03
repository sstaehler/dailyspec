# -*- coding: utf-8 -*-
"""

:copyright:
    2020 - 2021, Simon Stähler
:license:
    GNU General Public License, Version 3
    (https://www.gnu.org/licenses/gpl-3.0.html)

"""

import logging

# from .spectrogram import calc_specgram_dual
# Set matplotlib logging level and backend
logging.getLogger("matplotlib").setLevel(logging.INFO)
# try:
#     os.environ["DISPLAY"]
#     mpl_use("Qt5Agg")
# except KeyError:
# mpl_use("Agg")

name = "dailyspec"
__version__ = "0.2.0"
__description__ = """Tool to efficiently plot spectrograms of longer seismic datasets."""
__license__ = "GPLv3"
__author__ = "Simon Stähler"
__email__ = """mail@simonstaehler.com"""
