#!/usr/bin/env python

"""
# Author: Ying Wu
# File Name: __init__.py
# Description:
"""

__author__ = "Ying Wu"
__email__ = "wuy@big.ac.cn"
__name__ = "STASCAN"

#import run_STASCAN
#import preparation
#import model
#import stat
#import downstream

#from STASCAN import run_STASCAN
#from STASCAN import preparation
#from STASCAN import model
#from STASCAN import StatPlot
#from STASCAN import downstream

import STASCAN.run_STASCAN as run_STASCAN
import STASCAN.preparation as preparation
import STASCAN.model as model
import STASCAN.StatPlot as StatPlot
import STASCAN.downstream as downstream
import STASCAN.run_STASCAN_dbit as run_STASCAN_dbit
import STASCAN.preparation_dbit as preparation_dbit