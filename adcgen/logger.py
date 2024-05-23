"""
Basic logging functionality that enables adaptive on/off for debug
messages.
"""

import os

def log(*msg):
    if os.environ.get("ADCGEN_VERBOSE", 1) != "0":
        print(*msg)
