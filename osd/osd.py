#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import signal

from osd.logging import log

# Debugging
from osd.draw import Draw
# End Debugging

def signal_handler(sig, frame):
    log.info("CTRL-C caught, exiting...")


def start_thread(f, *args):
    threading.Thread(
        thread=f,
        args=(*args,),
        daemon=True,
    ).start()


def main(fully_threaded=False):
    signal.signal(signal.SIGINT, signal_handler)
    # Debugging
    # Draw here to test board
    # But first TODO will be calibration for havig calibration data

    # End Debugging


