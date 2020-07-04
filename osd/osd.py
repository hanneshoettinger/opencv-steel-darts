#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import threading
import signal

from osd.logging import log

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
    print("Test")