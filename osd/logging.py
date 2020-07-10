import logging
import sys

from osd.args import args


if args.DEBUG:
    FORMAT = '%(levelname).1s %(asctime)-15s '
    FORMAT += '%(filename)s:%(lineno)d %(message)s'
else:
    FORMAT = '%(levelname).1s %(asctime)-15s %(message)s'


logging.basicConfig(
    stream=sys.stdout,
    level=logging.DEBUG if args.DEBUG else logging.INFO,
    format=FORMAT,
    datefmt="%Y-%m-%d %H:%M:%S",
)

log = logging.getLogger(__name__)
