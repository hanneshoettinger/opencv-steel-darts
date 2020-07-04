import argparse
from osd._version import __version__

parser = argparse.ArgumentParser(
    description="OpenCV based steel darts recognition"
)

parser.add_argument(
    '-m', '--mode', dest="MODE", type=str, default="run", help=("first [calibrate] then [run] (default: %(default)s")
)

parser.add_argument(
    '-d', '--debug', dest="DEBUG", default=False, action="store_true", help=("show debug messages")
)

parser.add_argument(
    '-v', '--version', action='version', version=f'OpenCV steel darts {__version__}'
)

args = parser.parse_args()