import argparse
from osd._version import __version__

parser = argparse.ArgumentParser(
    description="OpenCV based steel darts recognition"
)

scoreboard_group = parser.add_argument_group(description="Give details where to reach scoreboard host listening for API requests")

scoreboard_group.add_argument(
    dest="SB_HOST", type=str, default="127.0.0.1", help=("scoreboard host to send api calls to (default: %(default)s)")
)

scoreboard_group.add_argument(
    dest="SB_PORT", type=int, default=5000, help=("scoreboard port to send api calls to (default: %(default)s)")
)

parser.add_argument(
    '-d', '--debug', dest="DEBUG", default=False, action="store_true", help=("show debug messages")
)

parser.add_argument(
    '-v', '--version', action='version', version=f'OpenCV steel darts {__version__}'
)

args = parser.parse_args()