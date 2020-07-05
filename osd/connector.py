import urllib.request

from osd.logging import log


class ScoreboardConnector(object):
    """
    This Class will connect the recognition to the Dart-O-Mat 3000 (https://github.com/patrickhener/dart-o-mat-3000) using API requests
    """

    def __init__(self, dst_host:str, dst_port:int):
        """
        Initialize the connector with scoreboard API ip and port
        """
        self.dst_host = dst_host
        self.dst_port = dst_port
        self.base_url = f"http://{self.dst_host}:{self.dst_port}"

    def send_throw(self, number:int, mod:int):
        """
        This method will be used to send throws to the scoreboard.
        API URL is /ip:port/throw/number/mod
        """

        url = f"http://{self.base_url}/throw/{number}/{mod}"
        with urllib.request.urlopen(url) as response:
            log.debug(f"Request {url} sent")
            log.debug(f"Response status code from scoreboard is: {response.getgode()}")
            log.debug(f"Response message is: {response.parse().decode()}")

    def send_next(self):
        """
        This method will be used to send next player action to the scoreboard.
        API URL is /ip:port/nextPlayer
        """

        url = f"http://{self.base_url}/nextPlayer"
        with urllib.request.urlopen(url) as response:
            log.debug(f"Request {url} sent")
            log.debug(f"Response status code from scoreboard is: {response.getgode()}")
            log.debug(f"Response message is: {response.parse().decode()}")
