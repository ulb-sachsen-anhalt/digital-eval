from typing import Final, Tuple, List

from requests import Response


class LanguageToolException(Exception):
    pass


class InvalidResponseException(LanguageToolException):
    def __init__(self, url: str, response: Response):
        super().__init__(f"'{url}' returned invalid '{response}!'")


class ConnectionException(LanguageToolException):
    def __init__(self, url: str):
        super().__init__(f"Unable to connect to {url}")


class DefaultServiceNotAvailable(LanguageToolException):
    pass


class NoFreePortAvailableException(LanguageToolException):
    pass


class NoApiPortFoundException(LanguageToolException):
    pass


class ContainerException(LanguageToolException):
    pass


class NotInizializedException(LanguageToolException):
    def __init__(self):
        super().__init__("LanguageTool is not initialized")


class Constant:
    ENDPOINT: Final[str] = "/v2/check"
    LOCAL_HOST: Final[str] = "localhost"
    DEFAULT_PORT: Final[int] = 8010
    DEFAULT_PROTOCOL: Final[str] = "http://"
    # use "safe" port range https://utho.com/docs/tutorial/most-common-network-port-numbers-for-linux/
    PORT_RANGE: Final[Tuple[int, int]] = (49151, 65535)
    EXCLUDED_PORTS: Final[List[int]] = []
    DOCKER_IMAGE: Final[str] = "silviof/docker-languagetool"
