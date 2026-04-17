from __future__ import annotations

import re
from pathlib import Path
from typing import BinaryIO, Final
from urllib.parse import ParseResult, urlparse, urlunparse

import requests
from lxml import etree
from lxml.etree import Element
from requests import Response

from ocr_util.corpus.common import MetsResource


class MetsFileObtainer:
    def __init__(
            self,
            urn: str,
            file_path: Path,
            url_urn_resolver: str = "https://nbn-resolving.org/",
            url_oai_pmh_data: str = "https://opendata.uni-halle.de/oai/dd",
            url_oai_pmh_id_prefix: str = "oai:opendata.uni-halle.de"
    ):
        self.__urn: Final[str] = urn
        self.__file_path: Final[Path] = file_path
        self.__url_urn_resolver: Final[str] = url_urn_resolver
        self.__url_oai_pmh_data: Final[str] = url_oai_pmh_data
        self.__url_oai_pmh_id_prefix: Final[str] = url_oai_pmh_id_prefix

    def run(self) -> MetsResource:
        if not self.__file_path.exists():
            open_data_url: str = self.__obtain_open_data_url_by_urn_resolver()
            oai_handle: str = re.search(r'/handle/(\d+/\d+)', open_data_url).group(1)
            identifier_oai_pmh: str = f'{self.__url_oai_pmh_id_prefix}:{oai_handle}'
            mets_content: bytes = self.__load_mets_file(identifier_oai_pmh)
            mets_file: BinaryIO
            with open(self.__file_path, "wb") as mets_file:
                mets_file.write(mets_content)
        return MetsResource(
            identifier_urn=self.__urn,
            file_path=self.__file_path
        )

    def __obtain_open_data_url_by_urn_resolver(self) -> str:
        response: Response = requests.get(
            url="https://nbn-resolving.org/"+self.__urn,
            timeout=30
        )
        if not response.ok:
            raise RuntimeError(f"Request Error - {response.status_code} for {response.url}")
        handlename = 'handle=[0-9]*\/[0-9]*'
        handlefinder = re.compile(handlename)
        url_open_data_ulb: str = "https://opendata.uni-halle.de/handle/"+handlefinder.search(response.url).group(0)[7:]
        url_parse_result: ParseResult = urlparse(url_open_data_ulb)
        url_open_data_ulb_without_params: str = urlunparse(
            (url_parse_result.scheme, url_parse_result.netloc, url_parse_result.path, "", "", "")
        )
        # print('url_open_data_ulb_without_params', url_open_data_ulb_without_params)
        return url_open_data_ulb_without_params

    def __load_mets_file(self, identifier: str) -> bytes:
        response: Response = requests.get(
            url=self.__url_oai_pmh_data,
            params={
                "identifier": identifier,
                "verb": "GetRecord",
                "metadataPrefix": "mets",
            },
            timeout=30
        )
        if not response.ok:
            raise RuntimeError(f"Request Error - {response.status_code} for {response.url}")
        return response.content
