# from typing import Final, Dict, Literal, Union, NamedTuple, List

import typing


# ISO 639-1 https://de.wikipedia.org/wiki/Liste_der_ISO-639-1-Codes
class LanguageMapItem(typing.NamedTuple):
    iso_639_2B: str
    iso_639_1: str
    lt_variant: str


# https://en.wikipedia.org/wiki/Category:Dialects_of_languages_with_ISO_639-3_code
LANGUAGE_MAP: typing.Dict[str, LanguageMapItem] = {
    'ger': LanguageMapItem(
        iso_639_2B='ger',
        iso_639_1='de',
        lt_variant='de-de',
    ),
    'ara': LanguageMapItem(
        iso_639_2B='ara',
        iso_639_1='ar',
        lt_variant='ar',
    ),
    'dut': LanguageMapItem(
        iso_639_2B='dut',
        iso_639_1='nl',
        lt_variant='nl',
    ),
    'eng': LanguageMapItem(
        iso_639_2B='eng',
        iso_639_1='en',
        lt_variant='en-gb',
    ),
    'fre': LanguageMapItem(
        iso_639_2B='fre',
        iso_639_1='fr',
        lt_variant='fr',
    ),
    'gre': LanguageMapItem(
        iso_639_2B='gre',
        iso_639_1='el',
        lt_variant='el',
    ),
    # old greek
    'grc': LanguageMapItem(
        iso_639_2B='grc',
        iso_639_1='el',
        lt_variant='el',
    ),
    'ita': LanguageMapItem(
        iso_639_2B='ita',
        iso_639_1='it',
        lt_variant='it',
    ),
    'pol': LanguageMapItem(
        iso_639_2B='pol',
        iso_639_1='pl',
        lt_variant='pl-pl',
    ),
    'spa': LanguageMapItem(
        iso_639_2B='spa',
        iso_639_1='es',
        lt_variant='es',
    ),
    'swe': LanguageMapItem(
        iso_639_2B='swe',
        iso_639_1='sv',
        lt_variant='sv',
    ),
    # 'per', premium only
    # 'heb', n.a.
    # 'cze', n.a.
    # 'hun', n.a.
    # 'ine', ???
    # 'lat', n.a.
    # 'lav', n.a.
    # 'nds', ???
    # 'urd', n.a.
    # 'yid', n.a.
}

LANGUAGE_KEYS: typing.List[str] = list(LANGUAGE_MAP.keys())
LANGUAGE_KEY_DEFAULT: str = LANGUAGE_KEYS[0]
