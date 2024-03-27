from __future__ import annotations

import sys

if sys.version_info >= (3, 11):
    from enum import StrEnum
else:
    from strenum import StrEnum


class CountMethod(StrEnum):
    exact = "exact"
    planned = "planned"
    estimated = "estimated"


class Filters(StrEnum):
    NOT = "not"
    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IS = "is"
    LIKE = "like"
    LIKE_ALL = "like(all)"
    LIKE_ANY = "like(any)"
    ILIKE = "ilike"
    ILIKE_ALL = "ilike(all)"
    ILIKE_ANY = "ilike(any)"
    FTS = "fts"
    PLFTS = "plfts"
    PHFTS = "phfts"
    WFTS = "wfts"
    IN = "in"
    CS = "cs"
    CD = "cd"
    OV = "ov"
    SL = "sl"
    SR = "sr"
    NXL = "nxl"
    NXR = "nxr"
    ADJ = "adj"


class RequestMethod(StrEnum):
    GET = "GET"
    POST = "POST"
    PATCH = "PATCH"
    PUT = "PUT"
    DELETE = "DELETE"
    HEAD = "HEAD"


class ReturnMethod(StrEnum):
    minimal = "minimal"
    representation = "representation"
