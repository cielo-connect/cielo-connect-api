"""Exceptions for Cielo."""

from typing import Any


class CieloError(Exception):
    """Generic Cielo API error."""

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)


class AuthenticationError(CieloError):
    """Authentication / authorization error with the Cielo API."""

    def __init__(self, *args: Any) -> None:
        super().__init__(*args)
