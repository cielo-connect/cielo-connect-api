"""Data classes for Cielo."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class CieloData:
    """Container for raw response and parsed device map keyed by MAC."""

    raw: Mapping[str, Any] | None
    parsed: dict[str, CieloDevice] | None


@dataclass(slots=True)
class CieloDevice:
    """Normalized view of a Cielo device."""

    id: str
    mac_address: str
    name: str

    ac_states: Mapping[str, Any]
    device_status: bool | None
    temp: float | None
    humidity: int | None
    target_temp: float | None
    target_heat_set_point: float | None
    target_cool_set_point: float | None
    hvac_mode: str | None
    fan_mode: str | None
    swing_mode: str | None
    preset_mode: int | None
    device_on: bool | None

    is_thermostat: bool | None
    appliance_id: int | None
    hvac_modes: list[str] | None
    fan_modes: list[str] | None
    fan_modes_translated: dict[str, str] | None
    swing_modes: list[str] | None
    swing_modes_translated: dict[str, str] | None
    temp_list: list[int]
    preset_modes: list[str] | None
    temp_unit: str | None
    temp_step: int | None
    supported_features: dict[str, str] | None
