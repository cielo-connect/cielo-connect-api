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

    def apply_update(self, data: Mapping[str, Any]) -> None:
        """Apply an API response payload to update device state attributes."""

        self.ac_states.update(data)

        if (temp := data.get("set_point")) is not None:
            self.target_temp = float(temp)
        if (mode := data.get("mode")) is not None:
            self.hvac_mode = mode
        if (fan_mode := data.get("fan_speed")) is not None:
            self.fan_mode = fan_mode
        if (preset := data.get("preset")) is not None:
            self.preset_mode = preset
        if (swing_mode := data.get("swing_position")) is not None:
            self.swing_mode = swing_mode

        device_power = data.get("power")
        if device_power is not None:
            self.device_on = device_power.lower() == "on"
            if not self.device_on and self.hvac_mode != "off":
                self.hvac_mode = "off"

        if (heat_temp := data.get("heat_set_point")) is not None:
            self.target_heat_set_point = float(heat_temp)
        if (cool_temp := data.get("cool_set_point")) is not None:
            self.target_cool_set_point = float(cool_temp)
