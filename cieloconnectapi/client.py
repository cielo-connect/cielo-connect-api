"""Async Python API client for Cielo Home."""

from __future__ import annotations

import asyncio
from bisect import bisect_left
from collections.abc import Mapping
import logging
from typing import Any

from aiohttp import ClientResponse, ClientSession, ClientTimeout

from .const import *
from .exceptions import AuthenticationError, CieloError
from .model import CieloData, CieloDevice

__version__ = "0.1.0"

BASE_URL = "https://api.smartcielo.com/openapi/v1"
DEFAULT_TIMEOUT = 5 * 60  # 5 minutes
AUTH_ERROR_CODES = {401, 403}

_LOGGER = logging.getLogger(__name__)


class CieloClient:
    """Asynchronous client for the Cielo Home API.

    Usage:
        async with CieloClient(api_key) as client:
            data = await client.get_devices_data()
    """

    def __init__(
        self,
        api_key: str,
        *,
        session: ClientSession | None = None,
        timeout: int = DEFAULT_TIMEOUT,
        token: str | None = None,
        max_retries: int = 2,
    ) -> None:
        self.api_key = api_key
        self.token = token
        self.device_data = None
        self.cached_supported_features = {}
        self._owned_session = session is None
        self._session: ClientSession = session or ClientSession(
            timeout=ClientTimeout(total=timeout)
        )
        self._timeout = ClientTimeout(total=timeout)
        self._max_retries = max(0, int(max_retries))

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    async def __aenter__(self) -> CieloClient:
        if self._session.closed:
            self._session = ClientSession(timeout=self._timeout)
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def close(self) -> None:
        """Close the session if it was created by this client."""
        if self._owned_session and not self._session.closed:
            await self._session.close()

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------
    async def get_or_refresh_token(self, force_refresh: bool = False) -> str:
        """Ensure an access token is available, refreshing if needed."""
        if force_refresh or not self.token:
            await self._login()
        assert self.token
        return self.token

    async def _login(self) -> None:
        """Authenticate"""
        headers = {"x-api-key": self.api_key}

        result = await self._post(
            f"{BASE_URL}/authenticate",
            json_data=None,
            headers=headers,
            auth_ok=False,
        )
        try:
            self.token = result["data"]["access_token"]
        except (KeyError, TypeError) as exc:
            raise AuthenticationError("Invalid authentication response format") from exc
        _LOGGER.debug("Authentication succeeded")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def get_devices_data(self) -> CieloData:
        """Fetch and parse all devices into normalized dataclasses."""
        await self.get_or_refresh_token()
        cached_appliances = self.get_cached_appliance_ids()
        response = await self._get(
            f"{BASE_URL}/devices/?cached_appliance_ids={cached_appliances}",
            headers=self._auth_headers(),
        )
        devices_payload = (response or {}).get("data", {})

        parsed: dict[str, CieloDevice] = {}

        if isinstance(devices_payload, dict):
            for _k, devices in devices_payload.items():
                if isinstance(devices, list):
                    for d in devices:
                        self._add_device(parsed, d)
                elif isinstance(devices, dict):
                    for v in devices.values():
                        if isinstance(v, list):
                            for d in v:
                                self._add_device(parsed, d)
        elif isinstance(devices_payload, list):
            for d in devices_payload:
                self._add_device(parsed, d)

        return CieloData(raw=response, parsed=parsed)

    async def set_ac_state(
        self, mac_address: str, action_type: str, actions: dict
    ) -> Mapping[str, Any]:
        """Send a control command to a specific AC unit."""
        await self.get_or_refresh_token()
        payload = {
            "mac_address": mac_address,
            "action_type": action_type,
            "actions": actions,
        }
        return await self._post(
            f"{BASE_URL}/action", json_data=payload, headers=self._auth_headers()
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _auth_headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key, "Authorization": self.token or ""}

    def _add_device(
        self, parsed: dict[str, CieloDevice], device: Mapping[str, Any]
    ) -> None:
        dev = self._parse_device(device)
        try:
            # dev = self._parse_device(device)
            parsed[dev.mac_address] = dev
        except Exception as exc:
            _LOGGER.debug("Skipping device due to parse error: %s", exc)

    def _parse_device(self, device: Mapping[str, Any]) -> CieloDevice:
        mac_address = device["mac_address"]
        sensor_readings = device.get("sensor_readings") or {}
        temp_unit = device["temperature_unit"]
        appliance_id = device["appliance_id"]
        supported_features = device["supported_features"]

        if appliance_id:
            if appliance_id not in self.cached_supported_features:  # not cached yet
                self.cached_supported_features.update(
                    {appliance_id: supported_features}
                )

            cache = self.cached_supported_features[appliance_id]

            # Modes
            cache["modes"] = supported_features.get("modes") or cache.get("modes")
            supported_features["modes"] = cache["modes"]

            # Presets
            cache["presets"] = supported_features.get("presets") or cache.get("presets")
            supported_features["presets"] = cache["presets"]

        is_thermostat = device["device_type"] == "Thermostat"
        ac_state = device["current_state"]
        target_temp = ac_state["set_point"]
        hvac_mode = ac_state["mode"]
        replace_mode = "heat" if hvac_mode == "aux" else hvac_mode
        supported_fans = supported_features["modes"][replace_mode]["fan_levels"]
        supported_swings = supported_features["modes"][replace_mode]["swing"]
        temp_range = supported_features["modes"][replace_mode]["temperatures"][
            temp_unit
        ]["values"]

        preset_modes = [
            preset["title"].lower()
            if preset["title"] in AVAILABLE_PRESETS_MODES
            else preset["title"]
            for preset in supported_features["presets"]
        ]

        return CieloDevice(
            id=mac_address,
            mac_address=mac_address,
            name=device["device_name"],
            ac_states=ac_state,
            appliance_id=appliance_id,
            device_status=device["connection_status"]["is_alive"],
            temp=sensor_readings["temperature"],
            humidity=sensor_readings["humidity"],
            target_temp=target_temp,
            target_heat_set_point=ac_state["heat_set_point"],
            target_cool_set_point=ac_state["cool_set_point"],
            hvac_mode=hvac_mode,
            device_on=str(ac_state.get("power") or "").lower() != "off",
            fan_mode=ac_state.get("fan_speed") or "",
            swing_mode=ac_state.get("swing_position") or "",
            hvac_modes=list(supported_features["modes"].keys()),
            fan_modes=supported_fans or None,
            fan_modes_translated=None
            if is_thermostat
            else {f.lower(): str(f) for f in supported_fans},
            swing_modes=supported_swings or None,
            swing_modes_translated=None
            if is_thermostat
            else {s.lower(): str(s) for s in supported_swings},
            temp_list=temp_range,
            preset_modes=preset_modes,
            preset_mode=ac_state["preset"],
            temp_unit=temp_unit,
            temp_step=device.get("temperature_increment", 1),
            is_thermostat=is_thermostat,
            supported_features=supported_features,
        )

    def get_cached_appliance_ids(self):
        return (
            "[" + ",".join(str(k) for k in self.cached_supported_features.keys()) + "]"
        )

    # ------------------------------------------------------------------
    # HTTP core
    # ------------------------------------------------------------------
    async def _request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
        json_data: Mapping[str, Any] | None = None,
        retries: int | None = None,
        auth_ok: bool = True,
    ) -> dict[str, Any]:
        attempts = 0
        max_retries = self._max_retries if retries is None else max(0, int(retries))

        while True:
            attempts += 1
            try:
                async with self._session.request(
                    method,
                    url,
                    headers=headers,
                    params=dict(params) if params else None,
                    json=dict(json_data) if json_data else None,
                    timeout=self._timeout,
                ) as resp:
                    if resp.status in AUTH_ERROR_CODES and auth_ok:
                        _LOGGER.debug(
                            "Auth failed (%s). Refreshing token…", resp.status
                        )
                        await self.get_or_refresh_token(force_refresh=True)
                        if headers and "Authorization" in headers:
                            headers = dict(headers)
                            headers["Authorization"] = self.token or ""
                        continue

                    return await self._handle_response(resp)

            except AuthenticationError:
                raise
            except Exception as exc:

                def _exp_backoff(attempt: int) -> float:
                    import random

                    base = min(8.0, 0.5 * (2**attempt))
                    return base + random.uniform(0.0, 0.25 * base)

                if attempts <= max_retries + 1:
                    delay = _exp_backoff(attempts - 1)
                    _LOGGER.warning(
                        "Request failed (%s %s): %s. Retrying in %.2fs",
                        method,
                        url,
                        exc,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise CieloError(
                    f"Request failed after {attempts - 1} retries: {exc}"
                ) from exc

    async def _handle_response(self, resp: ClientResponse) -> dict[str, Any]:
        if resp.status in AUTH_ERROR_CODES:
            raise AuthenticationError(f"Authentication failed (HTTP {resp.status})")
        if resp.status != 200:
            text = await resp.text()
            raise CieloError(f"HTTP {resp.status}: {text}")

        try:
            return await resp.json()
        except Exception as exc:
            raise CieloError(f"Invalid JSON response: {exc}") from exc

    async def _get(self, url: str, **kwargs) -> dict[str, Any]:
        return await self._request("GET", url, **kwargs)

    async def _post(self, url: str, **kwargs) -> dict[str, Any]:
        return await self._request("POST", url, **kwargs)

    #### HELPERS ####
    def _mode_caps(self) -> dict:
        """Return vendor caps for the current vendor mode string."""
        if self.device_data is None:
            return {}

        mode_key = self.device_data.hvac_mode or HVACMode.OFF
        return (self.device_data.supported_features.get("modes") or {}).get(
            mode_key, {}
        ) or {}

    def _mode_supports_temperature(self) -> bool:
        caps = self._mode_caps()
        if "rules" in caps and caps["rules"].split(":")[0] == "vanish":
            return False
        temps = (caps.get("temperatures") or {}).get(self.device_data.temp_unit, {})
        values = temps.get("values") or []
        # Consider it supported only if there are actual selectable values.
        return len(values) > 0

    def _current_mode_temp_values(self) -> list[int]:
        caps = self._mode_caps()
        temps = (caps.get("temperatures") or {}).get(self.device_data.temp_unit, {})
        return list(temps.get("values") or [])

    def _find_valid_target_temp(self, target: float, valid: list[int]) -> int:
        if not valid:
            return int(round(float(target)))
        target = int(round(float(target)))
        if target <= valid[0]:
            return valid[0]
        if target >= valid[-1]:
            return valid[-1]

        return valid[bisect_left(valid, target)]

    def _supports_half_step(self, ha_unit="°C") -> bool:
        """Return True if we should use 0.5°C resolution."""
        if not getattr(self.device_data, "is_thermostat", False):
            return False
        if self.device_data.temp_unit != "C":
            return False

        return ha_unit == UnitOfTemperature.CELSIUS

    def _round_to_half(self, value: float) -> float:
        return round(value * 2) / 2.0

    def available(self) -> bool:
        if self.device_data is None:
            return False
        return bool(self.device_data.device_status)

    def current_mode_fan_speed(self) -> list[str] | None:
        caps = self._mode_caps()
        modes = caps.get("fan_levels") or []
        # your API sometimes wraps this in nested arrays; normalize to flat list
        if modes and isinstance(modes[0], list):
            modes = modes[0]
        return list(modes) or None

    def fan_mode(self) -> str | None:
        available = self.fan_modes()
        if not available:
            return None  # truly no fan control in this mode

        # Try to derive current from either stored fan_mode or ac_states
        cur = (
            self.device_data.fan_mode
            or self.device_data.ac_states.get("fan_speed")
            or ""
        )

        if cur not in available:
            # Repair to a valid option (prefer 'auto' if present)
            cur = "auto" if "auto" in available else available[0]
            self.device_data.fan_mode = cur
            self.device_data.ac_states["fan_speed"] = cur

        return cur

    def fan_modes(self) -> list[str] | None:
        caps = self._mode_caps()
        modes = caps.get("fan_levels") or []
        if modes and isinstance(modes[0], list):
            modes = modes[0]
        return list(modes) or []

    def swing_modes(self) -> list[str] | None:
        caps = self._mode_caps()
        modes = caps.get("swing") or []
        if modes and isinstance(modes[0], list):
            modes = modes[0]
        return list(modes) or None

    def preset_mode(self) -> str | None:
        idx = self.device_data.preset_mode
        modes = self.device_data.preset_modes
        if not modes:
            return None
        # cloud uses numeric slots; 0 means none
        if str(idx) == "0" or idx is None:
            return None
        try:
            return modes[int(idx) - 1]
        except Exception:
            return None

    def preset_modes(self) -> list[str] | None:
        return (
            list(self.device_data.preset_modes)
            if self.device_data.preset_modes
            else None
        )

    def hvac_mode(self) -> HVACMode | None:
        dev = self.device_data
        if not dev.device_on:
            return HVACMode.OFF
        if dev.hvac_mode is None:
            return HVACMode.OFF
        if dev.hvac_mode == "aux":
            return HVACMode.HEAT
        return CIELO_TO_HA.get(dev.hvac_mode, HVACMode.OFF)

    def hvac_modes(self) -> list[HVACMode]:
        dev = self.device_data
        modes = dev.hvac_modes or []
        modes_list: list[HVACMode] = []
        for mode in modes:
            if mode == "aux":
                continue
            modes_list.append(CIELO_TO_HA.get(mode, HVACMode.OFF))
        # Always include OFF
        if HVACMode.OFF not in modes_list:
            modes_list.append(HVACMode.OFF)
        return modes_list

    def precision(self, ha_unit) -> float:
        """Return the precision of the thermostat."""
        if self._supports_half_step(ha_unit):
            return 0.5
        return 1.0

    def temperature_unit(self) -> str:
        return (
            UnitOfTemperature.FAHRENHEIT
            if self.device_data.temp_unit == "F"
            else UnitOfTemperature.CELSIUS
        )

    def current_temperature(self) -> float | None:
        if self.device_data.temp is None:
            return None

        return float(self.device_data.temp)

    def target_temperature_step(self, ha_unit) -> float | None:
        # Thermostats in Celsius: allow 0.5 steps
        if self._supports_half_step(ha_unit):
            return 0.5

        # Non-thermostat or non-Celsius: keep 1 degree
        return 1.0

    def target_temperature(self) -> float | None:
        # Only expose a target temp if supported right now
        if not self._mode_supports_temperature():
            return None
        t = self.device_data.target_temp
        return float(t) if t is not None else None

    def target_temperature_low(self, ha_unit):
        val = self.device_data.target_heat_set_point
        if val is None:
            return None
        if self._supports_half_step(ha_unit):
            return self._round_to_half(float(val))
        return int(float(val))

    def target_temperature_high(self, ha_unit):
        val = self.device_data.target_cool_set_point
        if val is None:
            return None
        if self._supports_half_step(ha_unit):
            return self._round_to_half(float(val))
        return int(float(val))

    def min_temp(self) -> float:
        values = self._current_mode_temp_values()
        if values:
            return float(values[0])
        # Fallback if mode has no temps (won’t be shown anyway)
        return float(self.device_data.target_temp or 0)

    def max_temp(self) -> float:
        values = self._current_mode_temp_values()
        if values:
            return float(values[-1])
        return float(self.device_data.target_temp or 0)

    async def async_send_api_call(self, action_type, action_value) -> dict:
        """Make service call to api."""
        from time import time

        if not self.available():
            return None

        if self.device_data.ac_states.get(
            "power"
        ) == HVACMode.OFF and action_type not in ("power", "preset"):
            return None

        self.last_action_timestamp = int(time())
        self.device_data.ac_states[action_type] = action_value
        response = await self.set_ac_state(
            mac_address=self.device_data.mac_address,
            action_type=action_type,
            actions=self.device_data.ac_states,
        )
        self.device_data.ac_states = response.get("data")
        self.device_data.hvac_mode = self.device_data.ac_states["mode"]
        self.last_action = self.device_data.ac_states
        return response

    async def async_set_temperature(self, ha_unit, **kwargs: Any) -> None:
        response = None

        if self.device_data.preset_mode != 0:
            self.device_data.ac_states["preset"] = 0

        # --- HEAT_COOL / range mode ---
        if self.device_data.hvac_mode == HVACMode.HEAT_COOL:
            heat_temp = kwargs.get(ATTR_TARGET_TEMP_LOW)
            cool_temp = kwargs.get(ATTR_TARGET_TEMP_HIGH)

            if (
                heat_temp == self.device_data.target_heat_set_point
                and cool_temp == self.device_data.target_cool_set_point
            ):
                return None

            # Thermostat in Celsius: accept 0.5 steps
            if self._supports_half_step(ha_unit):
                if heat_temp is not None:
                    new_heat = self._round_to_half(float(heat_temp))
                    if float(new_heat) != float(self.device_data.target_heat_set_point):
                        response = await self.async_send_api_call(
                            action_type="heat_set_point",
                            action_value=new_heat,
                        )

                if cool_temp is not None:
                    new_cool = self._round_to_half(float(cool_temp))
                    if float(new_cool) != float(self.device_data.target_cool_set_point):
                        response = await self.async_send_api_call(
                            action_type="cool_set_point",
                            action_value=new_cool,
                        )
                return response

            # Non-thermostat or non-Celsius: keep current integer snapping
            valid = self._current_mode_temp_values()
            if not valid:
                return None  # nothing to do

            heat_temp = self._find_valid_target_temp(heat_temp, valid)
            cool_temp = self._find_valid_target_temp(cool_temp, valid)

            if float(heat_temp) != float(self.device_data.target_heat_set_point):
                response = await self.async_send_api_call(
                    action_type="heat_set_point", action_value=float(heat_temp)
                )
            elif float(cool_temp) != float(self.device_data.target_cool_set_point):
                response = await self.async_send_api_call(
                    action_type="cool_set_point", action_value=float(cool_temp)
                )
            return response

        # --- Single setpoint mode ---
        t = kwargs.get(ATTR_TEMPERATURE)
        if (
            t is None
            or not self.device_data.device_on
            or self.device_data.hvac_mode == HVACMode.OFF
        ):
            return None

        if not self._mode_supports_temperature():
            return None

        # Thermostat in Celsius: 0.5°C steps
        if self._supports_half_step(ha_unit):
            new_t = self._round_to_half(float(t))
            if float(new_t) == float(self.device_data.target_temp or 0):
                return None
            self.device_data.ac_states["set_point"] = new_t
            response = await self.async_send_api_call(
                action_type="set_point", action_value=new_t
            )
            return response

        # Non-thermostat or non-Celsius: snap to supported integer values
        valid = self._current_mode_temp_values()
        if not valid:
            return None  # nothing to do

        new_t = self._find_valid_target_temp(t, valid)
        self.device_data.ac_states["set_point"] = int(new_t)
        response = await self.async_send_api_call(
            action_type="set_point", action_value=int(new_t)
        )
        return response

    async def async_set_hvac_mode(self, hvac_mode: HVACMode) -> dict:
        if hvac_mode == HVACMode.OFF:
            if self.device_data.is_thermostat:
                action_type = "mode"
            else:
                action_type = "power"
                if str(self.device_data.ac_states["preset"]) != 0:
                    self.device_data.ac_states.update({"preset": 0})

            return await self.async_send_api_call(
                action_type=action_type, action_value="off"
            )

        if hvac_mode == HVACMode.FAN_ONLY:
            hvac_mode = "fan"

        if (
            not self.device_data.is_thermostat
            and self.device_data.ac_states["preset"] != 0
        ):
            applied_preset = self.device_data.supported_features["presets"][
                int(self.device_data.ac_states["preset"]) - 1
            ]
            if (
                (
                    "vanish"
                    in self.device_data.supported_features["modes"][hvac_mode]["rules"]
                )
                and applied_preset["mode"] == "smart mode"
            ) or applied_preset["mode"] != "smart mode":
                self.device_data.ac_states.update({"preset": 0})

        if (
            not self.device_data.is_thermostat
            and self.device_data.ac_states["power"] == "off"
        ):
            self.device_data.ac_states["power"] = "on"

        if not getattr(self.device_data, "is_thermostat", False):
            values = self._current_mode_temp_values()
            self.device_data.temp_list = values

            if (
                values
                and int(self.device_data.ac_states.get("set_point")) not in values
            ):
                temp = values[0]
                try:
                    if int(self.device_data.ac_states["set_point"]) > values[-1]:
                        temp = values[-1]
                except (TypeError, ValueError):
                    pass
                self.device_data.ac_states["set_point"] = str(temp)
                self.device_data.target_temp = temp

        self.device_data.hvac_mode = hvac_mode
        new_fan_modes = self.current_mode_fan_speed()  # uses new mode caps dynamically
        if new_fan_modes:
            cur_fan = self.device_data.ac_states.get("fan_speed", "")
            if cur_fan not in new_fan_modes:
                # Repair to a valid option (prefer 'auto' if present)
                repaired = "auto" if "auto" in new_fan_modes else new_fan_modes[0]
                self.device_data.ac_states["fan_speed"] = repaired
                self.device_data.fan_mode = repaired

        self.device_data.ac_states["mode"] = hvac_mode
        self.device_data.hvac_mode = hvac_mode
        return await self.async_send_api_call(
            action_type="mode", action_value=hvac_mode
        )

    async def async_set_swing_mode(self, swing_mode: str) -> dict:
        self.device_data.ac_states["swing_position"] = swing_mode
        return await self.async_send_api_call(
            action_type="swing_position", action_value=swing_mode
        )

    async def async_set_fan_mode(self, fan_mode: str) -> dict | None:
        available = self.fan_modes()
        if not available:
            return None
        self.device_data.ac_states["fan_speed"] = fan_mode
        self.device_data.ac_states["preset"] = 0
        return await self.async_send_api_call(
            action_type="fan_speed", action_value=fan_mode
        )

    async def async_set_preset_mode(self, preset_mode: str) -> dict | None:
        if not self.device_data.preset_modes:
            return None

        idx = self.device_data.preset_modes.index(preset_mode) + 1
        self.device_data.ac_states.update({"preset": idx})

        applied_preset = self.device_data.supported_features["presets"][idx - 1]

        if applied_preset and not self.device_data.is_thermostat:
            current_mode = self.device_data.ac_states["mode"]
            if applied_preset["mode"] == "off":
                self.device_data.ac_states["power"] = "off"
            elif (
                applied_preset["mode"] == "smart mode"
                and current_mode in ["fan", "dry"]
                and (
                    "vanish"
                    in self.device_data.supported_features["modes"][current_mode][
                        "rules"
                    ]
                )
            ):
                # Preset switches device out of 'fan/dry' into 'cool'
                self.device_data.ac_states["mode"] = "cool"

        # keep hvac_mode and fan_speed valid for the (possibly) new mode ---
        new_mode = self.device_data.ac_states.get("mode")
        if new_mode and new_mode != self.device_data.hvac_mode:
            # Update the top-level hvac_mode used by _mode_caps()
            self.device_data.hvac_mode = new_mode

        # Now fix fan_speed for this mode
        new_fan_modes = self.current_mode_fan_speed()  # based on updated hvac_mode
        if new_fan_modes:
            cur_fan = (self.device_data.ac_states.get("fan_speed") or "").strip()
            if not cur_fan or cur_fan not in new_fan_modes:
                repaired = "auto" if "auto" in new_fan_modes else new_fan_modes[0]
                self.device_data.ac_states["fan_speed"] = repaired
                self.device_data.fan_mode = repaired
        else:
            # Mode doesn't support fan levels; clean up to avoid sending junk
            self.device_data.fan_mode = ""

        return await self.async_send_api_call(action_type="preset", action_value=idx)
