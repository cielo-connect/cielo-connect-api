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

__version__ = "1.0.5"

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
        self.username = api_key
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
            self.username = result["data"].get("username", self.api_key)
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
