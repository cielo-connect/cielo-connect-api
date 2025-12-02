# Cielo Connect API — Async Python Client

**Async Python API client for the Cielo Home platform.**

This library provides an asynchronous interface to the **Cielo Connect Cloud API**, enabling you to control and monitor Cielo Smart Thermostats and Cielo Breez devices from Python.

---

## Features

- Fetch device information and live sensor data  
- Control temperature, modes, presets, fan speed, and swing
- Robust retry and exponential backoff  
- Fully asynchronous (`aiohttp`)  
- Clean dataclass-based device model  

---

## Installation

```bash
pip install cielo-connect-api
```

## Quick Example
```bash
import asyncio
from cielo_connect_api import CieloClient
```

    async def main():
        api_key = "YOUR_API_KEY"
    
        async with CieloClient(api_key) as client:
            # Get all device information
            data = await client.get_devices_data()
            print(data.parsed)
    
            # Example: set the AC to COOL at 72°F
            device = list(data.parsed.values())[0]
            client.device_data = device
            await client.async_set_hvac_mode("cool")
            await client.async_set_temperature("F", temperature=72)
    
    asyncio.run(main())


## Library Usage
Initialize Client
```
client = CieloClient(api_key="your-key-here")

Common Operations
Get all device data
data = await client.get_devices_data()

Set HVAC mode
await client.async_set_hvac_mode("cool")

Set temperature
await client.async_set_temperature("F", temperature=72)

Set fan or swing mode
await client.async_set_fan_mode("high")
await client.async_set_swing_mode("pos1")
```

## License

This project is licensed under the MIT License, an OSI-approved open-source license.