"""
Maps action keys to handler functions.
Add a new command? Register it here.
"""

from actions import apps, system

_REGISTRY: dict = {
    "open_chrome": apps.open_chrome,
    "open_spotify": apps.open_spotify,
    "sleep_system": system.sleep_system,
    "tell_time": system.tell_time,
    "hotspot_on": system.hotspot_on,
    "hotspot_off": system.hotspot_off,

    "open_all": apps.open_all,
    "start_dev_mode": apps.open_all,
    "work_mode": apps.open_all,
    "open_claude": apps.open_claude,
    "open_edge": apps.open_edge,
    "open_vscode": apps.open_vscode,
    "open_steam": apps.open_steam,

    "music_play_pause": system.music_play_pause,
    "music_next": system.music_next,
    "music_prev": system.music_prev,
    "music_volume_up": system.music_volume_up,
    "music_volume_down": system.music_volume_down,
    "music_mute": system.music_mute,

    "lock_screen": system.lock_screen,
    "cancel": system.cancel,
}


def dispatch(action_key: str):
    handler = _REGISTRY.get(action_key)
    if handler:
        handler()
    else:
        print(f"[registry] Unknown action: {action_key}")
