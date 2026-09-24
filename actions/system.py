"""
System-level actions: sleep, time, hotspot, media/volume controls, lock screen.
Hotspot via WinRT — no admin required.
"""

import subprocess
from datetime import datetime


def sleep_system():
    """Bravo six, going dark.

    Hibernates rather than sleeping on this system (AC hibernate-after is 0,
    Modern Standby collapses straight to hibernate) — left as-is intentionally,
    hibernate is fine here since it also fully powers down (security benefit).
    """
    subprocess.Popen(["rundll32.exe", "powrprof.dll,SetSuspendState", "0", "1", "0"])


def tell_time():
    now = datetime.now().strftime("%I:%M %p")
    print(f"[system] It's {now}")
    # TODO: TTS output


def music_play_pause():
    _media_key("play_pause")


def music_next():
    _media_key("next")


def music_prev():
    _media_key("prev")


def _media_key(action: str):
    """Send a system media key via keybd_event (works globally: Spotify, YouTube, etc)."""
    import ctypes

    VK_MEDIA_PLAY_PAUSE = 0xB3
    VK_MEDIA_NEXT_TRACK = 0xB0
    VK_MEDIA_PREV_TRACK = 0xB1
    KEYEVENTF_EXTENDEDKEY = 0x1
    KEYEVENTF_KEYUP = 0x2

    key_map = {
        "play_pause": VK_MEDIA_PLAY_PAUSE,
        "next": VK_MEDIA_NEXT_TRACK,
        "prev": VK_MEDIA_PREV_TRACK,
    }
    vk = key_map[action]
    ctypes.windll.user32.keybd_event(vk, 0, KEYEVENTF_EXTENDEDKEY, 0)
    ctypes.windll.user32.keybd_event(vk, 0, KEYEVENTF_EXTENDEDKEY | KEYEVENTF_KEYUP, 0)
    print(f"[system] Media key: {action}")


def music_volume_up():
    _change_volume(0.10)


def music_volume_down():
    _change_volume(-0.10)


def _change_volume(delta: float):
    try:
        from pycaw.pycaw import AudioUtilities

        volume = AudioUtilities.GetSpeakers().EndpointVolume
        current = volume.GetMasterVolumeLevelScalar()
        new_level = min(1.0, max(0.0, current + delta))
        volume.SetMasterVolumeLevelScalar(new_level, None)
        print(f"[system] Volume -> {int(new_level * 100)}%")
    except Exception as e:
        print(f"[system] Volume control error: {e}")


def music_mute():
    try:
        from pycaw.pycaw import AudioUtilities

        volume = AudioUtilities.GetSpeakers().EndpointVolume
        muted = volume.GetMute()
        volume.SetMute(not muted, None)
        print(f"[system] Mute -> {not muted}")
    except Exception as e:
        print(f"[system] Mute control error: {e}")


def lock_screen():
    subprocess.Popen(["rundll32.exe", "user32.dll,LockWorkStation"])


def hotspot_on():
    _set_hotspot(True)


def hotspot_off():
    _set_hotspot(False)


def _set_hotspot(enable: bool):
    """
    Toggle mobile hotspot via WinRT NetworkOperatorTetheringManager.
    No admin required.
    """
    try:
        import winrt.windows.networking.networkoperators as netop

        manager = netop.NetworkOperatorTetheringManager.create_from_connection_profile(
            _get_connection_profile()
        )
        if enable:
            manager.start_tethering_async().get()
        else:
            manager.stop_tethering_async().get()

    except ImportError:
        print("[system] winrt not installed, hotspot control unavailable")
    except Exception as e:
        print(f"[system] Hotspot error: {e}")


def _get_connection_profile():
    import winrt.windows.networking.connectivity as connectivity
    return connectivity.NetworkInformation.get_internet_connection_profile()

def cancel():
    pass