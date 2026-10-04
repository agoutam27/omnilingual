import subprocess

import pytest

from omnilingual.ui import audio


@pytest.fixture
def no_sudo(monkeypatch):
    """Fail loudly if any test ever reaches a real privileged command."""
    def boom(*a, **k):
        raise AssertionError("test attempted a privileged/real command")
    monkeypatch.setattr(subprocess, "run", boom)


def _ready(**over):
    fields = dict(device=True, blackhole=True, ffmpeg=True, ffprobe=True,
                  mic_authorized=True, output="Speakers", detail=[])
    fields.update(over)
    return audio.Readiness(**fields)


# What the helper's `list` prints once setup() has run, in its real "uid | name"
# shape rather than the bare names the brief's own tests use.
_FULL_LISTING = ("BlackHole_2ch | BlackHole 2ch\n"
                 "omnilingual.aggregate | Omnilingual\n"
                 "BuiltInMicDevice | MacBook Pro Microphone\n")


def test_readiness_ok_requires_all_five_signals():
    assert _ready().ok is True
    assert _ready().as_dict()["ok"] is True


def test_readiness_not_ok_when_the_device_is_missing():
    assert _ready(device=False).ok is False


def test_readiness_not_ok_when_mic_permission_is_denied():
    assert _ready(mic_authorized=False).ok is False


def test_as_dict_shape_matches_the_spec():
    assert _ready(output="Multi-Output Device").as_dict() == {
        "ok": True, "device": "Omnilingual", "blackhole": True, "ffmpeg": True,
        "ffprobe": True, "mic_authorized": True,
        "output": "Multi-Output Device", "detail": [],
    }


def test_probe_reports_missing_binaries_without_running_anything(monkeypatch, no_sudo):
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    result = audio.probe(mic=False)
    assert result.ffmpeg is False and result.ffprobe is False
    assert any("ffmpeg" in line for line in result.detail)


def test_probe_never_rewrites_the_system_output(monkeypatch):
    """Switching output silently breaks the volume keys, so probe is read-only."""
    calls = []

    def fake_run(cmd, **k):
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd, 0, stdout="Omnilingual\nBlackHole 2ch\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(audio.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(audio, "device_list", lambda: "Omnilingual\nBlackHole 2ch\n")
    monkeypatch.setattr(audio, "_current_output", lambda: "Speakers")
    audio.probe(device="Omnilingual", mic=False)
    joined = [" ".join(cmd) for cmd in calls]
    assert not any("SwitchAudioSource" in cmd and "-s" in cmd for cmd in joined)


def test_mic_probe_treats_permission_denied_as_unauthorized(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **k: subprocess.CompletedProcess(
            cmd, 1, stdout="", stderr="AVFoundation: Permission denied"))
    assert audio._mic_authorized("Omnilingual") is False


def test_mic_probe_rejects_permission_markers_even_on_exit_zero(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **k: subprocess.CompletedProcess(
            cmd, 0, stdout="", stderr="Device not permitted"))
    assert audio._mic_authorized("Omnilingual") is False


def test_mic_probe_accepts_a_clean_capture(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""))
    assert audio._mic_authorized("Omnilingual") is True


def test_restart_daemon_uses_the_mac_authorisation_prompt(monkeypatch):
    seen = []

    def fake_run(cmd, **k):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    audio.restart_daemon()
    assert len(seen) == 1
    joined = " ".join(seen[0])
    assert "killall coreaudiod" in joined
    assert "administrator privileges" in joined


# --- below: coverage the brief's list leaves open.  Same rules, narrower claims.


def test_probe_consults_switchaudiosource_read_only(monkeypatch):
    """Stronger form of the read-only claim: probe really does run a command.

    The brief's version stubs `_current_output` away, so it cannot fail even if
    probe started setting the output.  Here only `_current_output`'s own internals
    are left intact, so the assertion is on a command that was actually issued.
    """
    calls = []

    def fake_run(cmd, **k):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="Speakers", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(audio.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(audio, "device_list", lambda: "omnilingual.aggregate | Omnilingual\n")
    result = audio.probe(device="Omnilingual", mic=False)
    assert result.output == "Speakers"
    switches = [cmd for cmd in calls if "SwitchAudioSource" in cmd]
    assert switches, "probe should report the current output device"
    assert all("-c" in cmd and "-s" not in cmd for cmd in switches)


def test_current_output_is_none_when_switchaudiosource_is_absent(monkeypatch, no_sudo):
    monkeypatch.setattr(audio.shutil, "which", lambda name: None)
    assert audio._current_output() is None


def test_probe_does_not_claim_a_denial_it_never_observed(monkeypatch, no_sudo):
    """No device means no capture was attempted, so no denial may be reported.

    `no_sudo` turns any stray subprocess call into a failure, which is what makes
    "not probed" observable rather than asserted in prose.
    """
    monkeypatch.setattr(audio.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(audio, "device_list", lambda: "BlackHole_2ch | BlackHole 2ch\n")
    monkeypatch.setattr(audio, "_current_output", lambda: "Speakers")
    result = audio.probe(device="Omnilingual", mic=True)
    assert result.device is False and result.blackhole is True
    assert result.mic_authorized is False
    assert not any("microphone" in line.lower() for line in result.detail)


def test_probe_reports_the_missing_device_and_blackhole(monkeypatch, no_sudo):
    monkeypatch.setattr(audio.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(audio, "device_list", lambda: "BuiltInMicDevice | MacBook Pro Microphone\n")
    monkeypatch.setattr(audio, "_current_output", lambda: "Speakers")
    result = audio.probe(device="Omnilingual", mic=False)
    joined = " ".join(result.detail)
    assert "Omnilingual" in joined
    assert "brew install blackhole-2ch" in joined


def test_probe_survives_a_helper_that_cannot_be_compiled(monkeypatch, no_sudo):
    monkeypatch.setattr(audio.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(audio, "_current_output", lambda: None)

    def no_such_helper():
        raise FileNotFoundError("no swiftc")

    monkeypatch.setattr(audio, "device_list", no_such_helper)
    result = audio.probe(device="Omnilingual", mic=False)
    assert result.device is False and result.blackhole is False
    assert any("audio-device helper" in line for line in result.detail)


def test_mic_probe_captures_exactly_one_second_through_ffmpeg(monkeypatch):
    seen = {}

    def fake_run(cmd, **k):
        seen["cmd"] = cmd
        seen["kwargs"] = k
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    assert audio._mic_authorized("Omnilingual") is True
    assert seen["cmd"][0] == "ffmpeg"
    assert seen["cmd"][seen["cmd"].index("-f") + 1] == "avfoundation"
    assert seen["cmd"][seen["cmd"].index("-t") + 1] == "1"
    # The colon is load-bearing: a bare name asks avfoundation for a *video*
    # device and fails "Video device not found" on a machine whose mic is fine.
    assert seen["cmd"][seen["cmd"].index("-i") + 1] == ":Omnilingual"
    assert seen["kwargs"].get("shell") in (None, False)


def test_capture_verdict_points_at_privacy_settings_on_a_real_denial(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **k: subprocess.CompletedProcess(
            cmd, 1, stdout="",
            stderr="[AVFoundation indev] Error opening input: Permission denied"))
    ok, reason = audio._capture_verdict("Omnilingual")
    assert ok is False
    assert "Privacy & Security" in reason


def test_capture_verdict_does_not_blame_permission_for_another_failure(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **k: subprocess.CompletedProcess(
            cmd, 251, stdout="",
            stderr="[AVFoundation indev] Video device not found\n"
                   "Error opening input file :Omnilingual."))
    ok, reason = audio._capture_verdict("Omnilingual")
    assert ok is False
    assert "Privacy & Security" not in reason
    assert "Error opening input file" in reason


def test_probe_does_not_blame_permission_for_an_unrelated_capture_failure(monkeypatch):
    calls = []

    def fake_run(cmd, **k):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(
            cmd, 251, stdout="",
            stderr="[AVFoundation indev] Video device not found\n"
                   "Error opening input file :Omnilingual.")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(audio.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(audio, "device_list", lambda: "omnilingual.aggregate | Omnilingual\n")
    monkeypatch.setattr(audio, "_current_output", lambda: "Speakers")
    result = audio.probe(device="Omnilingual", mic=True)
    assert result.mic_authorized is False
    joined = " ".join(result.detail)
    assert "Error opening input file" in joined
    assert "Privacy & Security" not in joined
    assert any(cmd[0] == "ffmpeg" for cmd in calls), "the capture must be attempted"


def test_probe_reports_authorized_when_the_capture_succeeds(monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""))
    monkeypatch.setattr(audio.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(audio, "device_list", lambda: _FULL_LISTING)
    monkeypatch.setattr(audio, "_current_output", lambda: "Speakers")
    result = audio.probe(device="Omnilingual", mic=True)
    assert result.mic_authorized is True
    assert result.detail == []


def test_restart_daemon_interpolates_nothing_into_the_shell_script(monkeypatch):
    """A value placed inside the `do shell script` string would be executed as shell."""
    seen = []

    def fake_run(cmd, **k):
        seen.append((cmd, k))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    audio.restart_daemon()
    cmd, kwargs = seen[0]
    assert cmd[0] == "osascript" and cmd[1] == "-e"
    assert cmd[2] == 'do shell script "killall coreaudiod" with administrator privileges'
    assert len(cmd) == 3
    assert kwargs.get("shell") in (None, False)
    # A cancelled auth dialog exits nonzero; without check=True that would read
    # as success and the UI would claim the daemon restarted.
    assert kwargs.get("check") is True


def test_helper_path_compiles_the_swift_helper_once(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **k):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(audio, "HELPER_SRC", tmp_path / "audio-devices.swift")
    monkeypatch.setattr(audio, "_helper", None)
    monkeypatch.setattr(subprocess, "run", fake_run)
    src = tmp_path / "audio-devices.swift"
    src.write_text("// stub\n", encoding="utf-8")
    first = audio.helper_path()
    second = audio.helper_path()
    assert first == second, "the compiled helper must be cached per process"
    assert first.parent.is_dir() and first.name == "audio-devices"
    assert len(calls) == 1
    assert calls[0][0] == "swiftc"
    assert str(src) in calls[0]


def test_helper_path_refuses_to_build_a_helper_that_is_not_there(monkeypatch, tmp_path):
    monkeypatch.setattr(audio, "HELPER_SRC", tmp_path / "absent.swift")
    monkeypatch.setattr(audio, "_helper", None)

    def boom(*a, **k):
        raise AssertionError("must not compile a missing helper")

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(FileNotFoundError):
        audio.helper_path()


def test_device_list_returns_the_helper_stdout(monkeypatch, tmp_path):
    binary = tmp_path / "audio-devices"
    monkeypatch.setattr(audio, "helper_path", lambda: binary)
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **k: subprocess.CompletedProcess(
            cmd, 0, stdout="omnilingual.aggregate | Omnilingual\n", stderr=""))
    assert audio.device_list() == "omnilingual.aggregate | Omnilingual\n"


def test_setup_creates_the_aggregate_and_reports_ready(monkeypatch, tmp_path):
    binary = tmp_path / "audio-devices"
    calls = []

    def fake_run(cmd, **k):
        calls.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(audio, "helper_path", lambda: binary)
    monkeypatch.setattr(audio, "device_list", lambda: "BlackHole_2ch | BlackHole 2ch\n")
    monkeypatch.setattr(audio.time, "sleep", lambda _s: None)
    monkeypatch.setattr(subprocess, "run", fake_run)
    lines = list(audio.setup())
    assert lines[0] == "building the audio-device helper"
    assert lines[-1] == "Aggregate Device 'Omnilingual' is ready"
    subcommands = [cmd[1] for cmd in calls if cmd[0] == str(binary)]
    assert subcommands == ["ensure", "check"], "ensure then verify, nothing else"
    assert not any(cmd[0] == "SwitchAudioSource" for cmd in calls), (
        "setup must never route system audio anywhere")


def test_setup_reports_a_helper_that_refuses_to_create_the_device(monkeypatch, tmp_path):
    binary = tmp_path / "audio-devices"
    monkeypatch.setattr(audio, "helper_path", lambda: binary)
    monkeypatch.setattr(audio, "device_list", lambda: "BlackHole_2ch | BlackHole 2ch\n")
    monkeypatch.setattr(audio.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **k: subprocess.CompletedProcess(
            cmd, 1 if cmd[1:2] == ["ensure"] else 0, stdout="", stderr="no parts"))
    with pytest.raises(RuntimeError) as ei:
        list(audio.setup())
    assert "no parts" in str(ei.value)


def test_setup_gives_up_when_blackhole_never_appears(monkeypatch, tmp_path):
    slept = []
    monkeypatch.setattr(audio, "helper_path", lambda: tmp_path / "audio-devices")
    monkeypatch.setattr(audio, "device_list", lambda: "BuiltInMicDevice | Microphone\n")
    monkeypatch.setattr(audio.time, "sleep", slept.append)
    monkeypatch.setattr(
        subprocess, "run",
        lambda cmd, **k: subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""))
    with pytest.raises(RuntimeError) as ei:
        list(audio.setup())
    assert "brew install blackhole-2ch" in str(ei.value)
    assert len(slept) == 30, "bounded retry, not an unbounded hang"
