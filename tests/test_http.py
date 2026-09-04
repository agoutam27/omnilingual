import httpx
import pytest

from omnilingual.http import (
    AuthError,
    QuotaError,
    SarvamError,
    TransientError,
    send_with_retry,
)


def _resp(status: int, body: str = "{}") -> httpx.Response:
    return httpx.Response(status, text=body, request=httpx.Request("POST", "https://x"))


def _sequence(*responses):
    it = iter(responses)

    def send():
        r = next(it)
        if isinstance(r, Exception):
            raise r
        return r

    return send


def test_success_first_try():
    r = send_with_retry(_sequence(_resp(200)), sleep=lambda s: None, jitter=lambda: 0)
    assert r.status_code == 200


def test_retries_on_429_then_succeeds():
    slept: list[float] = []
    r = send_with_retry(
        _sequence(_resp(429), _resp(500), _resp(200)),
        base_delay=1.0,
        sleep=slept.append,
        jitter=lambda: 0.0,
    )
    assert r.status_code == 200
    assert slept == [1.0, 2.0]


def test_transport_error_is_retried():
    req = httpx.Request("POST", "https://x")
    r = send_with_retry(
        _sequence(httpx.ConnectError("boom", request=req), _resp(200)),
        sleep=lambda s: None,
        jitter=lambda: 0.0,
    )
    assert r.status_code == 200


def test_gives_up_after_retries():
    with pytest.raises(TransientError) as ei:
        send_with_retry(
            _sequence(_resp(503), _resp(503), _resp(503), _resp(503)),
            retries=3,
            sleep=lambda s: None,
            jitter=lambda: 0.0,
        )
    assert ei.value.status == 503


def test_401_raises_auth_error_without_retry():
    calls = {"n": 0}

    def send():
        calls["n"] += 1
        return _resp(401, '{"error":"bad key"}')

    with pytest.raises(AuthError) as ei:
        send_with_retry(send, sleep=lambda s: None)
    assert calls["n"] == 1
    assert "bad key" in ei.value.body


def test_403_is_auth_error():
    with pytest.raises(AuthError):
        send_with_retry(_sequence(_resp(403)), sleep=lambda s: None)


def test_402_raises_quota_error():
    with pytest.raises(QuotaError):
        send_with_retry(_sequence(_resp(402)), sleep=lambda s: None)


def test_other_4xx_is_plain_sarvam_error():
    with pytest.raises(SarvamError) as ei:
        send_with_retry(_sequence(_resp(422, "bad field")), sleep=lambda s: None)
    assert not isinstance(ei.value, (AuthError, QuotaError, TransientError))
    assert ei.value.status == 422
