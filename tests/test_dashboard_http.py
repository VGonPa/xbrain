from xbrain.dashboard_http import (
    SlidingWindowRateLimiter,
    has_action_header,
    is_same_origin_request,
)


def test_rate_limiter_rejects_until_window_expires():
    now = [100.0]
    limiter = SlidingWindowRateLimiter(clock=lambda: now[0])

    assert limiter.allow("client:search", limit=2, window_seconds=60) == (True, 0)
    assert limiter.allow("client:search", limit=2, window_seconds=60) == (True, 0)
    allowed, retry_after = limiter.allow("client:search", limit=2, window_seconds=60)
    assert allowed is False
    assert retry_after == 60

    now[0] = 161.0
    assert limiter.allow("client:search", limit=2, window_seconds=60) == (True, 0)


def test_origin_check_rejects_cross_site_and_accepts_cli_requests():
    assert is_same_origin_request(origin=None, host="brain.example", fetch_site=None)
    assert is_same_origin_request(
        origin="https://brain.example", host="brain.example", fetch_site="same-origin"
    )
    assert not is_same_origin_request(
        origin="https://evil.example", host="brain.example", fetch_site="cross-site"
    )
    assert not is_same_origin_request(
        origin="https://sub.brain.example", host="brain.example", fetch_site="same-site"
    )


def test_action_header_is_explicit():
    assert has_action_header("1")
    assert not has_action_header(None)
    assert not has_action_header("true")
