from qwenpaw.providers.retry_chat_model import _is_retryable


def test_statusless_upstream_rate_limit_is_retryable() -> None:
    assert _is_retryable(Exception("Upstream rate limit exceeded, please retry later"))


def test_statusless_provider_internal_error_is_retryable() -> None:
    assert _is_retryable(Exception(
        "An error occurred while processing your request. You can retry your request"
    ))


def test_statusless_upstream_request_failure_is_retryable() -> None:
    assert _is_retryable(Exception("Upstream request failed"))


def test_statusless_auth_error_is_not_retryable() -> None:
    assert not _is_retryable(Exception("You didn't provide an API key"))
