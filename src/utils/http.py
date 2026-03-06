import asyncio
from typing import Any
from config.logger import get_logger

logger = get_logger(__name__)

RETRYABLE_STATUS_CODES = {500, 502, 503, 509, 529, 429}
MAX_RETRIES = 6
INITIAL_BACKOFF = 3.0


async def request_with_retry(
    client: Any,
    method_name: str,
    url: str,
    max_retries: int = MAX_RETRIES,
    initial_backoff: float = INITIAL_BACKOFF,
    **kwargs,
) -> Any:
    """
    Make an HTTP request with retry logic and exponential backoff.

    Retries on transient server errors (502, 503, 429, 500, etc).

    Args:
        client: httpx.AsyncClient instance
        method: HTTP method (e.g., "post", "get")
        url: URL to request
        max_retries: Maximum number of retry attempts
        initial_backoff: Initial backoff delay in seconds
        **kwargs: Additional arguments to pass to the HTTP method

    Returns:
        The HTTP response object
    """
    last_response = None
    for attempt in range(max_retries + 1):
        method = getattr(client, method_name)
        response = await method(url, **kwargs)

        if response.status_code not in RETRYABLE_STATUS_CODES:
            return response

        last_response = response
        if attempt < max_retries:
            backoff = initial_backoff * (2**attempt)
            logger.warning(
                f"Got {response.status_code}, retrying in {backoff:.0f}s "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            await asyncio.sleep(backoff)

    return last_response
