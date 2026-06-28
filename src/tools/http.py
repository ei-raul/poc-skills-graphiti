import json
from typing import Any, Dict, List, Optional
import httpx
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from config.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TIMEOUT = 30.0


class HttpGetInput(BaseModel):
    url: str = Field(..., description="Full URL to request (e.g. 'https://api.example.com/data')")
    params: Optional[Dict[str, Any]] = Field(
        None, description="Query string parameters as a dict (e.g. {'page': 1, 'limit': 10})"
    )
    headers: Optional[Dict[str, str]] = Field(
        None, description="Additional HTTP headers (e.g. {'Authorization': 'Bearer <token>'})"
    )
    timeout: float = Field(DEFAULT_TIMEOUT, description="Request timeout in seconds (default: 30)")


class HttpPostInput(BaseModel):
    url: str = Field(..., description="Full URL to send the POST to")
    body: Optional[Dict[str, Any]] = Field(
        None, description="JSON body payload as a dict. Use this for application/json requests."
    )
    form_data: Optional[Dict[str, str]] = Field(
        None,
        description="Form-encoded body as a dict. Use when the API expects application/x-www-form-urlencoded.",
    )
    headers: Optional[Dict[str, str]] = Field(
        None, description="Additional HTTP headers (e.g. {'Authorization': 'Bearer <token>'})"
    )
    timeout: float = Field(DEFAULT_TIMEOUT, description="Request timeout in seconds (default: 30)")


class HttpRequestInput(BaseModel):
    method: str = Field(
        ..., description="HTTP method: GET, POST, PUT, PATCH, DELETE"
    )
    url: str = Field(..., description="Full URL to request")
    params: Optional[Dict[str, Any]] = Field(None, description="Query string parameters")
    body: Optional[Dict[str, Any]] = Field(None, description="JSON body payload")
    headers: Optional[Dict[str, str]] = Field(None, description="HTTP headers")
    timeout: float = Field(DEFAULT_TIMEOUT, description="Request timeout in seconds (default: 30)")


def _build_response(response: httpx.Response) -> Dict[str, Any]:
    content_type = response.headers.get("content-type", "")
    try:
        body = response.json() if "application/json" in content_type else response.text
    except Exception:
        body = response.text

    return {
        "status_code": response.status_code,
        "ok": response.is_success,
        "headers": dict(response.headers),
        "body": body,
    }


@tool(args_schema=HttpGetInput)
async def http_get(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    Perform an HTTP GET request to any URL.

    Use this to call REST APIs, fetch JSON data, or read public web resources.
    Returns status_code, ok (bool), headers, and body (parsed JSON or raw text).

    Examples:
    - Fetch a public API: http_get(url='https://api.github.com/repos/owner/repo')
    - With auth header: http_get(url='...', headers={'Authorization': 'Bearer TOKEN'})
    - With query params: http_get(url='https://api.example.com/search', params={'q': 'python', 'page': 1})
    """
    logger.info(f"GET {url} params={params}")
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            response = await client.get(url, params=params, headers=headers)
            result = _build_response(response)
            logger.info(f"GET {url} -> {response.status_code}")
            return result
    except httpx.TimeoutException:
        return {"status_code": 408, "ok": False, "error": f"Request timed out after {timeout}s"}
    except Exception as e:
        return {"status_code": 0, "ok": False, "error": f"{type(e).__name__}: {e}"}


@tool(args_schema=HttpPostInput)
async def http_post(
    url: str,
    body: Optional[Dict[str, Any]] = None,
    form_data: Optional[Dict[str, str]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    Perform an HTTP POST request to any URL.

    Use `body` for JSON payloads (most REST APIs) and `form_data` for
    application/x-www-form-urlencoded. Returns status_code, ok (bool), headers, and body.

    Examples:
    - POST JSON: http_post(url='https://api.example.com/items', body={'name': 'test'})
    - With auth: http_post(url='...', body={...}, headers={'Authorization': 'Bearer TOKEN'})
    - Form POST: http_post(url='https://api.example.com/login', form_data={'user': 'x', 'pass': 'y'})
    """
    logger.info(f"POST {url}")
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            kwargs: Dict[str, Any] = {"headers": headers}
            if form_data is not None:
                kwargs["data"] = form_data
            else:
                kwargs["json"] = body
            response = await client.post(url, **kwargs)
            result = _build_response(response)
            logger.info(f"POST {url} -> {response.status_code}")
            return result
    except httpx.TimeoutException:
        return {"status_code": 408, "ok": False, "error": f"Request timed out after {timeout}s"}
    except Exception as e:
        return {"status_code": 0, "ok": False, "error": f"{type(e).__name__}: {e}"}


@tool(args_schema=HttpRequestInput)
async def http_request(
    method: str,
    url: str,
    params: Optional[Dict[str, Any]] = None,
    body: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, Any]:
    """
    Perform any HTTP request (GET, POST, PUT, PATCH, DELETE).

    Use this when you need methods other than GET/POST, such as PUT to update a
    resource or DELETE to remove one. For simple GET/POST prefer the dedicated tools.

    Returns status_code, ok (bool), headers, and body (parsed JSON or raw text).

    Examples:
    - PUT: http_request(method='PUT', url='https://api.example.com/items/1', body={'name': 'new'})
    - DELETE: http_request(method='DELETE', url='https://api.example.com/items/1', headers={...})
    - PATCH: http_request(method='PATCH', url='...', body={'field': 'value'})
    """
    method = method.upper()
    logger.info(f"{method} {url}")
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
            request_method = getattr(client, method.lower(), None)
            if request_method is None:
                return {"status_code": 0, "ok": False, "error": f"Unsupported HTTP method: {method}"}

            kwargs: Dict[str, Any] = {"params": params, "headers": headers}
            if body is not None:
                kwargs["json"] = body

            response = await request_method(url, **kwargs)
            result = _build_response(response)
            logger.info(f"{method} {url} -> {response.status_code}")
            return result
    except httpx.TimeoutException:
        return {"status_code": 408, "ok": False, "error": f"Request timed out after {timeout}s"}
    except Exception as e:
        return {"status_code": 0, "ok": False, "error": f"{type(e).__name__}: {e}"}
