"""Jira Cloud REST API v3 client (create + read issues).

Auth: HTTP Basic with JIRA_EMAIL + JIRA_API_TOKEN (from .env, never hard-coded).
Robustness:
  - timeouts on every request (JIRA_TIMEOUT_SECONDS)
  - retries with exponential backoff on connection errors, timeouts, 429 and 5xx (JIRA_MAX_RETRIES)
  - TicketingUnavailable  -> Jira unreachable / down: agents tell the employee to try again later
  - TicketingError        -> Jira rejected the request (bad project, missing permission, invalid field)
"""
from __future__ import annotations

import logging

import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config import settings

log = logging.getLogger(__name__)


class TicketingUnavailable(Exception):
    """Ticketing system can't be reached (network error, timeout, 5xx, rate-limited)."""


class TicketingError(Exception):
    """Ticketing system rejected the request (4xx other than 429)."""

    def __init__(self, message: str, status: int | None = None, details: dict | None = None):
        super().__init__(message)
        self.status = status
        self.details = details or {}


class _Retryable(Exception):
    pass


def _error_text(resp: requests.Response) -> tuple[str, dict]:
    try:
        body = resp.json()
    except ValueError:
        return resp.text[:300], {}
    msgs = list(body.get("errorMessages", [])) + [f"{k}: {v}" for k, v in body.get("errors", {}).items()]
    return "; ".join(msgs) or resp.text[:300], body.get("errors", {})


class JiraClient:
    def __init__(self, base_url: str | None = None, email: str | None = None,
                 api_token: str | None = None, session: requests.Session | None = None):
        self.base_url = (base_url or settings.jira_base_url()).rstrip("/")
        self.session = session or requests.Session()
        self.session.auth = (email or settings.jira_email(), api_token or settings.jira_api_token())
        self.session.headers.update({"Accept": "application/json", "Content-Type": "application/json"})

    # ------------------------------------------------------------------ low level
    def _request(self, method: str, path: str, **kwargs) -> dict:
        @retry(retry=retry_if_exception_type(_Retryable),
               stop=stop_after_attempt(settings.JIRA_MAX_RETRIES),
               wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
        def _once() -> requests.Response:
            try:
                resp = self.session.request(method, f"{self.base_url}{path}",
                                            timeout=settings.JIRA_TIMEOUT_SECONDS, **kwargs)
            except (requests.ConnectionError, requests.Timeout) as e:
                raise _Retryable(str(e)) from e
            if resp.status_code == 429 or resp.status_code >= 500:
                raise _Retryable(f"HTTP {resp.status_code}")
            return resp

        try:
            resp = _once()
        except _Retryable as e:
            log.error("Jira unreachable: %s %s -> %s", method, path, e)
            raise TicketingUnavailable(f"Jira is unreachable: {e}") from e

        if resp.status_code >= 400:
            text, details = _error_text(resp)
            if resp.status_code == 401:
                text = "Authentication failed – check JIRA_EMAIL and JIRA_API_TOKEN"
            raise TicketingError(f"Jira returned {resp.status_code}: {text}", resp.status_code, details)
        return resp.json() if resp.content else {}

    # ------------------------------------------------------------------ API
    def myself(self) -> dict:
        return self._request("GET", "/rest/api/3/myself")

    def get_project(self, key: str) -> dict:
        return self._request("GET", f"/rest/api/3/project/{key}")

    def create_issue(self, fields: dict) -> dict:
        """POST /rest/api/3/issue -> {'id', 'key', 'self'}."""
        return self._request("POST", "/rest/api/3/issue", json={"fields": fields})

    def get_issue(self, key: str, fields: str = "summary,status,priority,labels,created,updated") -> dict:
        return self._request("GET", f"/rest/api/3/issue/{key}", params={"fields": fields})

    def search_issues(self, jql: str, fields: str = "summary,status,priority,labels,created,updated",
                      max_results: int = 20) -> list[dict]:
        """Run a JQL search. Uses /rest/api/3/search/jql (current Jira Cloud) and falls back to the
        legacy /rest/api/3/search on sites that don't have it yet."""
        params = {"jql": jql, "fields": fields, "maxResults": max_results}
        try:
            body = self._request("GET", "/rest/api/3/search/jql", params=params)
        except TicketingError as e:
            if e.status not in (404, 405, 410):
                raise
            body = self._request("GET", "/rest/api/3/search", params=params)
        return body.get("issues", [])

    def get_comments(self, key: str, max_results: int = 5) -> list[dict]:
        """Newest comments first (raw Jira comment objects)."""
        body = self._request("GET", f"/rest/api/3/issue/{key}/comment",
                             params={"orderBy": "-created", "maxResults": max_results})
        return body.get("comments", [])

    def add_comment(self, key: str, text: str) -> dict:
        body = {"type": "doc", "version": 1,
                "content": [{"type": "paragraph", "content": [{"type": "text", "text": text}]}]}
        return self._request("POST", f"/rest/api/3/issue/{key}/comment", json={"body": body})

    def browse_url(self, key: str) -> str:
        return f"{self.base_url}/browse/{key}"
