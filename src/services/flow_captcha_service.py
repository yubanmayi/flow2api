"""Dedicated Flow browser captcha service client."""
import time
from typing import Any, Dict, Optional, List

from curl_cffi.requests import AsyncSession

from ..core.config import config
from ..core.logger import debug_logger


class FlowCaptchaServiceError(Exception):
    """Typed exception for Flow captcha service errors."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 500,
        error_type: str = "upstream_error",
        error_code: str = "captcha_upstream_error"
    ):
        super().__init__(message)
        self.status_code = status_code
        self.error_type = error_type
        self.error_code = error_code


class FlowCaptchaService:
    """Call external Flow captcha service through browser provider."""

    TASK_NAME = "Flow-RecaptchaV3TaskProxylessM1"
    TASK_TYPE = "RecaptchaV3TaskProxylessM1"
    PROVIDER = "browser"

    @staticmethod
    def _join_url(base_url: str, path: str) -> str:
        base = (base_url or "").rstrip("/")
        if not path:
            path = "/"
        if not path.startswith("/"):
            path = f"/{path}"
        return f"{base}{path}"

    @staticmethod
    def _extract_error_message(payload: Any) -> Optional[str]:
        if not isinstance(payload, dict):
            return None

        error = payload.get("error")
        if isinstance(error, dict):
            message = error.get("message") or error.get("detail")
            if message:
                return str(message)
        elif error:
            return str(error)

        for key in ("message", "detail", "errorDescription", "error_message"):
            value = payload.get(key)
            if value:
                return str(value)
        return None

    @staticmethod
    def _extract_int(payload: Dict[str, Any], keys: List[str]) -> Optional[int]:
        for key in keys:
            value = payload.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _extract_token(payload: Dict[str, Any]) -> Optional[str]:
        candidates = []

        for key in ("token", "gRecaptchaResponse"):
            value = payload.get(key)
            if isinstance(value, str) and value:
                candidates.append(value)

        solution = payload.get("solution")
        if isinstance(solution, dict):
            for key in ("token", "gRecaptchaResponse", "response"):
                value = solution.get(key)
                if isinstance(value, str) and value:
                    candidates.append(value)

        data = payload.get("data")
        if isinstance(data, dict):
            for key in ("token", "gRecaptchaResponse"):
                value = data.get(key)
                if isinstance(value, str) and value:
                    candidates.append(value)

        return candidates[0] if candidates else None

    @classmethod
    async def solve_recaptcha_v3_task_proxyless_m1(
        cls,
        *,
        project_id: Optional[str],
        website_url: Optional[str],
        page_action: str
    ) -> Dict[str, Any]:
        """Solve reCAPTCHA V3 via dedicated Flow captcha service."""
        if not project_id and not website_url:
            raise FlowCaptchaServiceError(
                "project_id or website_url is required",
                status_code=400,
                error_type="invalid_request_error",
                error_code="invalid_request"
            )

        api_key = config.flow_captcha_service_api_key
        if not api_key:
            raise FlowCaptchaServiceError(
                "flow captcha service api key is not configured",
                status_code=500,
                error_type="configuration_error",
                error_code="captcha_config_error"
            )

        url = cls._join_url(
            config.flow_captcha_service_base_url,
            config.flow_captcha_service_solve_path
        )
        timeout = max(1, int(config.flow_captcha_service_timeout_seconds))

        payload: Dict[str, Any] = {
            "provider": cls.PROVIDER,
            "task_type": cls.TASK_TYPE,
            "page_action": page_action
        }
        if project_id:
            payload["project_id"] = project_id
        if website_url:
            payload["website_url"] = website_url

        headers = {
            "Authorization": f"Bearer {api_key}",
            "X-API-Key": api_key,
            "Content-Type": "application/json"
        }

        start = time.time()
        try:
            async with AsyncSession() as session:
                response = await session.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=timeout,
                    impersonate="chrome110"
                )
        except Exception as e:
            raise FlowCaptchaServiceError(
                f"request failed: {e}",
                status_code=502,
                error_type="upstream_error",
                error_code="captcha_upstream_error"
            ) from e

        local_duration_ms = int((time.time() - start) * 1000)

        try:
            response_payload = response.json()
        except Exception:
            response_payload = None

        if response.status_code >= 400:
            message = cls._extract_error_message(response_payload) or response.text[:300]
            raise FlowCaptchaServiceError(
                message or f"http {response.status_code}",
                status_code=502,
                error_type="upstream_error",
                error_code="captcha_upstream_error"
            )

        if not isinstance(response_payload, dict):
            raise FlowCaptchaServiceError(
                "invalid json response from flow captcha service",
                status_code=502,
                error_type="upstream_error",
                error_code="captcha_upstream_error"
            )

        upstream_error = cls._extract_error_message(response_payload)
        if upstream_error and not cls._extract_token(response_payload):
            raise FlowCaptchaServiceError(
                upstream_error,
                status_code=502,
                error_type="upstream_error",
                error_code="captcha_upstream_error"
            )

        token = cls._extract_token(response_payload)
        if not token:
            debug_logger.log_error(f"[FlowCaptchaService] missing token in response: {response_payload}")
            raise FlowCaptchaServiceError(
                "captcha token missing in upstream response",
                status_code=502,
                error_type="upstream_error",
                error_code="captcha_upstream_error"
            )

        duration_ms = cls._extract_int(response_payload, ["duration_ms", "durationMs", "duration"])
        browser_id = cls._extract_int(response_payload, ["browser_id", "browserId"])
        resolved_action = (
            response_payload.get("page_action")
            or response_payload.get("pageAction")
            or page_action
        )

        return {
            "name": cls.TASK_NAME,
            "object": "captcha.solution",
            "provider": cls.PROVIDER,
            "page_action": resolved_action,
            "token": token,
            "duration_ms": duration_ms if duration_ms is not None else local_duration_ms,
            "browser_id": browser_id if browser_id is not None else 0,
            "task_type": cls.TASK_TYPE,
            "pricing": {
                "currency": "CNY",
                "price_per_1000_tasks": 15.0,
                "price_per_task": 0.015,
                "points_per_task": 15.0
            }
        }
