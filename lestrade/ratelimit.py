import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from . import config

logger = logging.getLogger(__name__)

_store: dict[str, list[float]] = {}


class RateLimitMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if not request.url.path.endswith('/chat/completions'):
            return await call_next(request)

        window = config.RATE_LIMIT_WINDOW
        max_req = config.RATE_LIMIT_MAX_REQUESTS
        client_ip = request.client.host if request.client else 'unknown'
        now = time.time()

        if client_ip in _store:
            _store[client_ip] = [t for t in _store[client_ip] if now - t < window]
        else:
            _store[client_ip] = []

        if len(_store[client_ip]) >= max_req:
            retry_after = int(window - (now - _store[client_ip][0]))
            logger.warning('Rate limit hit for %s (%d requests in %ds)',
                           client_ip, max_req, window)
            return JSONResponse(
                status_code=429,
                content={'error': 'Too many requests', 'retry_after': retry_after},
                headers={'Retry-After': str(retry_after)},
            )

        _store[client_ip].append(now)
        return await call_next(request)
