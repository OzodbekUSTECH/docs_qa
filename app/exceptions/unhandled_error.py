from fastapi import Request
from fastapi.responses import JSONResponse
from app.exceptions.messages import ErrorMessages


async def handle_unhandled_error(request: Request, exc: Exception):
    # sentry_sdk.capture_exception(exc)
    return JSONResponse(
        status_code=500, content={"error": ErrorMessages.INTERNAL_SERVER_ERROR}
    )
