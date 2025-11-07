from fastapi import Request
from fastapi.responses import JSONResponse


class AppError(Exception):
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        self.message = message
        super().__init__(message)


async def handle_app_error(request: Request, exc: AppError):
    return JSONResponse(status_code=exc.status_code, content={"error": exc.message})
