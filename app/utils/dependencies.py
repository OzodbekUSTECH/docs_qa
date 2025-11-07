from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from app.core.config import settings

security = HTTPBasic()


def get_current_user_for_docs(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = settings.DOCS_USERNAME
    correct_password = settings.DOCS_PASSWORD
    if (
        credentials.username != correct_username
        or credentials.password != correct_password
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

