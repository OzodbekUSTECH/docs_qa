from fastapi import FastAPI
from app.exceptions.app_error import AppError, handle_app_error
from app.exceptions.unhandled_error import handle_unhandled_error



def register_exceptions(app: FastAPI):
    app.add_exception_handler(AppError, handle_app_error)
    app.add_exception_handler(Exception, handle_unhandled_error)
