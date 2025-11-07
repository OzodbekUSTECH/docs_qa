from dishka import make_async_container
from dishka.integrations.fastapi import FastapiProvider

from app.di.providers.db import DBProvider
from app.di.providers.repositories import RepositoriesProvider
from app.di.providers.interactors import all_interactors
from app.di.providers.utils import UtilsProvider
from app.di.providers.services import ServicesProvider


def create_app_container():
    providers = [
        FastapiProvider(),
        DBProvider(),
        RepositoriesProvider(),
        UtilsProvider(),
        ServicesProvider(),
        *all_interactors,
    ]
    return make_async_container(*providers)

app_container = create_app_container()