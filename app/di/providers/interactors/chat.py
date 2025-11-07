from dishka import Provider, Scope, provide_all

from app.interactors.chat.generate import GenerateChatResponseInteractor


class ChatInteractorProvider(Provider):

    scope = Scope.REQUEST

    interactors = provide_all(
        GenerateChatResponseInteractor,
    )

