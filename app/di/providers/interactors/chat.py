from dishka import Provider, Scope, provide_all

from app.interactors.chat.gemini_agent import GeminiAgentInteractor
from app.interactors.chat.generate import GenerateChatResponseInteractor

class ChatInteractorProvider(Provider):

    scope = Scope.REQUEST

    interactors = provide_all(
        GenerateChatResponseInteractor,
        GeminiAgentInteractor,
    )

