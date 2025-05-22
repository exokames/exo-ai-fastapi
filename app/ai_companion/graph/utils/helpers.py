import re

from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from app.core.config import settings
from app.modules.image.image_to_text import ImageToText
from app.modules.image.text_to_image import TextToImage
from app.modules.speech import TextToSpeech


def get_chat_model(temperature: float = 0.7):
    return ChatGoogleGenerativeAI(model=settings.TEXT_MODEL_NAME, temperature=temperature)


def get_text_to_speech_module():
    return TextToSpeech()


def get_text_to_image_module():
    return TextToImage()


def get_image_to_text_module():
    return ImageToText()


def remove_asterisk_content(text: str) -> str:
    """Remove content between asterisks from the text."""
    return re.sub(r"\*.*?\*", "", text).strip()


class AsteriskRemovalParser(StrOutputParser):
    def parse(self, text):
        return remove_asterisk_content(super().parse(text))
