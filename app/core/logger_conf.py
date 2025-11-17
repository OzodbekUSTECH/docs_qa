import logging


def configure_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    
    # Suppress WARNING messages from google.genai about non-text parts (executable_code)
    # This is just informational - SDK handles it automatically
    google_genai_logger = logging.getLogger("google.genai")
    google_genai_logger.setLevel(logging.ERROR)  # Only show ERROR and above