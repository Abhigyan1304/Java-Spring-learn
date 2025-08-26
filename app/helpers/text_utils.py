import re

def clean_text(text: str) -> str:
    """
    Simple cleaner for user input or retrieved docs.
    """
    return re.sub(r"\s+", " ", text).strip()
