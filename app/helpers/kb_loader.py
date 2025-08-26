import os
from app.services.rag_service import load_kb
from app.logger import logger

def load_all_kbs(kb_root: str):
    """
    Load KBs for India, Philippines, Thailand.
    """
    for country in ["india", "philippines", "thailand"]:
        kb_file = os.path.join(kb_root, country, "rag_kb.json")
        if os.path.exists(kb_file):
            load_kb(country.upper(), kb_file)
        else:
            logger.warning(f"KB missing for {country}: {kb_file}")
