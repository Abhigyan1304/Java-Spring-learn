import os
from app.helpers.s3_utils import download_folder
from app.helpers.kb_loader import load_all_kbs
from app.logger import logger

KB_BUCKET = os.getenv("KB_BUCKET", "my-kb-bucket")
MODEL_BUCKET = os.getenv("MODEL_BUCKET", "my-model-bucket")

KB_PREFIX = "kb/"
MODEL_PREFIX = "models/"
LOCAL_KB_DIR = "kb/"
LOCAL_MODEL_DIR = "models/"

async def on_startup():
    """
    Download models + KBs from S3 on startup, then load KBs into memory.
    """
    logger.info("🚀 Service starting up...")

    # Download knowledge base
    download_folder(KB_BUCKET, KB_PREFIX, LOCAL_KB_DIR)

    # Download models (main + embedding)
    download_folder(MODEL_BUCKET, MODEL_PREFIX, LOCAL_MODEL_DIR)

    # Load KB into memory
    load_all_kbs(LOCAL_KB_DIR)

    logger.info("✅ Startup complete: KB + Models loaded.")


async def on_shutdown():
    logger.info("🛑 Shutting down service...")
