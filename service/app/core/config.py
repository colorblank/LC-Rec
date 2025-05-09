import os


class ConfigurationError(Exception):
    pass


class Settings:
    RQVAE_CKPT_PATH: str
    RQVAE_DATA_PATH: str | None  # Can be None if data_path is in checkpoint
    TEXT_EMBEDDING_MODEL_NAME: str

    def __init__(self):
        self.RQVAE_CKPT_PATH = os.getenv("RQVAE_CKPT_PATH")
        if not self.RQVAE_CKPT_PATH:
            raise ConfigurationError(
                "RQVAE_CKPT_PATH environment variable not set. "
                "Please provide the path to the RQVAE "
                "model checkpoint."
            )

        # RQVAE_DATA_PATH is optional if data_path is in the checkpoint,
        # will be validated later during model loading.
        self.RQVAE_DATA_PATH = os.getenv("RQVAE_DATA_PATH")

        self.TEXT_EMBEDDING_MODEL_NAME = os.getenv(
            "TEXT_EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )


settings = Settings()
