from fastapi import FastAPI
from contextlib import asynccontextmanager

from app.api.v1 import endpoints
from app.core.model_loader import (
    load_text_embedding_model,
    load_rqvae_model,
    ConfigurationError,
)  # Updated import
# Assuming models.rqvae will be in PYTHONPATH or adjusted relative import


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting application lifespan...")
    try:
        # Load text embedding model first
        text_embedding_model, text_model_output_dim = load_text_embedding_model()
        app.state.text_embedding_model = text_embedding_model
        print(f"Text embedding model loaded. Output dimension: {text_model_output_dim}")

        # Load RQVAE model using the output dimension from the text model
        rqvae_model = load_rqvae_model(in_dimension=text_model_output_dim)
        app.state.rqvae_model = rqvae_model
        print("RQVAE model loaded successfully.")

        print("All models loaded successfully and stored in app.state.")
    except ConfigurationError as e:
        # Log this error appropriately in a real application
        print(f"Configuration Error during startup: {e}")
        # Depending on the severity, you might want to prevent the app from starting
        # or let it start in a degraded state. For now, we'll print and continue.
        # To stop the app, you could re-raise the exception or exit.
        raise RuntimeError(f"Failed to initialize models due to configuration: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during model loading: {e}")
        raise RuntimeError(
            f"Failed to initialize models due to an unexpected error: {e}"
        )

    yield

    # Clean up resources if needed on shutdown
    print("Application shutdown...")
    app.state.rqvae_model = None
    app.state.text_embedding_model = None
    print("Cleaned up models from app.state.")


app = FastAPI(lifespan=lifespan, title="RQVAE Model Service")

app.include_router(endpoints.router, prefix="/api/v1")


@app.get("/health")
async def health_check():
    # Basic health check, can be expanded to check model status etc.
    model_status = (
        "loaded"
        if hasattr(app.state, "rqvae_model") and app.state.rqvae_model
        else "not loaded"
    )
    return {
        "status": "ok",
        "message": "Service is running",
        "rqvae_model_status": model_status,
    }


# If you want to run this directly using uvicorn for development:
# import uvicorn
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
