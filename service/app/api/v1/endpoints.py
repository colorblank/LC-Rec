from fastapi import APIRouter, Request, HTTPException
import torch

from app.models.schemas import BatchVectorInput, SingleTextItem, BatchTextItems
from app.services.rqvae_processing import get_rqvae_indices_from_embeddings

router = APIRouter()


@router.post("/predict")
async def predict_from_vectors(input_data: BatchVectorInput, request: Request):
    try:
        rqvae_model_instance = request.app.state.rqvae_model

        if (
            not input_data.data
            or not isinstance(input_data.data, list)
            or not all(isinstance(vec, list) for vec in input_data.data)
        ):
            raise HTTPException(
                status_code=400,
                detail="Invalid input data format. Expected a list of vectors (list of lists of floats).",
            )

        if input_data.data:
            first_vec_dim = len(input_data.data[0])
            if not all(len(vec) == first_vec_dim for vec in input_data.data):
                raise HTTPException(
                    status_code=400,
                    detail="All vectors in the batch must have the same dimension.",
                )
            if first_vec_dim != rqvae_model_instance.in_dim:
                raise HTTPException(
                    status_code=400,
                    detail=f"Input vector dimension ({first_vec_dim}) does not match RQVAE model's expected input dimension ({rqvae_model_instance.in_dim}).",
                )

        data_tensor = torch.tensor(input_data.data, dtype=torch.float32)
        indices_list = await get_rqvae_indices_from_embeddings(
            data_tensor, rqvae_model_instance
        )
        return {"indices": indices_list}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing vectors: {str(e)}"
        )


@router.post("/embed_text_single")
async def embed_single_text(item: SingleTextItem, request: Request):
    try:
        text_embedding_model_instance = request.app.state.text_embedding_model
        rqvae_model_instance = request.app.state.rqvae_model

        embedding_np = text_embedding_model_instance.encode(item.text)
        embedding_tensor = torch.tensor(embedding_np, dtype=torch.float32)

        indices_list = await get_rqvae_indices_from_embeddings(
            embedding_tensor, rqvae_model_instance
        )
        return {"indices": indices_list[0] if indices_list else []}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing single text item: {str(e)}"
        )


@router.post("/embed_texts_batch")
async def embed_batch_texts(items: BatchTextItems, request: Request):
    try:
        text_embedding_model_instance = request.app.state.text_embedding_model
        rqvae_model_instance = request.app.state.rqvae_model

        if not items.texts:
            return {"indices": []}

        embeddings_np = text_embedding_model_instance.encode(items.texts)
        embeddings_tensor = torch.tensor(embeddings_np, dtype=torch.float32)

        indices_list = await get_rqvae_indices_from_embeddings(
            embeddings_tensor, rqvae_model_instance
        )
        return {"indices": indices_list}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing batch text items: {str(e)}"
        )
