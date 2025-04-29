from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from data import EmbDataset
from models.rqvae import RQVAE

app = FastAPI()


class InputData(BaseModel):
    data: list


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    ckpt_path = "path/to/checkpoint"
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"))
    args = ckpt["args"]
    state_dict = ckpt["state_dict"]

    data = EmbDataset(args.data_path)

    model = RQVAE(
        in_dim=data.dim,
        num_emb_list=args.num_emb_list,
        e_dim=args.e_dim,
        layers=args.layers,
        dropout_prob=args.dropout_prob,
        bn=args.bn,
        loss_type=args.loss_type,
        quant_loss_weight=args.quant_loss_weight,
        kmeans_init=args.kmeans_init,
        kmeans_iters=args.kmeans_iters,
        sk_epsilons=args.sk_epsilons,
        sk_iters=args.sk_iters,
    )

    model.load_state_dict(state_dict)
    model.eval()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/predict")
async def predict(input_data: InputData):
    try:
        data_tensor = torch.tensor(input_data.data)
        indices = model.get_indices(data_tensor, use_sk=False)
        indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
        return {"indices": indices.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
