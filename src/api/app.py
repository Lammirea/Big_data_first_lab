from func import *
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn


SHOW_LOG = True

app = FastAPI(title="RandomForest API для CICIDS2017")

# Эндпоинты FastAPI
@app.post("/train/")
async def train_model(
    use_config: bool = False,
    n_estimators: int = 100,
    max_depth: int = None,
    min_samples_split: int = 2,
    predict_flag: bool = False
):
    return train_model_func(use_config, n_estimators, max_depth, min_samples_split, predict_flag)

@app.post("/predict/")
async def predict_model(mode: str, file: UploadFile = None):
    if mode == "upload" and file:
        file_contents = await file.read()
        return predict_model_func(mode, file_contents)
    elif mode == "smoke":
        return predict_model_func(mode)
    else:
        raise HTTPException(status_code=400, detail="Неверные параметры запроса")

if __name__ == "__main__":
    uvicorn.run("app:app", host="192.168.1.77", port=8000, reload=True)
