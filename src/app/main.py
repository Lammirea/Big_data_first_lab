import sys
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn
import os

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

from func import *

SHOW_LOG = True

parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)

@app.post("/train/")
async def train_model(
    use_config: bool = True,
    max_depth: int = 10,
    min_samples_split: int = 2,
    predict_flag: bool = False
):
    return train_model_func(use_config, max_depth, min_samples_split, predict_flag)

@app.post("/predict/")
async def predict_model(mode: str = "smoke", file: UploadFile = None):
    if mode == "upload" and file:
        file_contents = await file.read()
        return predict_model_func(mode, file_contents)
    elif mode == "smoke":
        return predict_model_func(mode)
    else:
        raise HTTPException(status_code=400, detail="Неверные параметры запроса")

if __name__ == "__main__":
    config = configparser.ConfigParser()
    config_path = os.path.join(current_dir, '..', "config.ini")
    config.read(config_path, encoding="utf-8")
    try:
        host = config["FASTAPI"]["host"]
        port = config.getint("FASTAPI", "port")
    except KeyError:
        raise ValueError("В config.ini отсутствует секция [FASTAPI] или ключи host/port")
    uvicorn.run(app, host=host, port=port)