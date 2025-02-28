from func import *
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn


SHOW_LOG = True

app = FastAPI(title="CatBoostClassifier API для CICIDS2017")


@app.post("/train")
async def train_endpoint(use_config: bool = False, iterations: int = 1000, depth: int = 6, learning_rate: float = 0.1, predict: bool = False):
    """
    Эндпоинт для обучения модели.
    Параметры (use_config, iterations, depth, learning_rate, predict) можно передать в запросе.
    """
    result = train_model_func(use_config, iterations, depth, learning_rate, predict)
    return {"message": "Модель успешно обучена", **result}

@app.post("/predict")
async def predict_endpoint(mode: str = "smoke", file: UploadFile = File(None)):
    """
    Эндпоинт для предсказания.
    Query-параметр mode определяет режим:
      - smoke: тестирование на данных, указаных в config.ini (SPLIT_DATA)
      - upload: предсказание на загруженном CSV-файле (без столбца 'Label')
    Для режима upload необходимо передать файл.
    """
    file_contents = None
    if mode == "upload":
        if file is None:
            raise HTTPException(status_code=400, detail="Для режима upload требуется загрузить файл")
        file_contents = await file.read()
    result = predict_model_func(mode, file_contents)
    return result

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
