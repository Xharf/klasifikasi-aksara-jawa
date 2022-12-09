from typing import Union

from fastapi import FastAPI

from pydantic import BaseModel

import numpy as np
import tensorflow as tf
from PIL import Image
from urllib import request
import io
from keras.utils import img_to_array
from preprocessing import preprocessing
from aksara_models import AksaraJawaModels

app = FastAPI()
# coba


class Item2(BaseModel):
    imgUrl: str


models = AksaraJawaModels()


@app.post("/predict")
def getPrediction(item: Item2):
    try:
        if item.imgUrl:
            url = request.urlopen(f'{item.imgUrl}').read()
            img = Image.open(io.BytesIO(url))
            images = preprocessing(img)

            return models.aksara_predict(images)
        else:
            return {
                "code": 400,
                "message": "Gambar tidak boleh kosong",
            }
    except Exception as e:
        return {
            "code": 500,
            "message": str(e),
        }


@app.post("/*")
def notFound():
    return {
        "code": 404,
        "message": "Endpoint tidak ditemukan",
    }
