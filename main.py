from typing import Union

from fastapi import FastAPI, File, UploadFile

from pydantic import BaseModel

import numpy as np
import tensorflow as tf
from PIL import Image
from urllib import request
import io
from keras.utils import img_to_array
from preprocessing import preprocessing
from aksara_models import AksaraJawaModels
import base64
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
# coba


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


class Base64img(BaseModel):
    imgbase64: str


@app.post("/predict-image")
def getPredictionImage(item: Base64img):
    try:
        if item.imgbase64:
            image_b64 = item.imgbase64.split(",")[1]
            binary = base64.b64decode(image_b64)
            img = Image.open(io.BytesIO(binary))
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


@ app.post("/*")
def notFound():
    return {
        "code": 404,
        "message": "Endpoint tidak ditemukan",
    }
