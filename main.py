from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Outfit_detect.predict import image_utils
from pydantic import BaseModel
import json, requests

app = FastAPI(title="Santosh",
                description="API endpoints for Santosh",
                version="1.0")
origins = [
    "*"
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# --------------------------

# Pydantic Models

# class Output(BaseModel):
#     Gender: str
#     Age: str
#     Wearing: str

# --------------------------------


@app.get('/')
async def home():
    return "Server is up and running!"

@app.get('/api/predict/{name}')
async def prediction(name: str):
    res = image_utils(f'media/{name}')
    return res


    