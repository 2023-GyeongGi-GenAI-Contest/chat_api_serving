# [main.py]
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from typing import Optional
from pydantic import BaseModel
from Agent.custom_agent import get_reply

app = FastAPI()

origins = ["http://127.0.0.1:5000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")


@app.get("/chat", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class Message(BaseModel):
    message: str


@app.post("/response")
async def response(msg: Message):
    print('INPUT: ')
    print(msg.message)
    reply = get_reply(msg.message)
    return {"reply": reply}
