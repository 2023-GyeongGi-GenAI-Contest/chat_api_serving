# [main.py]
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from typing import Optional
from pydantic import BaseModel
import Agent.base_agent
import Similarity.check_similarity

# uvicorn main:app --host 127.0.0.1 --port 5000
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
    reply = Agent.base_agent.get_reply(msg.message)
    return {"reply": reply}

@app.post("/similarity")
async def similarity(msg: Message):
    print('INPUT: ')
    print(msg.message)
    reply = Similarity.check_similarity.get_reply(msg.message)
    return {"reply": reply}
