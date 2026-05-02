from fastapi import FastAPI
from pydantic import BaseModel

from chain import load_chain, ask
from guardrails import InputGuardrails
from logger import log

app = FastAPI()

chain = load_chain()
guard = InputGuardrails()


class Query(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def chat(q: Query):

    ok, text = guard.validate(q.question)

    if not ok:
        return {"error": text}

    result = ask(chain, text)

    log(text, result["answer"], result["sources"])

    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000)