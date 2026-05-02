import json
from datetime import datetime

LOG_FILE = "logs.jsonl"


def log(q, a, sources):

    data = {
        "time": str(datetime.utcnow()),
        "question": q,
        "answer": a[:300],
        "sources": sources
    }

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(data) + "\n")