import json
from datetime import datetime
from pathlib import Path


def now_ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_writer(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fp = log_path.open("w", encoding="utf-8")

    def _log(level: str, event: str, message: str, data=None):
        record = {
            "ts": now_ts(),
            "level": level,
            "event": event,
            "message": message,
        }
        if data is not None:
            record["data"] = data
        line = json.dumps(record, ensure_ascii=False)
        print(line)
        fp.write(line + "\n")
        fp.flush()

    return _log, fp
