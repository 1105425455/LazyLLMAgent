import os
import time

if "APIKEY" not in os.environ:
    with open("api_key.env", "r", encoding="utf-8") as f:
        os.environ["APIKEY"] = f.readline().strip()

APIKEY = os.getenv("APIKEY")