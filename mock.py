import time
import json
import os
from io import BytesIO

import redis
from minio import Minio
from minio.error import S3Error

from das_processor import DasProcessingError, process_das_task
from mv_processor import MVProcessingError, process_mv_task

# =====================
# CONFIG
# =====================

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

INPUT_STREAM = "jobs-stream"
OUTPUT_STREAM = "jobs-events"  # stream consommé par Node (DB)

SSE_CHANNEL = "jobs:sse"  # pub/sub consommé par Node (SSE)

GROUP = "ml-workers"
CONSUMER = "worker-1"

BUCKET = "uploads"

MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "admin123")
MINIO_SECURE = False

VALID_TASKS = {"ML", "DAS", "MV"}

# =====================
# CLIENTS
# =====================

redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

minio_client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=MINIO_SECURE,
)

# =====================
# INIT
# =====================


def ensure_bucket():
    if not minio_client.bucket_exists(BUCKET):
        minio_client.make_bucket(BUCKET)
        print(f"🪣 Bucket '{BUCKET}' created")


def ensure_consumer_group():
    try:
        redis_client.xgroup_create(
            INPUT_STREAM,
            GROUP,
            id="0",
            mkstream=True,
        )
        print(f"✔ Consumer group '{GROUP}' created")
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" in str(e):
            print(f"ℹ Consumer group '{GROUP}' already exists")
        else:
            raise


# =====================
# MOCK JOB PROCESSING
# =====================


def parse_tasks(payload: dict):
    raw = payload.get("tasks") or ""
    arr = [t for t in raw.split(",") if t]
    normalized = []
    for t in arr:
        t_norm = t.upper()
        if t_norm in VALID_TASKS:
            normalized.append(t_norm)
    return normalized or ["ML"]  # compat: défaut ML


def process_job(payload: dict):
    job_id = payload["jobId"]
    user_id = payload["userId"]
    tasks = parse_tasks(payload)

    print(f"⚙ Processing job {job_id} for user {user_id} tasks={tasks}")

    # ---- Simule un traitement ----
    time.sleep(1)

    outputs = {}

    for task in tasks:
        if task == "DAS":
            print("▶ Running Delay-and-Sum processing")
            try:
                das_outputs = process_das_task(
                    minio_client=minio_client,
                    bucket=BUCKET,
                    payload=payload,
                )
            except DasProcessingError as e:
                print(f"❌ DAS processing failed: {e}")
                raise
            else:
                outputs[task] = das_outputs["png"]
                outputs[f"{task}_png"] = das_outputs["png"]
                print(f"📦 DAS output uploaded → {das_outputs}")
                continue

        if task == "MV":
            print("▶ Running MV beamforming processing")
            try:
                mv_outputs = process_mv_task(
                    minio_client=minio_client,
                    bucket=BUCKET,
                    payload=payload,
                )
            except MVProcessingError as e:
                print(f"❌ MV processing failed: {e}")
                raise
            else:
                outputs[task] = mv_outputs["png"]
                outputs[f"{task}_png"] = mv_outputs["png"]
                print(f"📦 MV output uploaded → {mv_outputs}")
                continue

        fake_output = BytesIO()
        fake_output.write(f"MOCK RESULT {task}\n".encode())
        fake_output.seek(0)

        object_key = f"{user_id}/{job_id}/output/{task}/result.rf"

        minio_client.put_object(
            BUCKET,
            object_key,
            fake_output,
            length=fake_output.getbuffer().nbytes,
            content_type="application/octet-stream",
        )

        outputs[task] = object_key
        print(f"📦 Output {task} uploaded → {object_key}")

    # =========================
    # 1️⃣ EVENT DURABLE (STREAM)
    # =========================
    redis_client.xadd(
        OUTPUT_STREAM,
        {
            "jobId": job_id,
            "userId": user_id,
            "status": "done",
            "outputs": json.dumps(outputs),  # contrat Node v2
        },
    )

    # =========================
    # 2️⃣ EVENT VOLATIL (PUB/SUB)
    # =========================
    nb = redis_client.publish(
        SSE_CHANNEL,
        json.dumps(
            {
                "type": "jobUpdate",
                "userId": user_id,
                "jobId": job_id,
                "status": "done",
            }
        ),
    )

    print(f"📣 Pub/Sub published to {nb} subscriber(s)")


# =====================
# MAIN LOOP
# =====================


def main():
    ensure_bucket()
    ensure_consumer_group()

    print("🚀 Mock worker started, waiting for jobs...")

    while True:
        resp = redis_client.xreadgroup(
            GROUP,
            CONSUMER,
            {INPUT_STREAM: ">"},
            count=1,
            block=5000,
        )

        if not resp:
            continue

        for stream, messages in resp:
            for message_id, fields in messages:
                try:
                    process_job(fields)
                    redis_client.xack(INPUT_STREAM, GROUP, message_id)
                except Exception as e:
                    print("❌ Error processing job:", e)
                    # pas d'ACK → retry automatique


if __name__ == "__main__":
    main()
