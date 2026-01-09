### voici la structure du projet

```bash
worker/
├── main.py            # boucle
├── worker.py          # process UN job
├── models/
│   └── job.py
├── infra/
│   ├── redis_streams.py
│   ├── redis_events.py
│   └── minio.py
├── processing/
│   ├── router.py
│   ├── das.py
│   ├── mv.py
│   └── ml.py
├── config.py
└── exceptions.py
```

Pour la logique tu peux retrouver la logique déjà fait dans le mock.py du dossier PROCOM_SIMULATEUR
tu peux aussi voir les fonctions, juste il faut organiser mieux que ca
Pour les variables d env je te laisse regarder mock.py

NB: le das n est pas encore coder laisse le en mode mock ou traite l erreur

Pour le code fait un code compréhensible sans etre trop lourd

### Variables d'environnement

- `REDIS_URL` (défaut: `redis://localhost:6379/0`)
- `REDIS_INPUT_STREAM` (défaut: `jobs-stream`)
- `REDIS_OUTPUT_STREAM` (défaut: `jobs-events`)
- `REDIS_SSE_CHANNEL` (défaut: `jobs:sse`)
- `REDIS_GROUP` (défaut: `ml-workers`)
- `REDIS_CONSUMER` (défaut: `worker-1`)
- `REDIS_READ_COUNT` (défaut: `1`)
- `REDIS_BLOCK_MS` (défaut: `5000`)
- `MINIO_ENDPOINT` (défaut: `localhost:9000`)
- `MINIO_ACCESS_KEY` (défaut: `admin`)
- `MINIO_SECRET_KEY` (défaut: `admin123`)
- `MINIO_SECURE` (défaut: `false`)
- `MINIO_BUCKET` (défaut: `uploads`)
- `LOG_LEVEL` (défaut: `INFO`)
- `DRY_RUN` (défaut: `false`)

### Job payload (Redis Streams)

Exemple minimum:

```json
{
  "jobId": "job-123",
  "userId": "user-42",
  "tasks": "ML,DAS,MV",
  "inputKey": "user-42/job-123/input/data.rf"
}
```

Champs reconnus pour l'input MinIO: `inputKey`, `input`, `objectKey`, `rfKey`.

### Assumptions

- Les tâches supportées sont `ML`, `DAS`, `MV`. Si `tasks` est vide, on exécute `ML`.
- `DAS` et `MV` nécessitent un `inputKey` (ou un alias) pour télécharger les données d'entrée.
- Les artefacts sont uploadés dans `minio://{MINIO_BUCKET}/{userId}/{jobId}/output/{TASK}/result.*`.
- TODO: si le format réel des payloads est différent, adapter la validation du job en conséquence.
