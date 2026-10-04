# supermemory

Self-hosted Supermemory-compatible vector memory API (FastAPI + PostgreSQL + pgvector).

Implements the subset of the cloud Supermemory API that Hermes / external agents use:
`GET /health`, `POST /v3/documents[/batch]`, `GET /v3/documents`, `POST /v3/search`,
`POST /v4/search`, `POST /v4/profile`, `DELETE /v4/memories`, settings endpoints.

## Build (multi-arch, GHCR)

```bash
export CR_PAT=...   # classic PAT with write:packages (or fine-grained: Packages write on this repo)
echo $CR_PAT | docker login ghcr.io -u nahdd123uosackr --password-stdin
docker buildx build --platform linux/arm64,linux/amd64 \
  -t ghcr.io/nahdd123uosackr/supermemory:latest --push .
```

## Deploy

Image is referenced in `/root/etc/kube/yaml/supermemory.yaml` (GitHub: nahdd123uosackr/etc).
Apply: `kubectl apply -f /root/etc/kube/yaml/supermemory.yaml`
