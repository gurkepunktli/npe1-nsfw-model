# NSFW Detector API

Docker-basierte API fuer die Erkennung von NSFW-Inhalten in Bildern mit dem [LAION CLIP-based NSFW Detector](https://github.com/LAION-AI/CLIP-based-NSFW-Detector).

## Features
- FastAPI REST-API
- Einzel- und Batch-Analyse
- Festes CLIP-Modell ViT-L/14 (kein Parameter notwendig)
- Bild per Datei-Upload **oder** direkter URL
- Health-Check Endpoint
- Docker & Portainer ready mit persistentem Model-Cache

## Installation & Deployment

### Voraussetzungen
- Docker & Docker Compose installiert
- Portainer (optional)
- Bestehendes Docker-Netzwerk `cloudflare_net`

### Deployment mit Portainer
1. Stack in Portainer anlegen: **Stacks** -> **Add stack**
2. Name: `nsfw-detector`
3. Build method: Repository oder Upload
4. Repository URL + `docker-compose.yml` angeben **oder** Dateien hochladen
5. Deploy

### Deployment via Command Line
```bash
# Repository klonen oder in das Projektverzeichnis wechseln
cd /pfad/zum/projekt

# Container bauen und starten
docker-compose up -d --build

# Logs anzeigen
docker-compose logs -f nsfw-detector

# Status pruefen
docker-compose ps
```

## API Verwendung

### Endpunkte

#### 1. Health Check
```bash
GET http://localhost:8101/health
```

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

#### 2. Einzelbild analysieren
```bash
POST http://localhost:8101/analyze
```

**Parameter:**
- `file` (optional): Bilddatei (multipart/form-data)
- `image_url` (optional): Direkte Bild-URL (multipart/form oder Query)
- Es darf genau **eine** der beiden Quellen gesetzt sein.

**Beispiel mit cURL:**
```bash
# Bild per URL
curl -X POST "http://localhost:8101/analyze" \
  -F "image_url=https://example.com/test.jpg"

# Alternativ per Datei-Upload
curl -X POST "http://localhost:8101/analyze" \
  -F "file=@/path/to/image.jpg"
```

**Response:**
```json
{
  "nsfw_score": 0.8234
}
```

#### 3. Mehrere Bilder analysieren (Batch)
```bash
POST http://localhost:8101/batch-analyze
```

**Parameter:**
- `files`: Mehrere Bilddateien (multipart/form-data)
- `threshold`: NSFW-Schwellenwert (default: 0.5)

**Beispiel mit cURL:**
```bash
curl -X POST "http://localhost:8101/batch-analyze?threshold=0.5" \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

**Response:**
```json
{
  "results": [
    {
      "filename": "image1.jpg",
      "nsfw_score": 0.1234,
      "is_nsfw": false,
      "threshold": 0.5,
      "message": "Safe content"
    },
    {
      "filename": "image2.jpg",
      "nsfw_score": 0.8765,
      "is_nsfw": true,
      "threshold": 0.5,
      "message": "NSFW content detected"
    }
  ],
  "total": 2,
  "processed": 2
}
```

### API Dokumentation
- Swagger UI: `http://localhost:8101/docs`
- ReDoc: `http://localhost:8101/redoc`

## Python Beispiel

```python
import requests

def analyze_image(image_source):
    url = "http://localhost:8101/analyze"

    if image_source.startswith("http"):
        response = requests.post(url, data={"image_url": image_source})
    else:
        with open(image_source, "rb") as f:
            response = requests.post(url, files={"file": f})

    response.raise_for_status()
    return response.json()


result = analyze_image("test_image.jpg")
print(f"NSFW Score: {result['nsfw_score']}")
```

## Konfiguration

### Umgebungsvariablen (docker-compose.yml)
- `TZ`: Zeitzone (default: Europe/Berlin)
- `PYTHONUNBUFFERED`: Python Buffering (default: 1)

### Ressourcen-Limits (docker-compose.yml)
- Limit: 4GB
- Reservation: 2GB

### Port-Konfiguration
- Standard: `8101`
- Anpassen in `docker-compose.yml` (z. B. `8080:8101`)

## CLIP Modell

Die API nutzt fest das Modell **ViT-L/14** (Embedding-Dimension: 768) fuer hohe Genauigkeit. Eine Modell-Auswahl ist nicht notwendig.

## Troubleshooting

- Logs: `docker-compose logs nsfw-detector`
- Neu bauen: `docker-compose up -d --build --force-recreate`
- Volume reset: `docker-compose down -v` danach neu starten
- Netzwerk pruefen: `docker network ls | grep cloudflare_net` (ggf. `docker network create cloudflare_net`)
- Memory erhoehen (docker-compose.yml):
  ```yaml
  deploy:
    resources:
      limits:
        memory: 8G
  ```

## Technische Details
- **Framework**: FastAPI
- **ML-Backend**: TensorFlow + AutoKeras
- **CLIP-Modell**: OpenCLIP (ViT-L/14)
- **Web-Server**: Uvicorn
- Modelle werden beim ersten Start geladen und im Volume `nsfw-models` gecached (ViT-L/14: ~100MB)
- Performance: erste Anfrage ca. 5-10s (Model-Load), danach ~200-500ms pro Bild; Batch ~100-300ms pro Bild (abh. vom Batch)

## Lizenz

Basierend auf [LAION CLIP-based NSFW Detector](https://github.com/LAION-AI/CLIP-based-NSFW-Detector) (MIT License)

## Quellen

- [GitHub - LAION-AI/CLIP-based-NSFW-Detector](https://github.com/LAION-AI/CLIP-based-NSFW-Detector)
- [CLIP-based-NSFW-Detector README](https://github.com/LAION-AI/CLIP-based-NSFW-Detector/blob/main/README.md)
- [Medium Article: CLIP-based NSFW Detector](https://medium.com/axinc-ai/clip-based-nsfw-detector-ai-model-that-can-detect-inappropriate-images-d84a4bc50972)
