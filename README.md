# NSFW Detector API

Docker-basierte API für die Erkennung von NSFW-Inhalten in Bildern mittels [LAION CLIP-based NSFW Detector](https://github.com/LAION-AI/CLIP-based-NSFW-Detector).

## Features

- RESTful API mit FastAPI
- NSFW-Erkennung für einzelne Bilder
- Batch-Verarbeitung mehrerer Bilder
- Unterstützung für CLIP ViT-L/14 und ViT-B/32 Modelle
- Health-Check Endpoint
- Docker & Portainer ready
- Persistente Modell-Speicherung via Docker Volumes

## Installation & Deployment

### Voraussetzungen

- Docker & Docker Compose installiert
- Portainer (optional)
- Bestehendes Docker-Netzwerk `cloudflare_net`

### Deployment mit Portainer

1. **Stack erstellen in Portainer:**
   - Navigiere zu "Stacks" → "Add stack"
   - Name: `nsfw-detector`
   - Build method: "Repository" oder "Upload"

2. **Via Repository:**
   - Repository URL eingeben
   - Compose path: `docker-compose.yml`
   - Deploy

3. **Via Upload:**
   - Repository klonen oder Dateien herunterladen
   - `docker-compose.yml` hochladen
   - Deploy

### Deployment via Command Line

```bash
# Repository klonen oder in Projekt-Verzeichnis wechseln
cd /pfad/zum/projekt

# Container bauen und starten
docker-compose up -d --build

# Logs anzeigen
docker-compose logs -f nsfw-detector

# Status prüfen
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
- `file`: Bilddatei (multipart/form-data)
- `threshold`: NSFW-Schwellenwert (0.0 - 1.0, default: 0.5)
- `clip_model`: CLIP-Modell ("ViT-L/14" oder "ViT-B/32", default: "ViT-L/14")

**Beispiel mit cURL:**
```bash
curl -X POST "http://localhost:8101/analyze?threshold=0.5&clip_model=ViT-L/14" \
  -F "file=@/path/to/image.jpg"
```

**Response:**
```json
{
  "nsfw_score": 0.8234,
  "is_nsfw": true,
  "threshold": 0.5,
  "message": "NSFW content detected"
}
```

#### 3. Mehrere Bilder analysieren (Batch)
```bash
POST http://localhost:8101/batch-analyze
```

**Parameter:**
- `files`: Multiple Bilddateien (multipart/form-data)
- `threshold`: NSFW-Schwellenwert (default: 0.5)
- `clip_model`: CLIP-Modell (default: "ViT-L/14")

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

Interaktive API-Dokumentation (Swagger UI) verfügbar unter:
```
http://localhost:8101/docs
```

Alternative Dokumentation (ReDoc):
```
http://localhost:8101/redoc
```

## Python Beispiel

```python
import requests

# Einzelbild analysieren
def analyze_image(image_path, threshold=0.5):
    url = "http://localhost:8101/analyze"
    files = {"file": open(image_path, "rb")}
    params = {"threshold": threshold, "clip_model": "ViT-L/14"}

    response = requests.post(url, files=files, params=params)
    return response.json()

# Verwendung
result = analyze_image("test_image.jpg", threshold=0.5)
print(f"NSFW Score: {result['nsfw_score']}")
print(f"Is NSFW: {result['is_nsfw']}")
print(f"Message: {result['message']}")
```

## Konfiguration

### Umgebungsvariablen

In [docker-compose.yml](docker-compose.yml) können folgende Variablen angepasst werden:

- `TZ`: Zeitzone (default: Europe/Berlin)
- `PYTHONUNBUFFERED`: Python Buffering (default: 1)

### Ressourcen-Limits

Memory Limits in [docker-compose.yml](docker-compose.yml):
- Limit: 4GB
- Reservation: 2GB

Bei Bedarf anpassen für bessere Performance oder niedrigere Ressourcennutzung.

### Port-Konfiguration

Standard-Port: `8101`

Ändern in [docker-compose.yml](docker-compose.yml):
```yaml
ports:
  - "8080:8101"  # Host:Container
```

## CLIP Modelle

### ViT-L/14 (Standard)
- Embedding-Dimension: 768
- Höhere Genauigkeit
- Mehr Ressourcen erforderlich

### ViT-B/32
- Embedding-Dimension: 512
- Schnellere Verarbeitung
- Geringere Ressourcennutzung

## Troubleshooting

### Container startet nicht
```bash
# Logs prüfen
docker-compose logs nsfw-detector

# Container neu bauen
docker-compose up -d --build --force-recreate
```

### Modell-Download schlägt fehl
```bash
# Volume löschen und neu starten
docker-compose down -v
docker-compose up -d --build
```

### Netzwerk nicht gefunden
```bash
# Prüfen ob cloudflare_net existiert
docker network ls | grep cloudflare_net

# Netzwerk erstellen (falls nicht vorhanden)
docker network create cloudflare_net
```

### Memory Fehler
Ressourcen-Limits in [docker-compose.yml](docker-compose.yml) erhöhen:
```yaml
deploy:
  resources:
    limits:
      memory: 8G
```

## Technische Details

### Architektur
- **Framework**: FastAPI
- **ML-Backend**: TensorFlow + AutoKeras
- **CLIP-Modell**: OpenCLIP
- **Web-Server**: Uvicorn

### Modell-Caching
Modelle werden beim ersten Start heruntergeladen und im Volume `nsfw-models` gespeichert:
- ViT-L/14: ~100MB
- ViT-B/32: ~70MB

### Performance
- Erste Anfrage: ~5-10s (Modell-Laden)
- Folgende Anfragen: ~200-500ms pro Bild
- Batch-Processing: ~100-300ms pro Bild (abhängig von Batch-Größe)

## Lizenz

Basierend auf [LAION CLIP-based NSFW Detector](https://github.com/LAION-AI/CLIP-based-NSFW-Detector) (MIT License)

## Quellen

- [GitHub - LAION-AI/CLIP-based-NSFW-Detector](https://github.com/LAION-AI/CLIP-based-NSFW-Detector)
- [CLIP-based-NSFW-Detector README](https://github.com/LAION-AI/CLIP-based-NSFW-Detector/blob/main/README.md)
- [Medium Article: CLIP-based NSFW Detector](https://medium.com/axinc-ai/clip-based-nsfw-detector-ai-model-that-can-detect-inappropriate-images-d84a4bc50972)
