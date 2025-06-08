# Speech Extraction Service

A web interface for running speech extraction on video files from a surveillance camera.

## Features

- Web-based admin interface
- View folder processing status
- Manually trigger speech extraction
- Automatic daily extraction at 10:30 AM
- Real-time status updates

## Requirements

- Python 3.11+
- FFmpeg
- Whisper.cpp (for speech-to-text)
- Ollama with gemma3:4b model (for text processing)

## Installation

1. Clone this repository
2. Install dependencies using `uv`:
   ```bash
   uv pip install -e .
   ```

## Configuration

The service looks for video files in `/Volumes/surveillance/Bedroom Camera` by default. You can modify this in `serve.py` by changing the `BASE_DIR` variable.

## Running the Service

Start the web server:

```bash
python serve.py
```

The web interface will be available at http://localhost:8000

## Usage

1. **Dashboard** - View the current status and recent activity
2. **Run Extraction** - Click the "Run Extraction Now" button to manually start processing
3. **Folder Status** - View which folders have been processed
4. **Scheduled Tasks** - View the status of automated tasks

## API Endpoints

- `GET /` - Web interface
- `GET /api/status` - Get current extraction status
- `POST /api/extract` - Trigger manual extraction
- `GET /api/folders` - List folders and processing status

## Scheduled Tasks

The service automatically runs the extraction process every day at 10:30 AM. This can be modified in `serve.py` by changing the cron schedule.
