# RBD - Speech Extraction Service

## Overview
This project is a Python-based speech extraction service designed to process surveillance camera footage. It automatically extracts voice clips from video files, transcribes them using Whisper.cpp, and provides a web interface for monitoring and managing the extraction process.

## Project Structure
```
rbd/
├── extract_voice_clips.py  # Core extraction logic
├── serve.py               # FastAPI web server
├── pyproject.toml         # Project dependencies
├── templates/             # HTML templates for web interface
│   ├── index.html
│   ├── folder_detail.html
│   ├── folder_rows.html
│   └── status_indicator.html
├── static/               # Static web assets
└── README.md
```

## Core Components

### 1. Speech Extraction Pipeline (`extract_voice_clips.py`)
The main processing pipeline that:
- Extracts audio from video files using FFmpeg
- Runs Voice Activity Detection (VAD) using Silero VAD
- Merges nearby speech segments
- Extracts video clips for each speech segment
- Transcribes audio using Whisper.cpp
- Uses Ollama (gemma3:4b model) for:
  - Correcting transcription errors
  - Generating 2-4 word summaries
  - Scoring clips for sleep talk likelihood
- Ranks clips by relevance (time-based + sleep talk scoring)
- Generates thumbnails for each clip

### 2. Web Interface (`serve.py`)
FastAPI-based web server providing:
- Admin dashboard for monitoring extraction status
- Manual extraction triggers
- Real-time status updates using HTMX
- Folder processing status overview
- Scheduled daily extraction at 10:30 AM

## Key Features

### Voice Activity Detection
- Uses Silero VAD to detect speech segments
- Merges segments within 10 seconds of each other
- Filters out non-speech audio

### AI-Powered Processing
- **Whisper.cpp**: Speech-to-text transcription
- **Ollama with gemma3:4b**: 
  - Grammar/spelling correction
  - Automatic summarization
  - Sleep talk classification

### Smart Ranking System
- Time-based scoring (prioritizes clips from midnight-6am)
- Sleep talk likelihood scoring
- Combined scoring for final ranking

### Web Dashboard
- Real-time status monitoring
- Folder processing status
- Manual extraction triggers
- Scheduled task management

## Dependencies

### Core Libraries
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `apscheduler` - Task scheduling
- `torch` + `torchaudio` - PyTorch for audio processing
- `silero-vad` - Voice Activity Detection
- `soundfile` - Audio file handling
- `scipy` + `numpy` - Scientific computing

### External Tools Required
- **FFmpeg** - Video/audio processing
- **Whisper.cpp** - Speech transcription
- **Ollama** with gemma3:4b model - Text processing

## Configuration

### File Paths
- Default surveillance directory: `/Volumes/surveillance/Bedroom Camera`
- Whisper binary: `/usr/local/bin/whisper-cli`
- Whisper model: `~/models/whisper.cpp/ggml-medium.en.bin`

### Processing Settings
- Audio sample rate: 16kHz
- Speech segment merge threshold: 10 seconds
- Clip padding: 3 seconds before/after speech
- Daily extraction schedule: 10:30 AM

## Running the Service

### Installation
```bash
# Install dependencies
uv pip install -e .

# Ensure external tools are available
# - FFmpeg
# - Whisper.cpp
# - Ollama with gemma3:4b model
```

### Starting the Web Server
```bash
# Development mode
python serve.py

# Production mode
uvicorn serve:app --host 0.0.0.0 --port 8000
```

### Direct Script Execution
```bash
# Run extraction directly
python extract_voice_clips.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main dashboard |
| `/api/status` | GET | Current extraction status |
| `/api/extract` | POST | Trigger manual extraction |
| `/api/folders` | GET | List folders and status |
| `/folder/{path}` | GET | Folder detail view |
| `/files/{path}` | GET | Serve static files |

## File Organization

### Input Structure
```
/Volumes/surveillance/Bedroom Camera/
├── 2024-01-01/
│   ├── video1.mp4
│   └── video2.mp4
└── 2024-01-02/
    └── video3.mp4
```

### Output Structure
```
/Volumes/surveillance/Bedroom Camera/
└── 2024-01-01/
    ├── .processed          # Processing marker
    └── voice_clips/
        ├── 1_summary.mp4   # Ranked clips
        ├── 1_summary.txt   # Transcripts
        ├── 1_summary.png   # Thumbnails
        ├── 2_summary.mp4
        └── ...
```

## Development Notes

### Processing Pipeline Flow
1. **Audio Extraction**: FFmpeg extracts 16kHz mono audio
2. **VAD Processing**: Silero VAD identifies speech segments
3. **Segment Merging**: Nearby segments are combined
4. **Clip Extraction**: Video clips created with padding
5. **Transcription**: Whisper.cpp generates text
6. **AI Enhancement**: Ollama corrects and summarizes
7. **Scoring**: Time and content-based ranking
8. **Organization**: Files renamed and ranked

### Key Functions
- `extract_voice_clips.py:main()` - Main batch processing
- `extract_voice_clips.py:process_video()` - Single video processing
- `serve.py:run_extraction()` - Web-triggered extraction
- `serve.py:get_folder_status()` - Status monitoring

### Web Interface Features
- HTMX for real-time updates
- Tailwind CSS for styling
- Background processing with threading
- Automatic status refresh

## Testing & Validation

### Prerequisites Check
```bash
# Verify external tools
ffmpeg -version
whisper-cli --help
ollama list | grep gemma3
```

### Manual Testing
1. Place test videos in surveillance directory
2. Run extraction via web interface
3. Verify clips, transcripts, and thumbnails generated
4. Check ranking accuracy

This service is designed for continuous operation with minimal manual intervention, automatically processing new surveillance footage and providing easy access to extracted voice clips through the web interface.