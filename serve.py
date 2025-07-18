import os
import uvicorn
from pathlib import Path
from datetime import datetime
from typing import List, Dict
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, FileResponse
from fastapi import status
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

# Import the existing speech extraction functionality
from extract_voice_clips import main as extract_voice_clips

# Set up templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Startup code
    os.makedirs(os.path.join(os.path.dirname(__file__), "templates"), exist_ok=True)
    os.makedirs(os.path.join(os.path.dirname(__file__), "static"), exist_ok=True)
    
    # Ensure templates exist
    templates_to_check = ["index.html", "folder_rows.html", "status_indicator.html"]
    for template in templates_to_check:
        template_path = os.path.join(os.path.dirname(__file__), "templates", template)
        if not os.path.exists(template_path):
            if template == "folder_rows.html":
                # Create folder_rows.html template
                with open(template_path, "w") as f:
                    f.write("""
                    {% for folder in folders %}
                    <tr>
                        <td class="px-6 py-4 whitespace-nowrap">
                            <div class="text-sm font-medium text-gray-900">{{ folder.name }}</div>
                            <div class="text-sm text-gray-500">{{ folder.path }}</div>
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap">
                            <span class="px-2 inline-flex text-xs leading-5 font-semibold rounded-full 
                                {{ 'bg-green-100 text-green-800' if folder.processed else 'bg-yellow-100 text-yellow-800' }}">
                                {{ 'Processed' if folder.processed else 'Pending' }}
                            </span>
                        </td>
                        <td class="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                            {{ folder.last_modified }}
                        </td>
                    </tr>
                    {% endfor %}
                    """.strip())
            elif template == "status_indicator.html":
                # Create status_indicator.html template
                with open(template_path, "w") as f:
                    f.write("""
                    <div class="flex items-center mb-4">
                        <div class="w-3 h-3 rounded-full mr-2 
                            {{ 'bg-green-500' if status == 'completed' else 
                               'bg-yellow-500' if status == 'running' else 
                               'bg-red-500' if status == 'error' else 
                               'bg-gray-500' }}"></div>
                        <span class="font-medium">{{ status|title }}</span>
                    </div>
                    <p class="text-gray-700 mb-2">{{ message }}</p>
                    <p class="text-sm text-gray-500">Last updated: {{ timestamp }}</p>
                    """.strip())
            else:
                raise FileNotFoundError(f"Template file not found at {template_path}")
    
    yield  # Application runs here
    
    # Shutdown code would go here
    pass

app = FastAPI(lifespan=lifespan, title="Speech Extraction Service")

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# Configuration
BASE_DIR = Path("/Volumes/surveillance/Bedroom Camera")  # Base directory to scan for folders
PROCESSED_MARKER = ".processed"

class ExtractionStatus(BaseModel):
    status: str
    message: str
    timestamp: str

# In-memory storage for extraction status
current_status = ExtractionStatus(
    status="idle",
    message="Service started",
    timestamp=datetime.now().isoformat()
)

def get_folder_status() -> List[Dict[str, str]]:
    """Scan the base directory and return folder status."""
    if not BASE_DIR.exists():
        return []
    
    folders = []
    for item in sorted(BASE_DIR.iterdir(), reverse=True):
        if item.is_dir() and item.name.startswith("202"):  # Assuming folders start with year
            is_processed = (item / PROCESSED_MARKER).exists()
            
            # Count processed voice clips in the voice_clips subfolder
            voice_clips_dir = item / "voice_clips"
            if voice_clips_dir.exists() and voice_clips_dir.is_dir():
                clip_count = len(list(voice_clips_dir.glob("*.mp4")))
            else:
                clip_count = 0
            
            folders.append({
                "name": item.name,
                "path": str(item),
                "processed": is_processed,
                "video_count": clip_count,
                "last_modified": datetime.fromtimestamp(item.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            })
    return folders

def get_voice_clips(folder_path: Path) -> List[Dict[str, str]]:
    """Get list of voice clips in a folder's voice_clips subdirectory."""
    clips = []
    
    if not folder_path.exists() or not folder_path.is_dir():
        return clips
    
    # Look for voice_clips subdirectory
    voice_clips_dir = folder_path / 'voice_clips'
    if not voice_clips_dir.exists() or not voice_clips_dir.is_dir():
        return clips
        
    # Look for video files in voice_clips directory
    for video_file in voice_clips_dir.glob('*.mp4'):
        # Look for corresponding text file
        txt_file = video_file.with_suffix('.txt')
        text_content = ""
        
        if txt_file.exists():
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text_content = f.read().strip()
            except Exception as e:
                print(f"Error reading {txt_file}: {e}")
        
        clips.append({
            'video_path': str(video_file.relative_to(BASE_DIR)),
            'title': video_file.stem,
            'text': text_content
        })
    
    return clips

def run_extraction():
    """Run the voice extraction process and update status."""
    global current_status
    try:
        current_status.status = "running"
        current_status.message = "Extraction in progress..."
        current_status.timestamp = datetime.now().isoformat()
        
        # Run the extraction
        extract_voice_clips()
        
        current_status.status = "completed"
        current_status.message = f"Extraction completed at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        current_status.timestamp = datetime.now().isoformat()
        
    except Exception as e:
        current_status.status = "error"
        current_status.message = f"Error during extraction: {str(e)}"
        current_status.timestamp = datetime.now().isoformat()

# Set up the scheduler
scheduler = BackgroundScheduler()
scheduler.add_job(
    run_extraction,
    CronTrigger(hour=10, minute=30),
    id="daily_extraction",
    name="Run daily extraction at 10:30 AM",
    replace_existing=True
)
scheduler.start()

# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Render the admin dashboard."""
    folders = get_folder_status()
    
    # Apply server-side filtering for initial page load (default to hiding processed empty folders)
    hide_empty = request.query_params.get('hide_empty', 'true').lower() == 'true'
    if hide_empty:
        # Only hide folders with 0 clips if they are already processed
        # Always show pending folders even if they have 0 clips
        folders = [folder for folder in folders if folder['video_count'] > 0 or not folder['processed']]
    
    # Add URL to each folder using relative path (same as API endpoint)
    for folder in folders:
        folder_path = Path(folder['path'])
        relative_path = folder_path.relative_to(BASE_DIR)
        folder['url'] = f"/folder/{relative_path}"
    
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "status": current_status,
            "folders": folders,
            "base_dir": str(BASE_DIR)
        }
    )

@app.get("/api/status")
async def get_status(request: Request):
    """Get current extraction status."""
    if "hx-request" in request.headers:
        return templates.TemplateResponse(
            "status_indicator.html",
            {
                "request": request,
                "status": current_status.status,
                "message": current_status.message,
                "timestamp": current_status.timestamp
            },
            headers={"HX-Retarget": "#status-container", "HX-Reswap": "innerHTML"}
        )
    return current_status

@app.post("/api/extract")
async def trigger_extraction(request: Request):
    """Trigger manual extraction."""
    if current_status.status == "running":
        return HTMLResponse(
            "<div class='text-red-500'>Extraction is already in progress</div>",
            status_code=400
        )
    
    # Run in background
    import threading
    thread = threading.Thread(target=run_extraction)
    thread.start()
    
    return HTMLResponse("""
        <div class="text-green-600">
            Extraction started in the background. This page will update automatically.
        </div>
    """)

@app.get("/folder/{folder_path:path}", response_class=HTMLResponse)
async def folder_detail(request: Request, folder_path: str):
    """Display details for a specific folder."""
    try:
        full_path = BASE_DIR / folder_path
        if not full_path.exists() or not full_path.is_dir():
            raise HTTPException(status_code=404, detail="Folder not found")
            
        clips = get_voice_clips(full_path)
        return templates.TemplateResponse("folder_detail.html", {
            "request": request,
            "folder_name": full_path.name,
            "folder_path": str(full_path.relative_to(BASE_DIR)),
            "clips": clips
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading folder: {str(e)}")

@app.get("/api/folders")
async def list_folders(request: Request):
    """Get list of folders and their status."""
    try:
        folders = get_folder_status()
        
        # Apply server-side filtering based on query parameter
        hide_empty = request.query_params.get('hide_empty', 'true').lower() == 'true'
        if hide_empty:
            # Only hide folders with 0 clips if they are already processed
            # Always show pending folders even if they have 0 clips
            folders = [folder for folder in folders if folder['video_count'] > 0 or not folder['processed']]
        
        # Add URL to each folder using relative path
        for folder in folders:
            folder_path = Path(folder['path'])
            relative_path = folder_path.relative_to(BASE_DIR)
            folder['url'] = f"/folder/{relative_path}"
        return templates.TemplateResponse("folder_rows.html", {"request": request, "folders": folders})
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error listing folders: {str(e)}"}
        )

# Serve static files from the base directory
@app.get("/files/{file_path:path}")
async def serve_file(file_path: str):
    """Serve static files from the base directory."""
    file_path = Path(file_path)
    full_path = BASE_DIR / file_path
    
    if not full_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(full_path)

if __name__ == "__main__":
    uvicorn.run(
        "serve:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )