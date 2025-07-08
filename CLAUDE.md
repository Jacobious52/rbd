# RBD - Speech Extraction Service

## Overview
This project is a Python-based speech extraction service designed to process surveillance camera footage. It automatically extracts voice clips from video files, transcribes them using Whisper.cpp, and provides a web interface for monitoring and managing the extraction process.

## Project Structure
```
rbd/
├── extract_voice_clips.py  # Core extraction logic
├── batch_llm_processor.py  # Batch LLM processing module
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
- Uses batch LLM processing (via `batch_llm_processor.py`) for:
  - Correcting transcription errors
  - Generating 2-4 word summaries
  - Scoring clips for sleep talk likelihood
- Ranks clips by relevance (time-based + sleep talk scoring)
- Generates thumbnails for each clip

### 2. Batch LLM Processing (`batch_llm_processor.py`)
Optimized LLM processing module that:
- Consolidates transcript correction, summarization, and classification into single API calls
- Reduces LLM overhead from 3N calls to 1 call per video
- Provides robust JSON response parsing and validation
- Implements fallback mechanisms for batch processing failures
- Maintains consistent processing quality across all clips

### 3. Web Interface (`serve.py`)
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
- **Batch LLM Processing**: Efficient processing using Ollama with gemma3:4b
  - Grammar/spelling correction
  - Automatic summarization
  - Sleep talk classification
  - Consolidated into single API calls per video for optimal performance

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
5. **Transcription**: Whisper.cpp generates text for each clip
6. **Batch LLM Processing**: Single API call processes all transcripts for:
   - Grammar/spelling correction
   - 2-4 word summary generation
   - Sleep talk likelihood scoring
7. **Time-based Scoring**: Clips scored based on timestamp (midnight-6am priority)
8. **Final Ranking**: Combined scoring and file organization with numeric prefixes

### Key Functions
- `extract_voice_clips.py:main()` - Main batch processing
- `extract_voice_clips.py:process_video()` - Single video processing with batch LLM
- `batch_llm_processor.py:BatchLLMProcessor` - Consolidated LLM processing
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

---

## Recent Performance Improvements

### Batch LLM Processing Optimization (2024)
A major performance optimization was implemented to consolidate individual LLM calls into batch processing:

**Before**: Each voice clip required 3 separate LLM calls (correction, summary, classification)
- For 10 clips = 30 LLM API calls per video
- High latency and API overhead

**After**: Single batch call processes all clips simultaneously
- For 10 clips = 1 LLM API call per video
- ~90% reduction in API calls
- Maintained same quality and file naming conventions

**Implementation Details**:
- **Module**: `batch_llm_processor.py` - Centralized batch processing
- **JSON Response**: Structured parsing with validation and error handling
- **Fallback System**: Graceful degradation to individual processing on failures
- **Preserved Functionality**: Same "1_summary.mp4" file naming and ranking system

**Benefits**:
- Dramatic performance improvement for video processing
- Reduced API costs and latency
- Improved consistency across clips (single context)
- Enhanced error handling and logging

---

## Development Guidelines & Best Practices

### Web Interface Architecture Patterns

#### Dual Rendering System
The application uses a **dual rendering approach**:
- **Server-side rendering**: Initial page load via `/` endpoint
- **HTMX rendering**: Dynamic updates via `/api/folders` endpoint

**Critical Requirements**:
1. **Data structure consistency**: Both endpoints must return identical data structures
2. **URL field synchronization**: Both paths must add `folder['url']` with relative paths
3. **Filtering logic alignment**: Server-side and client-side filters must match

#### HTMX Integration Best Practices
- **Avoid redundant calls**: Don't call `loadFolders()` immediately on page load if server content is already correct
- **Chain callbacks**: Use `.then()` callbacks to apply filters after HTMX content loads
- **Loading states**: Provide visual feedback during HTMX requests
- **Error handling**: Always include error handling for HTMX failures

### Data Structure Patterns

#### Folder Data Structure
```python
{
    "name": "2024-01-01",
    "path": "/full/absolute/path",  # Used for file operations
    "url": "/folder/relative/path", # Used for web links
    "processed": boolean,
    "video_count": integer,
    "last_modified": "YYYY-MM-DD HH:MM:SS"
}
```

#### Template Data Requirements
- `folder_rows.html` expects: `folder.url`, `folder.video_count`, `folder.processed`
- Always add `data-clip-count` attribute to table rows for JavaScript filtering
- Include `folder-row` class for consistent styling and selection

---

## Debugging Guide

### Visual Timing Issues (Flash Problems)

**Symptoms**: Content briefly appears then disappears on page reload

**Common Causes**:
1. **Race conditions**: Multiple refresh mechanisms competing
2. **HTMX overwrites**: Fresh content replacing filtered content
3. **JavaScript timing**: Filters applied before content is stable

**Debugging Steps**:
```javascript
// 1. Monitor filter applications
let filterCallCount = 0;
const originalApplyFilter = applyFolderFilter;
applyFolderFilter = function() {
    console.log("Filter call #" + ++filterCallCount);
    return originalApplyFilter.apply(this, arguments);
};

// 2. Check data integrity
document.querySelectorAll('.folder-row').forEach((row, i) => {
    console.log(`Row ${i}: clips=${row.getAttribute('data-clip-count')}, display=${row.style.display}`);
});

// 3. Monitor DOM changes
const observer = new MutationObserver(mutations => {
    mutations.forEach(mutation => {
        if (mutation.target.id === 'folders-list') {
            console.log("Content changed:", mutation.type);
        }
    });
});
observer.observe(document.getElementById('folders-list'), {childList: true, subtree: true});
```

### Server vs Client Rendering Issues

**Problem**: Different behavior on page reload vs manual refresh

**Investigation Steps**:
1. **Network Tab**: Compare response content between `/` and `/api/folders`
2. **Data Structure**: Verify both endpoints return identical folder objects
3. **Timing**: Check if initial HTMX calls are overriding server content

**Solution Pattern**:
- Ensure both endpoints use identical data processing logic
- Eliminate redundant HTMX calls on page load
- Use server-side filtering as the source of truth

### JavaScript Execution Order Debugging

**Tools**:
```javascript
// Breakpoint on key functions
debugger; // Add before critical operations

// Execution timing
console.time("filterApplication");
applyFolderFilter();
console.timeEnd("filterApplication");

// Function call tracing
function trace(fn, name) {
    return function() {
        console.log(`Calling ${name}`);
        return fn.apply(this, arguments);
    };
}
loadFolders = trace(loadFolders, "loadFolders");
```

---

## Architecture Patterns

### Filtering Logic Architecture

#### Server-Side Filtering (Preferred)
```python
# In both / and /api/folders endpoints
hide_empty = request.query_params.get('hide_empty', 'true').lower() == 'true'
if hide_empty:
    # Business logic: Show pending folders even with 0 clips
    folders = [folder for folder in folders if folder['video_count'] > 0 or not folder['processed']]
```

#### Client-Side Filtering (Backup)
```javascript
// Only used for immediate UI feedback
function applyFolderFilter() {
    const hideEmpty = document.getElementById('hideEmptyFolders').checked;
    const folderRows = document.querySelectorAll('.folder-row');
    
    folderRows.forEach(row => {
        const clipCount = parseInt(row.getAttribute('data-clip-count') || '0');
        const isProcessed = row.querySelector('.bg-green-100') !== null;
        
        if (hideEmpty && clipCount === 0 && isProcessed) {
            row.style.display = 'none';
        } else {
            row.style.display = '';
        }
    });
}
```

### State Management Patterns

#### Filter State Synchronization
1. **URL parameters**: Pass filter state via query params
2. **Checkbox state**: Reflects current filter settings  
3. **Server filtering**: Apply logic server-side for consistency
4. **Immediate updates**: Trigger HTMX reload on filter changes

#### Data Flow Pattern
```
User Action → Checkbox Change → loadFolders() → Server Filter → Template Render → DOM Update
```

---

## Common Issues & Solutions

### Issue: Visual Flash on Page Reload
**Cause**: HTMX immediately replacing server-rendered content
**Solution**: Remove immediate `loadFolders()` call from `DOMContentLoaded`

### Issue: Inconsistent Filtering Behavior  
**Cause**: Different data structures between server and HTMX endpoints
**Solution**: Ensure both paths use identical folder processing logic

### Issue: Filter Not Applied After HTMX Update
**Cause**: Missing `.then()` callback after HTMX request
**Solution**: Always apply filters in HTMX callbacks

### Issue: Race Conditions Between Refresh Mechanisms
**Cause**: Multiple `setInterval` timers with different refresh logic
**Solution**: Consolidate all refresh logic into single mechanism

### Issue: Broken Folder Links
**Cause**: Using absolute paths instead of relative paths for URLs
**Solution**: Use `Path.relative_to(BASE_DIR)` for URL construction

---

## Code Modification Guidelines

### Adding New Filtering Logic

**Server-side modifications** (preferred):
1. Update `get_folder_status()` in `serve.py:100-125`
2. Modify filtering logic in both:
   - `read_root()` endpoint (`serve.py:195-201`)
   - `list_folders()` endpoint (`serve.py:283-286`)

**Client-side modifications** (if needed):
1. Update `applyFolderFilter()` in `templates/index.html:227-252`
2. Ensure data attributes are available in `folder_rows.html`

### Template Modifications

**Adding new folder data**:
1. Update `get_folder_status()` to include new fields
2. Add corresponding HTML in `folder_rows.html`
3. Update table headers in `index.html`
4. Ensure both server and HTMX endpoints include new data

**Adding new UI elements**:
1. Add HTML structure in `index.html`
2. Include necessary JavaScript event handlers
3. Update HTMX targets and triggers as needed
4. Test both page reload and dynamic updates

### JavaScript Timing Considerations

**Safe patterns**:
```javascript
// Wait for DOM before accessing elements
document.addEventListener('DOMContentLoaded', function() {
    // Safe to access DOM elements here
});

// Chain operations after HTMX
htmx.ajax('GET', url, options).then(function() {
    // Safe to process new content here
});

// Small delays for visual smoothness
setTimeout(function() {
    // Apply visual changes
}, 50);
```

---

## Testing Strategies

### Page Reload vs Manual Refresh Testing
```bash
# Test different scenarios
1. Fresh page load (Ctrl+R/Cmd+R)
2. Manual refresh button click
3. Checkbox toggle
4. Automatic 30-second refresh
```

### Visual Timing Verification
```javascript
// Monitor content changes
function checkVisualState() {
    const hiddenRows = document.querySelectorAll('.folder-row[style*="none"]').length;
    const visibleRows = document.querySelectorAll('.folder-row:not([style*="none"])').length;
    console.log(`Hidden: ${hiddenRows}, Visible: ${visibleRows}`);
}

// Check at intervals
setInterval(checkVisualState, 100);
```

### Data Integrity Checks
```javascript
// Verify server vs client data consistency
function validateDataIntegrity() {
    fetch('/api/folders?hide_empty=false')
        .then(r => r.text())
        .then(html => {
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = html;
            const serverRows = tempDiv.querySelectorAll('.folder-row');
            const clientRows = document.querySelectorAll('.folder-row');
            
            console.log(`Server rows: ${serverRows.length}, Client rows: ${clientRows.length}`);
            
            serverRows.forEach((row, i) => {
                const serverClips = row.getAttribute('data-clip-count');
                const clientClips = clientRows[i]?.getAttribute('data-clip-count');
                if (serverClips !== clientClips) {
                    console.error(`Mismatch at row ${i}: server=${serverClips}, client=${clientClips}`);
                }
            });
        });
}
```

### Network Analysis
1. **Check request timing**: Ensure no competing requests
2. **Verify response content**: Compare server vs HTMX responses
3. **Monitor query parameters**: Confirm filter state is passed correctly
4. **Validate response structure**: Ensure consistent HTML structure

This comprehensive debugging and development guide should help future agents avoid common pitfalls and maintain consistency when making modifications to the project.

---

## Refactoring and Optimization Best Practices

### LLM Processing Architecture

#### Individual vs Batch Processing
When working with LLM-based processing in this project, consider the performance implications:

**Individual Processing Pattern** (Legacy):
```python
for clip in clips:
    corrected = correct_transcript(clip)
    summary = summarize_text(clip)
    score = classify_sleep_talk(clip)
    # Results in 3N LLM calls
```

**Batch Processing Pattern** (Optimized):
```python
# Single batch call for all clips
batch_results = batch_processor.process_with_fallback(clips, fallback_functions)
# Results in 1 LLM call per video
```

#### Key Refactoring Principles

1. **Preserve File Naming**: Always maintain the existing "1_summary.mp4" convention
2. **Implement Fallbacks**: Ensure graceful degradation when batch processing fails
3. **Validate Outputs**: Comprehensive JSON parsing and response validation
4. **Error Isolation**: Individual clip failures should not affect the entire batch

#### Performance Optimization Strategies

**Before Optimization**:
- Profile existing bottlenecks (use logging to identify slow operations)
- Measure current performance metrics (API calls, processing time)
- Identify repeated operations that can be consolidated

**During Optimization**:
- Implement with backward compatibility in mind
- Test with various input sizes (1 clip, 10 clips, 20+ clips)
- Ensure error handling doesn't break the entire pipeline

**After Optimization**:
- Verify functionality preservation (same outputs, same file names)
- Monitor performance improvements (reduced API calls, faster processing)
- Update documentation with new architecture patterns

#### Testing Refactored Systems

**Syntax Validation**:
```bash
python -m py_compile your_module.py
python -c "import your_module; print('Import successful')"
```

**Functional Testing**:
```python
# Test class instantiation
processor = BatchLLMProcessor()

# Test key methods
prompt = processor.create_batch_prompt(sample_data)
results = processor.parse_batch_response(sample_response, expected_count)
```

**Integration Testing**:
- Test with actual video files (if available)
- Verify file naming conventions remain consistent
- Ensure ranking and scoring algorithms produce expected results

### Module Architecture Guidelines

#### When to Create New Modules
- **Complex functionality**: When adding substantial new features (>100 lines)
- **Reusable components**: When functionality might be used in multiple places
- **Performance optimization**: When consolidating related operations
- **Clear separation**: When functionality has distinct responsibilities

#### Module Design Patterns
```python
class ProcessorModule:
    def __init__(self, config_params):
        # Initialize with configurable parameters
        pass
    
    def process_batch(self, inputs):
        # Main processing logic
        pass
    
    def process_with_fallback(self, inputs, fallback_functions):
        # Implement graceful fallback mechanisms
        pass
    
    def validate_outputs(self, outputs):
        # Comprehensive output validation
        pass
```

This refactoring guide provides patterns for future performance optimizations while maintaining system reliability and backward compatibility.