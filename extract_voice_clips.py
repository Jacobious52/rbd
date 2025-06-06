import os
import subprocess
from pathlib import Path
import torch
from silero_vad import get_speech_timestamps, save_audio, read_audio
from typing import List
import shutil

def summarize_text(transcript_file: Path) -> str:
    """Use Ollama (gemma3:4b) to summarize transcript into a 2–4 word title."""
    # Read the full transcript
    with open(transcript_file, 'r') as f:
        content = f.read()

    try:
        # Build the prompt string
        prompt = f"Provide exactly a 2 to 4 word title summarizing the following text. Return only the title with no additional text: {content}"

        # Call Ollama with `run gemma3:4b`
        result = subprocess.run(
            [
                "ollama",
                "run",
                "gemma3:4b",
                prompt
            ],
            capture_output=True,
            text=True,
            check=True
        )

        # Debug: show raw stdout from Ollama
        raw_output = result.stdout.strip()
        # Validate that the summary is exactly 2 to 4 words
        words = raw_output.split()
        if len(words) < 2 or len(words) > 4:
            print(f"[DEBUG] Ollama output did not meet 2-4 word requirement: '{raw_output}'. Using filename stem.")
            raw_output = transcript_file.stem
        print(f"[DEBUG] Ollama raw output for {transcript_file.name}: '{raw_output}'")

        # Sanitize into a safe filename
        summary = raw_output.replace(" ", "_")
        summary = "".join(c for c in summary if c.isalnum() or c in ("-", "_"))
        return summary or transcript_file.stem

    except subprocess.CalledProcessError as e:
        print(f"Error summarizing transcript: {e}")
        return transcript_file.stem

# Path to the Whisper.cpp executable (adjust as needed)
WHISPER_BIN = Path("/usr/local/bin/whisper-cli")
# Path to the Whisper.cpp model file (adjust as needed)
WHISPER_MODEL = Path("~/models/whisper.cpp/ggml-medium.en.bin").expanduser()

def extract_audio(video_path: Path, audio_path: Path) -> None:
    """Extract audio from video using FFmpeg."""
    print("Stage 1: Extracting audio from video...")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-err_detect", "ignore_err",
            "-fflags", "+discardcorrupt",
            "-i", str(video_path),
            "-ar", "16000",
            "-ac", "1",
            "-vn", str(audio_path),
        ]
    )
    print(f"Audio extracted to {audio_path}")

def load_audio(audio_path: Path) -> torch.Tensor:
    """Load audio file into a tensor for VAD processing."""
    print("Stage 2: Loading audio for VAD processing...")
    audio = read_audio(str(audio_path), sampling_rate=16000)
    print("Audio loaded into memory.")
    return audio

def run_vad(audio: torch.Tensor) -> List[dict]:
    """Run Silero VAD on the loaded audio and return speech timestamps."""
    print("Stage 3: Running voice activity detection...")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
    )  # type: ignore
    get_speech, _, _, _, _ = utils
    speech_timestamps = get_speech(audio, model, sampling_rate=16000)
    print(f"Detected {len(speech_timestamps)} speech segments.")
    return speech_timestamps

def merge_segments(speech_timestamps: List[dict], gap_seconds: float = 2.0) -> List[dict]:
    """Merge adjacent speech segments that are separated by <= gap_seconds."""
    print("Merging speech segments...")
    merged: List[dict] = []
    gap_threshold = int(gap_seconds * 16000)  # convert seconds to samples
    for ts in speech_timestamps:
        if not merged:
            merged.append(ts.copy())
        else:
            last = merged[-1]
            if ts["start"] - last["end"] <= gap_threshold:
                last["end"] = ts["end"]
            else:
                merged.append(ts.copy())
    print(f"Merged into {len(merged)} segments.")
    return merged

def extract_clips(video_path: Path, merged_timestamps: List[dict], output_dir: Path) -> List[Path]:
    """Extract video clips for each merged speech segment and return list of clip paths."""
    print("Stage 4: Extracting individual voice clips...")
    clip_paths: List[Path] = []
    for i, ts in enumerate(merged_timestamps):
        start_sec = max(0, ts["start"] / 16000 - 1)
        duration_sec = (ts["end"] - ts["start"]) / 16000 + 2
        clip_file = output_dir / f"{video_path.stem}_clip_{i+1:03d}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel", "error",
                "-ss", str(start_sec),
                "-i", str(video_path),
                "-t", str(duration_sec),
                "-map", "0:v:0",
                "-map", "0:a:0?",
                "-c:v", "libx264",
                "-c:a", "aac",
                "-movflags", "+faststart",
                str(clip_file),
            ]
        )
        clip_paths.append(clip_file)
    print(f"Extracted {len(clip_paths)} voice clips to {output_dir}")
    return clip_paths

def transcribe_clip(clip_path: Path) -> None:
    """Transcribe a single video clip using Whisper.cpp and save to a .txt file."""
    print(f"Transcribing {clip_path.name}...")
    audio_clip = clip_path.with_suffix(".wav")
    # Extract audio from clip
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(clip_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            str(audio_clip),
        ]
    )
    # Run Whisper.cpp and write output to .txt
    transcript_file = clip_path.with_suffix(".txt")
    with open(transcript_file, "w") as tf:
        subprocess.run(
            [
                str(WHISPER_BIN),
                "-m",
                str(WHISPER_MODEL),
                "-f",
                str(audio_clip),
            ],
            stdout=tf,
        )
    # Cleanup WAV file
    if audio_clip.exists():
        audio_clip.unlink()
    print(f"Transcript saved to {transcript_file}")

def cleanup_temp(audio_path: Path) -> None:
    """Remove temporary audio file."""
    if audio_path.exists():
        audio_path.unlink()
        print("Temporary audio file removed.")

def process_video(video_path: Path) -> None:
    """Orchestrate the entire processing pipeline for a single video."""
    print(f"\n=== Processing {video_path.name} ===")
    output_dir = video_path.parent / "voice_clips"
    audio_path = output_dir / f"{video_path.stem}_audio.wav"
    os.makedirs(output_dir, exist_ok=True)

    extract_audio(video_path, audio_path)
    audio_tensor = load_audio(audio_path)
    speech_timestamps = run_vad(audio_tensor)
    merged_timestamps = merge_segments(speech_timestamps)
    clip_files = extract_clips(video_path, merged_timestamps, output_dir)

    # Transcribe, summarize, and rename each extracted clip
    for clip in clip_files:
        transcribe_clip(clip)
        transcript_file = clip.with_suffix(".txt")
        # Generate a concise title using Ollama
        summary = summarize_text(transcript_file)
        print(f"[DEBUG] Renaming to: clip='{summary}{clip.suffix}', transcript='{summary}.txt'")
        # Build new filenames using the summary text
        new_clip = clip.with_name(f"{summary}{clip.suffix}")
        new_txt = transcript_file.with_name(f"{summary}.txt")
        # Rename the clip and transcript files to the summary-based names
        clip.rename(new_clip)
        transcript_file.rename(new_txt)

    cleanup_temp(audio_path)

def clean_output_folder(directory: Path) -> None:
    """Delete existing voice_clips folder if it exists."""
    voice_clips_dir = directory / "voice_clips"
    if voice_clips_dir.exists():
        print(f"Cleaning up existing voice_clips folder: {voice_clips_dir}")
        shutil.rmtree(voice_clips_dir)

def main() -> None:
    """Batch process all videos in the NVR directory structure."""
    print("=== Batch Voice Extraction Script Started ===")
    root_dir = Path("/Volumes/surveillance/Bedroom Camera")
    for subdir in sorted(root_dir.iterdir()):
        if subdir.is_dir() and subdir.name.startswith("202"):
            processed_marker = subdir / ".processed"
            if processed_marker.exists():
                print(f"Skipping already processed folder: {subdir.name}")
                continue
            print(f"\n--- Entering directory: {subdir.name} ---")
            clean_output_folder(subdir)
            for file in sorted(subdir.glob("*.mp4")):
                process_video(file)
            # Mark this folder as processed
            processed_marker.touch()
            print(f"Marked folder as processed: {subdir.name}")

if __name__ == "__main__":
    main()
