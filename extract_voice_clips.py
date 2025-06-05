import os
import subprocess
from pathlib import Path
import torch
import torchaudio
from silero_vad import get_speech_timestamps, save_audio, read_audio


def main() -> None:
    print("=== Voice Extraction Script Started ===")

    # Config
    video_path = Path(
        "/Volumes/surveillance/Bedroom Camera/20250605PM/Bedroom Camera-20250605-214810-1749124090161-7.mp4"
    )
    output_dir = video_path.parent / "voice_clips"
    audio_path = output_dir / "audio.wav"
    os.makedirs(output_dir, exist_ok=True)

    # Stage 1: Extracting audio from video...
    print("Stage 1: Extracting audio from video...")
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-err_detect",
            "ignore_err",
            "-fflags",
            "+discardcorrupt",
            "-i",
            str(video_path),
            "-ar",
            "16000",
            "-ac",
            "1",
            "-vn",
            str(audio_path),
        ]
    )
    print(f"Audio extracted to {audio_path}")

    # Stage 2: Loading audio for VAD processing...
    print("Stage 2: Loading audio for VAD processing...")
    audio = read_audio(str(audio_path), sampling_rate=16000)
    print("Audio loaded into memory.")

    # Stage 3: Running voice activity detection...
    print("Stage 3: Running voice activity detection...")
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=False
    )
    (get_speech_timestamps, _, _, _, _) = utils

    speech_timestamps = get_speech_timestamps(audio, model, sampling_rate=16000)
    print(
        f"Detected {len(speech_timestamps)} speech segments: {speech_timestamps}"
    )

    # Merge speech segments that are close to each other
    merged_timestamps = []
    gap_threshold = 2 * 16000  # 2 seconds in samples
    for ts in speech_timestamps:
        if not merged_timestamps:
            merged_timestamps.append(ts.copy())
        else:
            last = merged_timestamps[-1]
            if ts["start"] - last["end"] <= gap_threshold:
                # extend the last segment
                last["end"] = ts["end"]
            else:
                merged_timestamps.append(ts.copy())
    print(f"Merged into {len(merged_timestamps)} segments: {merged_timestamps}")

    # Stage 4: Extracting individual voice clips...
    print("Stage 4: Extracting individual voice clips...")
    for i, ts in enumerate(merged_timestamps):
        start = max(0, ts["start"] / 16000 - 1)
        duration = (ts["end"] - ts["start"]) / 16000 + 2
        print(
            f"  -> Processing clip {i+1}/{len(merged_timestamps)}: start={start:.2f}s, duration={duration:.2f}s"
        )
        out_file = output_dir / f"voice_clip_{i+1:03d}.mp4"
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-ss",
                str(start),
                "-i",
                str(video_path),
                "-t",
                str(duration),
                "-map",
                "0:v:0",
                "-map",
                "0:a:0?",
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-movflags",
                "+faststart",
                str(out_file),
            ]
        )

    print(f"Extracted {len(merged_timestamps)} voice clips to {output_dir}")
    print("Stage 5: Cleaning up temporary files...")

    # Cleanup temporary audio file
    if audio_path.exists():
        audio_path.unlink()
        print("Temporary audio file removed. Process complete.")


if __name__ == "__main__":
    main()
