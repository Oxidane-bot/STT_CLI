# STT CLI - Speech-to-Text Command Line Interface

A command-line tool for transcribing audio and video files using OpenAI's Whisper speech recognition model.

## Features

- Transcribe audio and video files to SRT or TXT format
- Supports multiple languages with auto-detection
- Batch processing for multiple files
- Progress bar with time remaining estimation
- Automatic audio extraction from video files

## Requirements

- Python 3.7 or higher
- FFmpeg (required for video processing and accurate progress reporting)
- PyTorch and Whisper (installed separately or using the `[full]` extra)

## Installation

### Method 1: Install from PyPI

```bash
# Install the base package (does not include PyTorch or Whisper due to size)
pip install speech-to-text-cli

# Install required dependencies separately (can be skipped if already installed)
pip install torch
pip install openai-whisper
```

OR install everything at once:

```bash
# Install the package with all dependencies
pip install speech-to-text-cli[full]
```

### Method 2: Install from source

```bash
# Clone the repository
git clone https://github.com/yourusername/speech-to-text-cli.git
cd speech-to-text-cli

# Install in development mode
pip install -e .

# Install required dependencies separately
pip install torch
pip install openai-whisper
```

### Install FFmpeg

FFmpeg is required for video file processing:

- **Windows**: Download from [ffmpeg.org](https://ffmpeg.org/download.html) or [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) and add to your PATH
- **macOS**: `brew install ffmpeg`
- **Linux**: `apt install ffmpeg` or equivalent for your distribution

## Usage

### Command Structure

```
sttcli <command> [options]
```

### Available Commands

#### 1. transcribe

Transcribe audio or video files.

```bash
sttcli transcribe [options] [files...]
```

Options:
- `--output-format`, `-f`: Output format (`srt` or `txt`, default: `srt`)
- `--language`, `-l`: Language code (e.g., `en`, `zh`) or `auto` for detection (default: `auto`)
- `--model`, `-m`: Whisper model to use (default: `tiny`)
- `--directory`, `-d`: Directory to search for media files if specific files aren't provided (default: current directory)
- `files`: Optional list of specific audio/video files to process

Examples:
```bash
# Transcribe a specific file with default settings
sttcli transcribe my_audio.mp3

# Transcribe all media in a directory using the "base" model
sttcli transcribe --model base --directory /path/to/media/folder

# Transcribe multiple specific files in English
sttcli transcribe --language en file1.mp3 file2.wav video.mp4
```

#### 2. load_model

Pre-download and cache a Whisper model for later use.

```bash
sttcli load_model <model_name>
```

Model options:
- `tiny`: ~39M parameters (fastest, least accurate)
- `base`: ~74M parameters
- `small`: ~244M parameters
- `medium`: ~769M parameters
- `large`: ~1550M parameters (slowest, most accurate)
- `large-v3`: Latest large model

Example:
```bash
sttcli load_model base
```

#### 3. help

Show help information.

```bash
sttcli help
```

## Notes

- Larger models provide better accuracy but require more processing time and memory
- For video files, audio is automatically extracted before transcription
- Progress reporting works best when FFmpeg is installed