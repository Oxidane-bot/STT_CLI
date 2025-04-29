import os
import pytest
import subprocess
import sys

# Define the directory containing the media files
MEDIA_DIR = "test media"

# List of media files to test
MEDIA_FILES = [
    "part1.mp3",
    "part2.mp4",
    "part3.mkv",
]

# Path to the CLI script
CLI_SCRIPT = "speechtotextcli/cli.py"

def check_ffmpeg_installed():
    """Check if ffmpeg is installed on the system."""
    import shutil
    if shutil.which("ffmpeg"):
        print("ffmpeg is installed.")
        return True
    else:
        print("Error: ffmpeg is not installed. This is required for audio/video processing.")
        print("Installation instructions:")
        if sys.platform.startswith('win'):
            print("  Windows: Download from https://ffmpeg.org/download.html or install via Chocolatey with 'choco install ffmpeg'")
        elif sys.platform.startswith('linux'):
            print("  Linux: Install via package manager, e.g., 'sudo apt-get install ffmpeg' on Ubuntu")
        elif sys.platform.startswith('darwin'):
            print("  macOS: Install via Homebrew with 'brew install ffmpeg'")
        return False

@pytest.mark.parametrize("filename", MEDIA_FILES)
def test_cli_transcribe_media_file(filename):
    """Tests the CLI script's ability to transcribe a media file."""
    file_path = os.path.join(MEDIA_DIR, filename)
    print(f"Starting test for file: {file_path}")
    print("-" * 30)
    
    if not check_ffmpeg_installed():
        pytest.fail("ffmpeg is not installed. Please install ffmpeg and rerun the tests.")

    # Construct the command to run the CLI script
    # We use sys.executable to ensure the correct python interpreter is used
    command = [
        sys.executable,
        CLI_SCRIPT,
        "transcribe",
        "--model", "tiny", # Use a small model for faster testing
        "--language", "en", # Specify language if known, or use "auto"
        "--output-format", "txt", # Or 'srt'
        file_path,
    ]

    print(f"\nRunning command: {' '.join(command)}") # Print command for debugging
    print("-" * 30)

    try:
        print("Executing command...")
        # Execute the command
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True # Raise CalledProcessError if the command returns a non-zero exit code
        )
        print("Command execution complete.")
        print("-" * 30)

        print("STDOUT:\n", result.stdout)
        print("STDERR:\n", result.stderr)
        print("-" * 30)

        # Assert that the command ran successfully (check=True already does this for non-zero exit codes)
        # You might add checks for specific output in stdout or stderr here if needed
        print(f"Checking return code...")
        assert result.returncode == 0, f"CLI script failed with exit code {result.returncode}"
        print(f"Return code check passed. Return code: {result.returncode}")
        print(f"Test completed successfully for {file_path}.")
        print("=" * 30)


        # Optional: Clean up generated output files if any
        output_file = os.path.splitext(file_path)[0] + ".txt" # Adjust based on your script's output naming
        if os.path.exists(output_file):
            print(f"Cleaning up output file: {output_file}")
            os.remove(output_file)
            print(f"Cleaned up {output_file}")

    except FileNotFoundError:
        print(f"Error: CLI script not found at {CLI_SCRIPT}.")
        pytest.fail(f"CLI script not found at {CLI_SCRIPT}. Ensure it's in your PATH or the correct relative path is used.")
    except subprocess.CalledProcessError as e:
        print(f"Error: CLI script failed with error.")
        print("STDOUT:\n", e.stdout)
        print("STDERR:\n", e.stderr)
        pytest.fail(f"CLI script failed with error:\n{e.stdout}\n{e.stderr}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        pytest.fail(f"An unexpected error occurred: {e}")