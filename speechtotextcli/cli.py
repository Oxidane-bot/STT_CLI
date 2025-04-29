import argparse
import os
import sys
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s') # Set level to INFO
logger = logging.getLogger(__name__)

# Import from package
from speechtotextcli import cli_core
from speechtotextcli import utils

# Check for required packages
def check_requirements():
    missing_packages = []
    
    # Check for torch
    try:
        import torch
    except ImportError:
        missing_packages.append("PyTorch")
    
    # Check for whisper
    try:
        import whisper
    except ImportError:
        missing_packages.append("Whisper")
    
    # If any packages are missing, display error and installation instructions
    if missing_packages:
        print("Error: The following required packages are not installed:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        
        print("\nInstallation instructions:")
        if "PyTorch" in missing_packages:
            print("  PyTorch: pip install torch")
        if "Whisper" in missing_packages:
            print("  Whisper: pip install openai-whisper")
        
        print("\nAfter installing the required packages, run this command again.")
        sys.exit(1)

# Main CLI function
def main():
    # Check requirements before proceeding
    check_requirements()
    
    parser = argparse.ArgumentParser(
        description="CLI application for audio transcription",
        epilog="Example: sttcli transcribe --output-format txt --language en --model base ./my_audio_file.wav"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Transcribe command
    transcribe_parser = subparsers.add_parser("transcribe", help="Transcribe audio or video files from a directory or specific files.")
    transcribe_parser.add_argument("--output-format", "-f", default="srt", choices=['srt', 'txt'], help="Output format (default: srt)")
    transcribe_parser.add_argument("--language", "-l", default="auto", help="Language code (e.g., en, zh) or 'auto' for detection (default: auto)")
    transcribe_parser.add_argument("--model", "-m", default="tiny", help="Whisper model name (default: tiny)")
    transcribe_parser.add_argument("--directory", "-d", default=".", help="Directory to search for media files if specific files are not provided (default: .)")
    transcribe_parser.add_argument("files", nargs='*', help="Optional: Specific audio/video file(s) to process. If provided, --directory is ignored.")

    # Load model command
    load_model_parser = subparsers.add_parser("load_model", help="Download and load a specific Whisper model into cache")
    load_model_parser.add_argument("model_name", help="Name of the Whisper model to load (e.g., tiny, base, large-v3)")

    # Help command
    help_parser = subparsers.add_parser("help", help="Show this help message and exit")

    args = parser.parse_args()

    # Handle help command
    if args.command == "help":
        parser.print_help()
        sys.exit(0)

    # --- Command Execution ---
    if args.command == "load_model":
        logger.info(f"Executing 'load_model' command for model: {args.model_name}")
        print(f"\nAttempting to load model '{args.model_name}'...")
        success = cli_core.load_model(args.model_name)
        if not success:
            print(f"Failed to load model '{args.model_name}'.")
            sys.exit(1)

    elif args.command == "transcribe":
        logger.info(f"Executing 'transcribe' command with args: {args}")
        print(f"\n--- Starting Transcription Task ---")
        print(f"Model: {args.model}, Language: {args.language}, Format: {args.output_format}")

        # 1. Load the model
        print(f"\nLoading model '{args.model}'...")
        if not cli_core.load_model(args.model):
             print(f"Error: Failed to load model '{args.model}'. Cannot proceed.")
             sys.exit(1)

        # 2. Determine files to process
        files_to_process = []
        if args.files:
            print(f"\nProcessing specified files:")
            valid_files = []
            for f in args.files:
                abs_path = os.path.abspath(f)
                if os.path.isfile(abs_path):
                    valid_files.append(abs_path)
                    print(f"  - Queued: {os.path.basename(abs_path)}")
                else:
                    print(f"  - Warning: Specified file not found, skipping: {f}")
            files_to_process = valid_files
        else:
            target_dir = os.path.abspath(args.directory)
            print(f"\nSearching for media files in directory: {target_dir}")
            files_to_process = utils.get_media_files(target_dir)
            if files_to_process:
                 print(f"Found {len(files_to_process)} supported media file(s).")
            else:
                 print(f"No supported media files found in '{target_dir}'.")

        if not files_to_process:
            print("\nNo files to process. Exiting.")
            sys.exit(0)

        # 3. Process each file
        print(f"\n--- Starting Batch Processing ({len(files_to_process)} file(s)) ---")
        success_count = 0
        fail_count = 0
        start_time_batch = time.time()

        for i, file_path in enumerate(files_to_process):
            logger.info(f"Processing file {i+1}/{len(files_to_process)}: {file_path}")
            start_time_file = time.time()
            try:
                success = cli_core.transcribe_audio(
                    input_file=file_path,
                    model_name=args.model,
                    language=args.language,
                    output_format=args.output_format
                )
                elapsed_file = time.time() - start_time_file
                if success:
                    success_count += 1
                    print(f"Finished '{os.path.basename(file_path)}' in {elapsed_file:.2f}s. [OK]")
                else:
                    fail_count += 1
                    print(f"Failed '{os.path.basename(file_path)}' after {elapsed_file:.2f}s. [FAILED]")
            except KeyboardInterrupt:
                 print("\n\nOperation cancelled by user during batch processing.")
                 logger.warning("KeyboardInterrupt received during batch processing.")
                 fail_count = len(files_to_process) - success_count
                 break
            except Exception as e:
                 fail_count += 1
                 elapsed_file = time.time() - start_time_file
                 logger.error(f"Unexpected error in main loop for {file_path}: {e}", exc_info=True)
                 print(f"An unexpected error occurred processing {os.path.basename(file_path)} after {elapsed_file:.2f}s: {e} [UNEXPECTED ERROR]")

        # 4. Batch summary
        elapsed_batch = time.time() - start_time_batch
        print("\n--- Batch Processing Summary ---")
        print(f"Total files attempted: {len(files_to_process)}")
        print(f"  Successful: {success_count}")
        print(f"  Failed:     {fail_count}")
        print(f"Total time: {elapsed_batch:.2f}s")
        print("--------------------------------")

    else:
        if args.command is None:
             print("No command specified. Use 'help' for usage information.")
        else:
             print(f"Invalid command: '{args.command}'. Use 'help' for usage information.")
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        logger.warning("KeyboardInterrupt received at top level.")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"An unhandled error occurred: {e}", exc_info=True)
        print(f"\nA critical error occurred: {e}")
        sys.exit(1)