import os
import logging
import shutil

logger = logging.getLogger(__name__)

# Supported file extensions (add more as needed)
SUPPORTED_EXTENSIONS = (
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", # Video
    ".wav", ".mp3", ".m4a", ".ogg", ".flac"        # Audio
)

def _is_tool_available(name):
    """Check whether `name` is on PATH and marked as executable."""
    return shutil.which(name) is not None

def get_media_files(directory):
    """Finds supported audio and video files in a directory."""
    media_files = []
    try:
        for filename in os.listdir(directory):
            if filename.lower().endswith(SUPPORTED_EXTENSIONS):
                full_path = os.path.join(directory, filename)
                if os.path.isfile(full_path): # Ensure it's a file
                     media_files.append(full_path)
    except FileNotFoundError:
        logger.error(f"Directory not found: {directory}")
        print(f"Error: Directory not found - {directory}")
    except Exception as e:
        logger.error(f"Error listing files in {directory}: {e}")
        print(f"Error listing files in {directory}: {e}")
    return media_files

# format_timestamp is needed by save_transcription if outputting SRT
def format_timestamp(seconds):
    """将秒数格式化为SRT时间戳格式 (00:00:00,000)"""
    if not isinstance(seconds, (int, float)) or seconds < 0:
        seconds = 0 # Default to 0 if invalid
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def save_transcription(transcription, output_file, output_format="srt"):
    """Saves the transcription result to a file."""
    logger.info(f"Saving transcription to {output_file} (Format: {output_format})")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            if output_format.lower() == "srt":
                if "segments" not in transcription or not transcription["segments"]:
                    logger.warning("No segments found in transcription for SRT output.")
                    f.write("") # Write empty file for consistency
                    return True # Still considered "successful" save of empty content

                segment_id = 1
                for segment in transcription["segments"]:
                    # Ensure keys exist and handle potential None values
                    start_time_sec = segment.get("start")
                    end_time_sec = segment.get("end")
                    text = segment.get("text", "").strip()

                    # Basic validation
                    if start_time_sec is None or end_time_sec is None:
                        logger.warning(f"Skipping segment {segment_id} due to missing time data: {segment}")
                        continue

                    start_time = format_timestamp(start_time_sec)
                    end_time = format_timestamp(end_time_sec)

                    f.write(f"{segment_id}\n")
                    f.write(f"{start_time} --> {end_time}\n")
                    f.write(f"{text}\n\n")
                    segment_id += 1
            elif output_format.lower() == "txt":
                full_text = transcription.get("text", "")
                f.write(full_text)
            else:
                logger.error(f"Unsupported output format: {output_format}")
                print(f"Error: Unsupported output format '{output_format}'. Use 'srt' or 'txt'.")
                return False

        logger.info(f"Successfully saved transcription to {output_file}")
        return True # Indicate success
    except KeyError as e:
         logger.error(f"Missing key in transcription data: {e}. Data: {transcription}")
         print(f"Error saving transcription: Missing expected data field '{e}'.")
         return False
    except Exception as e:
        logger.error(f"Error saving transcription to {output_file}: {e}", exc_info=True)
        print(f"Error: Could not save transcription to {output_file}: {e}")
        return False # Indicate failure