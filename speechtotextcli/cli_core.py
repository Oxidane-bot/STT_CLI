import whisper
import torch
import os
import io
import re
import sys
import time
import subprocess
import json
import tempfile
import gc
import logging
import threading
from tqdm import tqdm
from pathlib import Path

# Import from package
from speechtotextcli import utils

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

_MODEL = None
_CURRENT_TQDM_BAR = None # Global reference to the current tqdm bar

# --- Time Formatting Utilities ---
def format_timestamp(seconds):
    """将秒数格式化为SRT时间戳格式 (00:00:00,000)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds - int(seconds)) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

def format_duration(seconds):
    """将秒数格式化为人类可读的时间格式"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h{minutes}m{secs}s"
    elif minutes > 0:
        return f"{minutes}m{secs}s"
    else:
        return f"{secs}s"

def format_remaining_time(seconds):
    """将剩余秒数格式化为可读格式"""
    if seconds < 0: return "..."
    if seconds < 10: return "<10s"
    if seconds < 60: return f"~{int(seconds)}s"
    if seconds < 3600: return f"~{int(seconds / 60)}m"
    hours = int(seconds / 3600)
    minutes = int((seconds % 3600) / 60)
    return f"~{hours}h{minutes}m"

def timestamp_to_seconds(timestamp):
    """将[HH:]MM:SS.mmm格式的时间戳转换为秒"""
    parts = timestamp.split(':')
    if len(parts) == 2:  # MM:SS.mmm
        minutes, seconds_str = parts
        hours = 0
    elif len(parts) == 3: # HH:MM:SS.mmm
        hours, minutes, seconds_str = parts
    else:
        return 0 # Invalid format

    seconds_parts = seconds_str.split('.')
    if len(seconds_parts) == 2:
        seconds, milliseconds = seconds_parts
        try:
            milliseconds = float(f"0.{milliseconds}")
        except ValueError:
            milliseconds = 0
    elif len(seconds_parts) == 1:
        seconds = seconds_parts[0]
        milliseconds = 0
    else:
        return 0 # Invalid format

    try:
        total_seconds = (int(hours) * 3600) + \
                        (int(minutes) * 60) + \
                        int(seconds) + \
                        milliseconds
    except ValueError:
        return 0 # Invalid format

    return total_seconds

# --- Media Duration ---
def get_media_duration(file_path):
    """获取媒体文件的总时长(秒)"""
    try:
        # Check ffprobe availability using the utility function
        if not utils._is_tool_available("ffprobe"):
            logger.warning("ffprobe command not found. Cannot get exact media duration.")
            print("\nWarning: ffprobe not found in PATH. Cannot determine media duration accurately.")
            print("Please install ffmpeg (from https://ffmpeg.org/download.html or https://www.gyan.dev/ffmpeg/builds/)")
            print("and ensure it is added to your system's PATH environment variable.")
            return None

        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json', file_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
        info = json.loads(result.stdout)
        duration = float(info["format"]["duration"])
        return duration
    except Exception as e:
        logger.error(f"Error getting media duration for {file_path}: {e}")
        return None

# --- Whisper Output Capture with Time-Based Progress Bar ---
class TimeBasedProgressTracker:
    def __init__(self, total_duration, pbar):
        self.total_duration = total_duration if total_duration else 0  # in seconds
        self.pbar = pbar
        self.start_time = time.time()
        self.running = True
        self.detected_language = None
        self.language_detect_pattern = re.compile(r'Detected language: (\w+)')
        self.progress_update_interval = 0.25  # Update every 250ms
        self.last_timestamp = 0  # 最后从实际转录获取的时间戳
        self.last_real_update_time = time.time()  # 最后一次实际更新的时间
        self.time_calibrated = False  # 是否已经通过实际时间戳校准过
        self.speed_factor = 2.0  # 默认速度因子，会根据实际情况动态调整
        
    def _update_progress_thread(self):
        """Thread function to update progress bar based on elapsed time"""
        last_update_time = 0
        
        while self.running and self.pbar:
            # 如果没有收到实际时间戳的更新，使用时间估算
            if not self.time_calibrated or time.time() - self.last_real_update_time > 3.0:
                # 计算基于时间的估计进度
                elapsed_seconds = time.time() - self.start_time
                
                # 计算估计位置
                if self.total_duration > 0:
                    # 使用当前的速度因子
                    estimated_position = min(self.last_timestamp + (elapsed_seconds - self.last_real_update_time) * self.speed_factor,
                                            self.total_duration)
                    
                    # 只有当有足够大的变化时才更新
                    if estimated_position - last_update_time > 0.2:
                        update_amount = estimated_position - last_update_time
                        self.pbar.update(update_amount)
                        last_update_time = estimated_position
                        
                        # 计算并显示ETA
                        if estimated_position > 0:
                            progress_ratio = estimated_position / self.total_duration
                            if progress_ratio > 0.02:  # 避免早期不稳定估计
                                estimated_total_time = elapsed_seconds / progress_ratio
                                remaining_time = estimated_total_time - elapsed_seconds
                                self.pbar.set_postfix_str(f"ETA: {format_remaining_time(remaining_time)}", refresh=True)
            
            # 短暂休眠以便下次更新
            time.sleep(self.progress_update_interval)
    
    def update_with_timestamp(self, timestamp_seconds):
        """使用实际时间戳更新进度"""
        if self.pbar and self.total_duration > 0:
            if timestamp_seconds > self.last_timestamp:
                # 计算进度更新量
                update_amount = timestamp_seconds - self.last_timestamp
                
                # 计算基于实际处理速度的速度因子
                current_time = time.time()
                time_elapsed = current_time - self.last_real_update_time
                
                if time_elapsed > 0.1:  # 确保有足够的时间经过以计算有意义的速度
                    # 计算新的速度因子 = 实际进度/经过时间
                    new_speed_factor = update_amount / time_elapsed
                    
                    # 平滑速度因子的变化
                    if self.time_calibrated:
                        self.speed_factor = self.speed_factor * 0.7 + new_speed_factor * 0.3
                    else:
                        self.speed_factor = new_speed_factor
                        self.time_calibrated = True
                
                # 更新进度条
                self.pbar.update(update_amount)
                self.last_timestamp = timestamp_seconds
                self.last_real_update_time = current_time
                
                # 更新ETA
                if self.total_duration > 0 and timestamp_seconds > 0:
                    elapsed_seconds = time.time() - self.start_time
                    progress_ratio = timestamp_seconds / self.total_duration
                    if progress_ratio > 0.02:
                        estimated_total_time = elapsed_seconds / progress_ratio
                        remaining_time = estimated_total_time - elapsed_seconds
                        self.pbar.set_postfix_str(f"ETA: {format_remaining_time(remaining_time)}", refresh=True)
                        
                # 刷新进度条使更改可见
                self.pbar.refresh()
    
    def start(self):
        """Start the progress tracking thread"""
        if self.total_duration > 0 and self.pbar:
            self.progress_thread = threading.Thread(target=self._update_progress_thread)
            self.progress_thread.daemon = True
            self.progress_thread.start()
    
    def stop(self):
        """Stop the progress tracking thread"""
        self.running = False
        if hasattr(self, 'progress_thread') and self.progress_thread.is_alive():
            self.progress_thread.join(timeout=1.0)  # Wait for thread to terminate
            
    def process_output(self, text):
        """Process output text for language detection"""
        # Check for language detection
        if self.detected_language is None:
            lang_match = self.language_detect_pattern.search(text)
            if lang_match:
                self.detected_language = lang_match.group(1)
                if self.pbar:
                    self.pbar.set_description(f"Transcribing (Lang: {self.detected_language})")
    
    def finalize_progress(self):
        """Ensure the progress bar reaches 100% at the end."""
        self.stop()  # Stop the progress thread
        
        if self.pbar and self.total_duration > 0:
            # Set progress to 100%
            remaining_update = self.total_duration - self.pbar.n
            if remaining_update > 0.01:  # Update only if there's a noticeable difference
                self.pbar.update(remaining_update)
            # Ensure the final value is exactly the total duration
            self.pbar.n = self.total_duration
            self.pbar.refresh()  # Refresh to show the final state
            self.pbar.set_postfix_str("Done!", refresh=True)

class CliOutputRedirector(io.StringIO):
    def __init__(self, total_duration, pbar):
        super().__init__()
        self.progress_tracker = TimeBasedProgressTracker(total_duration, pbar)
        self.progress_tracker.start()  # Start progress tracking thread
        self.segment_pattern = re.compile(r'\[\s*(\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}\.\d{3})\s*\]\s*(.*)')
        self.last_timestamp_seconds = 0  # 跟踪最后处理的时间戳
        
    def write(self, text):
        super().write(text)  # Keep internal buffer if needed
        
        # Process for language detection
        self.progress_tracker.process_output(text)
        
        # 解析时间戳用于更新进度条，但不打印输出
        text = text.strip()
        if text.startswith('[') and '-->' in text:
            match = self.segment_pattern.search(text)
            if match:
                try:
                    _, end_time_str, _ = match.groups()
                    # 将结束时间转换为秒
                    end_sec = timestamp_to_seconds(end_time_str.replace(',', '.'))
                    if end_sec > self.last_timestamp_seconds:
                        # 更新进度条，但不强制覆盖基于时间的估算，只是提供更准确的参考点
                        self.progress_tracker.update_with_timestamp(end_sec)
                        self.last_timestamp_seconds = end_sec
                except Exception:
                    # 如果解析失败，保持使用时间估算
                    pass
        
    def finalize_progress(self):
        """Ensure the progress bar reaches 100% at the end."""
        self.progress_tracker.finalize_progress()

def _extract_audio_cli(video_file, temp_dir):
    """Extracts audio using ffmpeg, showing progress via prints."""
    logger.info(f"Extracting audio from: {video_file}")
    
    # Check if ffmpeg is available before proceeding
    if not utils._is_tool_available("ffmpeg"):
        logger.error("ffmpeg command not found. Please install ffmpeg and add it to your system's PATH to process video files.")
        print("\nError: ffmpeg is required to extract audio from video files.")
        print("Please install ffmpeg (from https://ffmpeg.org/download.html or https://www.gyan.dev/ffmpeg/builds/)")
        print("and ensure it is added to your system's PATH environment variable.")
        return None
        
    duration = get_media_duration(video_file)
    if duration is None:
        logger.warning("Could not get video duration, cannot show extraction progress.")
        duration = 0 # Set to 0 if unknown

    temp_audio_path = os.path.join(temp_dir, "temp_audio.wav")
    print("  Extracting audio...", end='', flush=True)

    # Base command
    base_cmd = [
        'ffmpeg', '-y',
        '-i', video_file,
        '-vn',
        '-acodec', 'pcm_s16le',
        '-ar', '16000', # Sample rate expected by Whisper
        '-ac', '1',     # Mono channel
        temp_audio_path
    ]

    # Try with progress reporting if duration is known
    if duration > 0:
        cmd = base_cmd + ['-progress', 'pipe:1']
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Capture stderr as well
            text=True,
            encoding="utf-8",
            errors="replace"
        )

        last_progress_percent = -1
        try:
            for line in process.stdout:
                line = line.strip()
                if line.startswith("out_time_ms="):
                    try:
                        out_time_ms = int(line.split("=")[1])
                        progress_percent = int((out_time_ms / (duration * 1000000)) * 100) # out_time is in microseconds
                        progress_percent = min(progress_percent, 100)
                        if progress_percent > last_progress_percent:
                            print(f"\r  Extracting audio... {progress_percent}%", end='', flush=True)
                            last_progress_percent = progress_percent
                    except (ValueError, IndexError, ZeroDivisionError):
                        pass # Ignore parsing errors
                elif "error" in line.lower():
                     logger.warning(f"ffmpeg output: {line}") # Log potential ffmpeg errors/warnings

            process.wait()
            print("\r  Extracting audio... Done.   ") # Clear progress line
            if process.returncode != 0:
                 # Read stderr if available
                 stderr_output = process.stderr.read() if process.stderr else "N/A"
                 raise Exception(f"ffmpeg failed with code {process.returncode}. Stderr: {stderr_output}")

        except Exception as e:
             print("\r  Extracting audio... Failed.")
             logger.error(f"Audio extraction failed: {e}")
             # Fallback to simple extraction if progress parsing failed
             try:
                 print("  Retrying audio extraction (without progress)...", end='', flush=True)
                 subprocess.run(base_cmd, check=True, capture_output=True)
                 print("\r  Retrying audio extraction... Done.          ")
             except Exception as fallback_e:
                 print("\r  Retrying audio extraction... Failed.")
                 logger.error(f"Fallback audio extraction failed: {fallback_e}")
                 return None

    else: # Duration unknown, run without progress pipe
        try:
            print("\r  Extracting audio (duration unknown)...", end='', flush=True)
            subprocess.run(base_cmd, check=True, capture_output=True)
            print("\r  Extracting audio... Done.             ")
        except Exception as e:
            print("\r  Extracting audio... Failed.")
            logger.error(f"Audio extraction failed (no duration): {e}")
            return None

    if not os.path.exists(temp_audio_path) or os.path.getsize(temp_audio_path) == 0:
        logger.error("Extracted audio file is empty or missing.")
        return None

    logger.info(f"Audio extracted successfully to {temp_audio_path}")
    return temp_audio_path


def transcribe_audio(input_file, model_name="tiny", language="auto", output_format="srt"):
    global _MODEL, _CURRENT_TQDM_BAR
    if _MODEL is None:
        logger.error("Model not loaded. Please load a model first using the 'load_model' command.")
        print("Error: Model not loaded.")
        return False # Indicate failure

    logger.info(f"Starting transcription for: {input_file}")
    print(f"\nProcessing: {os.path.basename(input_file)}")

    temp_dir = None
    audio_path_to_transcribe = input_file
    is_temp_audio = False

    try:
        # --- 1. Handle Input File (Extract Audio if Video) ---
        file_path_obj = Path(input_file)
        if file_path_obj.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']:
            temp_dir = tempfile.TemporaryDirectory()
            extracted_audio = _extract_audio_cli(input_file, temp_dir.name)
            if not extracted_audio:
                logger.error("Audio extraction failed.")
                print("Error: Could not extract audio from video.")
                return False # Indicate failure
            audio_path_to_transcribe = extracted_audio
            is_temp_audio = True
        elif not file_path_obj.suffix.lower() in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']:
             logger.warning(f"Input file '{input_file}' might not be a supported audio/video format.")
             print(f"Warning: Unrecognized file extension '{file_path_obj.suffix}'. Attempting transcription anyway.")


        # --- 2. Get Duration and Setup Progress Bar ---
        total_duration = get_media_duration(audio_path_to_transcribe)
        if total_duration is None:
            logger.warning("Could not determine audio duration. Progress bar will not show time estimates.")
            print("Info: Could not get audio duration. Progress bar may be less accurate.")
            total_duration = 0 # Set to 0 for tqdm if unknown

        # Setup tqdm progress bar
        _CURRENT_TQDM_BAR = tqdm(
            total=total_duration,
            unit='s',
            unit_scale=True,
            desc="Transcribing",
            leave=False, # Keep the bar after completion
            bar_format='{l_bar}{bar}| {n:.1f}/{total:.1f}s [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
        )

        # --- 3. Transcribe with Progress ---
        original_stdout = sys.stdout
        output_capture = CliOutputRedirector(total_duration, _CURRENT_TQDM_BAR)
        sys.stdout = output_capture # Redirect stdout

        result = None
        transcription_successful = False
        try:
            logger.info(f"Calling whisper.transcribe (Model: {model_name}, Lang: {language}, File: {audio_path_to_transcribe})")
            transcribe_kwargs = {"verbose": True}
            if language and language != "auto":
                transcribe_kwargs["language"] = language

            result = _MODEL.transcribe(audio_path_to_transcribe, **transcribe_kwargs)
            transcription_successful = True
            logger.info("Whisper transcription call completed.")

        except Exception as transcribe_error:
            logger.error(f"Whisper transcription failed: {transcribe_error}", exc_info=True)
            print(f"\nError during transcription: {transcribe_error}")
            # Keep result=None

        finally:
            # --- 4. Restore Output and Finalize Progress ---
            sys.stdout = original_stdout # Restore stdout IMPORTANTLY
            output_capture.finalize_progress()
            if _CURRENT_TQDM_BAR:
                _CURRENT_TQDM_BAR.close()
                _CURRENT_TQDM_BAR = None
            gc.collect() # Suggest garbage collection

        # --- 5. Save Results ---
        if transcription_successful and result:
            output_filename_base = os.path.splitext(input_file)[0] # Use original input filename base
            output_file = f"{output_filename_base}.{output_format}"
            save_successful = utils.save_transcription(result, output_file, output_format)
            if save_successful:
                print(f"Transcription saved to: {output_file}")
                return True # Indicate success
            else:
                print(f"Error: Failed to save transcription to {output_file}")
                return False # Indicate failure
        else:
            print("Transcription failed or produced no result.")
            return False # Indicate failure

    except Exception as e:
        logger.error(f"Error processing {input_file}: {e}", exc_info=True)
        print(f"\nAn unexpected error occurred while processing {os.path.basename(input_file)}: {e}")
        # Ensure progress bar is closed on unexpected errors
        if _CURRENT_TQDM_BAR:
            _CURRENT_TQDM_BAR.close()
            _CURRENT_TQDM_BAR = None
        return False # Indicate failure
    finally:
        # --- 6. Cleanup ---
        if temp_dir:
            try:
                temp_dir.cleanup()
                logger.info("Temporary directory cleaned up.")
            except Exception as cleanup_error:
                logger.error(f"Failed to cleanup temporary directory: {cleanup_error}")
        # Ensure model reference is not held if not needed? (Maybe not here, depends on usage pattern)
        gc.collect()


def load_model(model_name):
    global _MODEL
    try:
        logger.info(f"Attempting to load Whisper model: {model_name}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        _MODEL = whisper.load_model(model_name, device=device)
        print(f"  Model '{model_name}' loaded successfully on {device}.")
        logger.info(f"Model {model_name} loaded successfully.")
        return True # Indicate success
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {e}", exc_info=True)
        print(f"  Error loading model '{model_name}': {e}")
        _MODEL = None
        return False # Indicate failure