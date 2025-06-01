"""
Audio processing module with transcription capabilities.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Union, Any
import tempfile

logger = logging.getLogger(__name__)


class AudioProcessor:
    """
    Processes audio files and extracts text content using transcription.
    Supports MP3, WAV, M4A, and other common audio formats.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the audio processor with configuration options.

        Args:
            config: Configuration dictionary with processing options
        """
        self.config = config or {}
        
        # Set up transcription configuration
        self.transcription_config = self.config.get("transcription_config", {
            "model": "whisper-1",  # Default model
            "language": None,  # Auto-detect language
            "prompt": None,  # Optional prompt to guide transcription
            "response_format": "text",  # Options: text, vtt, srt, verbose_json
            "temperature": 0,  # Lower is more deterministic
            "timeout": 300,  # Timeout in seconds
        })
        
        # Check for API key
        self.api_key = os.environ.get("OPENAI_API_KEY")
        if not self.api_key and not self.config.get("offline_transcription", False):
            logger.warning("No OpenAI API key found. Set OPENAI_API_KEY environment variable for audio transcription.")
        
        self.model_loaded = False
        logger.info("Audio processor initialized with transcription config: %s", self.transcription_config)

    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process an audio file and transcribe text content.

        Args:
            file_path: Path to the audio file

        Returns:
            Dictionary with transcribed text content and metadata
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        start_time = time.time()
        logger.info(f"Processing audio file: {file_path}")
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        
        try:
            # Choose transcription method based on configuration
            if self.config.get("offline_transcription", False):
                text = self._offline_transcription(file_path)
            else:
                text = self._online_transcription(file_path)
            
            # Extract metadata
            metadata = self._extract_metadata(file_path, file_size_mb)
            
            processing_time = time.time() - start_time
            logger.info(f"Audio processing completed in {processing_time:.2f}s")
            
            return {
                "text": text,
                "metadata": metadata,
                "processing_stats": {
                    "processing_time_seconds": processing_time,
                    "file_size_mb": file_size_mb,
                    "transcription_method": "offline" if self.config.get("offline_transcription", False) else "openai",
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing audio {file_path}: {str(e)}")
            raise

    def _online_transcription(self, file_path: str) -> str:
        """
        Transcribe audio using OpenAI Whisper API.

        Args:
            file_path: Path to the audio file

        Returns:
            Transcribed text
        """
        try:
            import openai
            
            if not self.api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
            
            openai.api_key = self.api_key
            
            with open(file_path, "rb") as audio_file:
                response = openai.Audio.transcribe(
                    model=self.transcription_config.get("model", "whisper-1"),
                    file=audio_file,
                    language=self.transcription_config.get("language"),
                    prompt=self.transcription_config.get("prompt"),
                    response_format=self.transcription_config.get("response_format", "text"),
                    temperature=self.transcription_config.get("temperature", 0),
                    timeout=self.transcription_config.get("timeout", 300),
                )
                
            # Extract text from response based on response format
            if self.transcription_config.get("response_format") == "verbose_json":
                return response["text"]
            else:
                return response
                
        except ImportError:
            logger.error("OpenAI package not installed. Install with: pip install openai")
            raise
        except Exception as e:
            logger.error(f"Error in OpenAI transcription: {str(e)}")
            # Fall back to offline transcription if available
            if self.config.get("offline_transcription_fallback", True):
                logger.info("Falling back to offline transcription")
                return self._offline_transcription(file_path)
            else:
                raise

    def _offline_transcription(self, file_path: str) -> str:
        """
        Transcribe audio using local Whisper model.

        Args:
            file_path: Path to the audio file

        Returns:
            Transcribed text
        """
        try:
            import whisper
            
            # Load model if not already loaded
            if not hasattr(self, "model") or not self.model_loaded:
                model_size = self.config.get("offline_model_size", "base")
                logger.info(f"Loading Whisper {model_size} model...")
                self.model = whisper.load_model(model_size)
                self.model_loaded = True
                
            # Transcribe audio
            result = self.model.transcribe(
                file_path,
                language=self.transcription_config.get("language"),
                initial_prompt=self.transcription_config.get("prompt"),
                temperature=self.transcription_config.get("temperature", 0),
            )
            
            return result["text"]
                
        except ImportError:
            logger.error("Whisper package not installed. Install with: pip install openai-whisper")
            raise
        except Exception as e:
            logger.error(f"Error in offline transcription: {str(e)}")
            raise

    def _extract_metadata(self, file_path: str, file_size_mb: float) -> Dict[str, Any]:
        """
        Extract metadata from the audio file.

        Args:
            file_path: Path to the audio file
            file_size_mb: Size of the file in MB

        Returns:
            Dictionary with metadata
        """
        filename = os.path.basename(file_path)
        file_ext = os.path.splitext(filename)[1].lower()
        
        metadata = {
            "source": "audio",
            "source_type": file_ext.replace(".", ""),
            "filename": filename,
            "file_size_mb": file_size_mb,
        }
        
        # Try to extract audio-specific metadata if possible
        try:
            from mutagen import File
            
            audio = File(file_path)
            if audio is not None:
                # Add basic audio information
                metadata.update({
                    "duration_seconds": audio.info.length if hasattr(audio.info, 'length') else None,
                    "bitrate": audio.info.bitrate if hasattr(audio.info, 'bitrate') else None,
                    "channels": audio.info.channels if hasattr(audio.info, 'channels') else None,
                    "sample_rate": audio.info.sample_rate if hasattr(audio.info, 'sample_rate') else None,
                })
                
                # Add tags/metadata if available
                if hasattr(audio, 'tags') and audio.tags:
                    for key in ['title', 'artist', 'album', 'date']:
                        if key in audio.tags:
                            metadata[key] = str(audio.tags[key][0])
        except ImportError:
            logger.warning("Mutagen package not installed. Audio metadata will be limited.")
        except Exception as e:
            logger.warning(f"Error extracting audio metadata: {str(e)}")
        
        return metadata
