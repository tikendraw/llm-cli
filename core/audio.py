import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from groq import Groq
from groq.types.audio import Transcription, Translation
from pydub import AudioSegment

from core.chunking_strategies import (
    ChunkingStrategy,
    SilenceBasedChunkStrategy,
    SizeBasedChunkStrategy,
    TimedChunkStrategy,
)

logger = logging.getLogger(__name__)

@dataclass
class DummyTranscription:
    text: str
    segments: List[dict]
    language: str
    
@dataclass
class Segment:
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float

@dataclass
class TranscriptionResult:
    text: str
    segments: List[Segment]
    language: str

class AudioProcessor:
    """A class to process large audio files for transcription or translation using Groq API."""
    
    # Maximum file size allowed by Groq API (in bytes)
    MAX_FILE_SIZE: int = 25 * 1024 * 1024  # 25 MB
    
    # Overlap duration in milliseconds between chunks
    OVERLAP_DURATION: int = 10 * 1000  # 10 seconds
    
    # Available models
    AVAILABLE_MODELS: List[str] = [
        'whisper-large-v3', 
        'whisper-large-v3-turbo', 
        'distil-whisper-large-v3-en'
    ]
    
    def __init__(
        self, 
        groq_api_key: Optional[str] = None,
        chunking_strategy: Optional[ChunkingStrategy] = None
    ) -> None:
        """
        Initialize the AudioProcessor with the Groq API client.
        
        Args:
            groq_api_key: Optional API key for Groq. If not provided, will use GROQ_API_KEY environment variable.
            chunking_strategy: Strategy to use for chunking audio files. Defaults to SizeBasedChunkStrategy.
        """
        self.client = Groq(api_key=groq_api_key)
        self.chunking_strategy = chunking_strategy or SizeBasedChunkStrategy(
            max_size_bytes=self.MAX_FILE_SIZE,
            overlap_ms=self.OVERLAP_DURATION
        )

    def preprocess_audio(self, file_path: str, output_path: Optional[str] = None) -> str:
        """
        Preprocess audio file to optimize for speech recognition:
        - Downsample to 16kHz
        - Convert to mono
        - Export as FLAC for lossless compression
        
        Args:
            file_path: Path to the input audio file
            output_path: Optional path for the output file. If not provided, will create a temporary file.
            
        Returns:
            Path to the preprocessed audio file
        """
        if output_path is None:
            output_dir = tempfile.gettempdir()
            output_file = os.path.join(output_dir, f"preprocessed_{os.path.basename(file_path)}.flac")
        else:
            output_file = output_path
            
        # Ensure output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            
        try:
            cmd = [
                "ffmpeg",
                "-i", file_path,
                "-ar", "16000",
                "-ac", "1",
                "-map", "0:a",
                "-c:a", "flac",
                output_file
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Audio preprocessed and saved to {output_file}")
            return output_file
            
        except subprocess.CalledProcessError as e:
            logger.error(f"FFmpeg preprocessing failed: {e.stderr.decode()}")
            raise RuntimeError(f"Audio preprocessing failed: {e}")
    
    def get_audio_file_size(self, file_path: str) -> int:
        """Get the size of an audio file in bytes."""
        return os.path.getsize(file_path)
    
    def chunk_audio(self, file_path: str, temp_dir: Optional[str] = None) -> List[str]:
        """
        Chunk audio file using the configured strategy
        
        Args:
            file_path: Path to the audio file
            temp_dir: Optional directory to store temporary chunks
            
        Returns:
            List of paths to chunked audio files
        """
        if temp_dir is None:
            temp_dir = tempfile.mkdtemp()
        else:
            os.makedirs(temp_dir, exist_ok=True)
            
        audio = AudioSegment.from_file(file_path)
        return self.chunking_strategy.chunk_audio(audio, temp_dir)
    
    def merge_transcriptions(self, transcriptions: List[Union[Transcription, str, dict]], chunk_files: List[str], response_format: str = "verbose_json") -> Union[str, dict, TranscriptionResult]:
        """Merge multiple transcription results from chunks"""
        if not transcriptions:
            return "" if response_format == "text" else TranscriptionResult(text="", segments=[], language="")
            
        # For text response format, just concatenate with spaces
        if response_format == "text":
            return " ".join(t if isinstance(t, str) else t.text for t in transcriptions)
            
        # For json response format, return dict with text only
        if response_format == "json":
            full_text = " ".join(t if isinstance(t, str) else t.text for t in transcriptions)
            return {"text": full_text}

        # For verbose_json, proceed with full merge including timestamps
        if len(transcriptions) == 1:
            segments = [
                Segment(
                    id=s["id"],
                    seek=s["seek"],
                    start=s["start"],
                    end=s["end"],
                    text=s["text"],
                    tokens=s["tokens"],
                    temperature=s["temperature"],
                    avg_logprob=s["avg_logprob"],
                    compression_ratio=s["compression_ratio"],
                    no_speech_prob=s["no_speech_prob"]
                )
                for s in transcriptions[0].segments
            ]
            return TranscriptionResult(
                text=transcriptions[0].text,
                segments=segments,
                language=getattr(transcriptions[0], "language", "en")
            )

        def get_chunk_duration(chunk_path: str) -> float:
            """Get duration of audio chunk in seconds using pydub"""
            audio = AudioSegment.from_file(chunk_path)
            return len(audio) / 1000.0  # Convert ms to seconds

        # Calculate offsets based on actual chunk durations
        chunk_durations = [get_chunk_duration(chunk) for chunk in chunk_files]
        chunk_offsets = [0]  # First chunk starts at 0
        total_offset = 0

        for i in range(1, len(chunk_durations)):
            # Each chunk should start at: previous_offset + prev_duration - overlap
            total_offset += chunk_durations[i-1] - (self.OVERLAP_DURATION / 1000)
            chunk_offsets.append(total_offset)

        all_segments = []
        overlap_threshold = self.OVERLAP_DURATION / 1000  # Convert to seconds

        # Process each chunk with correct offset
        for i, trans in enumerate(transcriptions):
            offset = chunk_offsets[i]

            # Debug printing
            # print(f'Chunk {i} (offset: {offset:.2f}s) {"="*20}')
            # print(f'{i}-'*22)
            
            for s in trans.segments:
                # For non-first chunks, skip segments entirely within overlap region
                if i > 0 and s["end"] < overlap_threshold:
                    continue

                # Apply offset to timestamps
                segment = Segment(
                    id=len(all_segments),
                    seek=s["seek"],
                    start=s["start"] + offset,
                    end=s["end"] + offset,
                    text=s["text"].strip(),
                    tokens=s["tokens"],
                    temperature=s["temperature"],
                    avg_logprob=s["avg_logprob"],
                    compression_ratio=s["compression_ratio"],
                    no_speech_prob=s["no_speech_prob"]
                )

                # print(f"Segment {s['id']:3} {s['start']:5.1f}-{s['end']:5.1f} -> "
                #       f"{segment.start:5.1f}-{segment.end:5.1f}: {s['text']}")
                
                all_segments.append(segment)

        # Sort segments by start time
        all_segments.sort(key=lambda s: s.start)

        # Deduplicate overlapping segments
        merged_segments = []
        i = 0
        while i < len(all_segments):
            current = all_segments[i]
            
            # Look ahead for overlapping segments
            j = i + 1
            while j < len(all_segments) and all_segments[j].start <= current.end:
                next_seg = all_segments[j]
                
                # Calculate text similarity ratio
                from difflib import SequenceMatcher
                similarity = SequenceMatcher(None, current.text.lower(), next_seg.text.lower()).ratio()
                
                # If segments have similar text (>70% similarity), keep the one with better confidence
                if similarity > 0.7:
                    if current.avg_logprob < next_seg.avg_logprob:
                        current = next_seg
                    j += 1
                    continue
                    
                # If minimal overlap and different text, keep both
                if next_seg.start > current.end - 1.0:  # Less than 1 second overlap
                    break
                    
                j += 1
            
            merged_segments.append(current)
            i = j if j > i + 1 else i + 1

        # Recalculate IDs and ensure no gaps between segments
        for i, segment in enumerate(merged_segments):
            segment.id = i
            if i > 0:
                prev_segment = merged_segments[i-1]
                if segment.start - prev_segment.end > 0.1:  # Small gap
                    segment.start = prev_segment.end

        return TranscriptionResult(
            text=" ".join(s.text for s in merged_segments),
            segments=merged_segments,
            language=max(set(getattr(t, "language", "en") for t in transcriptions), 
                        key=lambda x: sum(1 for t in transcriptions if getattr(t, "language", "en") == x))
        )
    
    def process_audio(
        self, 
        file_path: str, 
        mode: Literal["transcribe", "translate"] = "transcribe",
        model: str = "whisper-large-v3",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        preprocess: bool = False,
        cleanup: bool = True,
        response_format: Literal["text", "json", "verbose_json"] = "verbose_json"
    ) -> Union[str, dict, TranscriptionResult]:
        """
        Process audio file for transcription or translation:
        1. Optionally preprocess the audio
        2. Check if chunking is needed
        3. Chunk if necessary
        4. Process each chunk
        5. Merge results
        6. Clean up temporary files
        
        Args:
            file_path: Path to the audio file
            mode: "transcribe" or "translate"
            model: Groq model to use
            language: Language code (e.g., "en", "fr")
            prompt: Optional prompt to guide transcription
            preprocess: Whether to preprocess the audio file
            cleanup: Whether to clean up temporary files
            response_format: Format of the response ("text", "json", or "verbose_json")
            
        Returns:
            TranscriptionResult object with text and segments
        """
        # Validate model
        if model not in self.AVAILABLE_MODELS:
            raise ValueError(f"Invalid model. Choose from: {', '.join(self.AVAILABLE_MODELS)}")
            
        # Validate mode
        if mode not in ["transcribe", "translate"]:
            raise ValueError("Mode must be either 'transcribe' or 'translate'")
            
        # Preprocess audio if requested
        processed_file = file_path
        temp_dir = None
        chunk_files = []
        
        try:
            if preprocess:
                logger.info("Preprocessing audio file...")
                processed_file = self.preprocess_audio(file_path)
                
            # Check if chunking is needed
            file_size = self.get_audio_file_size(processed_file)
            
            if file_size > self.MAX_FILE_SIZE:
                logger.info(f"File size ({file_size/1024/1024:.2f} MB) exceeds limit. Chunking...")
                temp_dir = tempfile.mkdtemp()
                chunk_files = self.chunk_audio(processed_file, temp_dir)  # Store chunk files
                logger.info(f"Created {len(chunk_files)} chunks")
            else:
                # No chunking needed
                chunk_files = [processed_file]
                
            # Process each chunk
            transcriptions = []
            
            for i, chunk_path in enumerate(chunk_files):
                logger.info(f"Processing chunk {i+1}/{len(chunk_files)}...")
                
                with open(chunk_path, "rb") as audio_file:
                    if mode == "transcribe":
                        kwargs = {
                            "model": model,
                            "file": audio_file,
                            "response_format": response_format,
                        }
                        
                        if prompt:
                            kwargs["prompt"] = prompt
                            
                        if language:
                            kwargs["language"] = language
                            
                        result = self.client.audio.transcriptions.create(**kwargs)
                    else:  # translate
                        kwargs = {
                            "model": model,
                            "file": audio_file,
                            "response_format": response_format,
                        }
                        
                        if prompt:
                            kwargs["prompt"] = prompt
                            
                        result = self.client.audio.translations.create(**kwargs)
                        
                    transcriptions.append(result)
                    
            # Merge results with appropriate response format
            final_result = self.merge_transcriptions(transcriptions, chunk_files, response_format)
            
            return final_result
            
        finally:
            # Clean up temporary files
            if cleanup:
                try:
                    # Clean up preprocessed file if it's different from original
                    if preprocess and processed_file != file_path:
                        os.unlink(processed_file)
                        
                    # Clean up chunk files and temp directory
                    if temp_dir and os.path.exists(temp_dir):
                        for chunk_file in chunk_files:
                            if os.path.exists(chunk_file) and chunk_file != file_path and chunk_file != processed_file:
                                os.unlink(chunk_file)
                        os.rmdir(temp_dir)
                except Exception as e:
                    logger.warning(f"Error during cleanup: {e}")

def transcribe_audio(
    file_path: str,
    model: str = "whisper-large-v3",
    language: Optional[str] = None,
    prompt: Optional[str] = None,
    preprocess: bool = False,
    api_key: Optional[str] = None,
    response_format: str = "verbose_json"
) -> Union[str, dict, TranscriptionResult]:
    """
    Transcribe audio file to text with optional preprocessing and chunking.
    
    Args:
        file_path: Path to the audio file
        model: Groq model to use
        language: Language code (e.g., "en", "fr")
        prompt: Optional prompt to guide transcription
        preprocess: Whether to preprocess the audio file
        api_key: Optional Groq API key
        response_format: Format of the response ("text", "json", or "verbose_json")
        
    Returns:
        TranscriptionResult with text and segments
    """
    processor = AudioProcessor(groq_api_key=api_key)
    return processor.process_audio(
        file_path=file_path,
        mode="transcribe",
        model=model,
        language=language,
        prompt=prompt,
        preprocess=preprocess,
        response_format=response_format
    )

def translate_audio(
    file_path: str,
    model: str = "whisper-large-v3",
    prompt: Optional[str] = None,
    preprocess: bool = False,
    api_key: Optional[str] = None,
    response_format: str = "verbose_json"
) -> Union[str, dict, TranscriptionResult]:
    """
    Translate audio file to English with optional preprocessing and chunking.
    
    Args:
        file_path: Path to the audio file
        model: Groq model to use
        prompt: Optional prompt to guide translation
        preprocess: Whether to preprocess the audio file
        api_key: Optional Groq API key
        response_format: Format of the response ("text", "json", or "verbose_json")
        
    Returns:
        TranscriptionResult with translated text and segments
    """
    processor = AudioProcessor(groq_api_key=api_key)
    return processor.process_audio(
        file_path=file_path,
        mode="translate",
        model=model,
        prompt=prompt,
        preprocess=preprocess,
        response_format=response_format
    )

# # Example usage
# if __name__ == "__main__":
#     import argparse
    
#     parser = argparse.ArgumentParser(description="Transcribe or translate audio files using Groq API")
#     parser.add_argument("file", help="Path to audio file")
#     parser.add_argument("--mode", choices=["transcribe", "translate"], default="transcribe", 
#                         help="Mode: transcribe or translate")
#     parser.add_argument("--model", default="whisper-large-v3", 
#                         help="Model to use (whisper-large-v3, whisper-large-v3-turbo, distil-whisper-large-v3-en)")
#     parser.add_argument("--language", help="Language code for transcription (e.g., en, fr)")
#     parser.add_argument("--prompt", help="Prompt to guide transcription/translation")
#     parser.add_argument("--preprocess", action="store_true", help="Preprocess audio before processing")
#     parser.add_argument("--output", help="Output file path for transcription/translation text")
    
#     args = parser.parse_args()
    
#     try:
#         processor = AudioProcessor(
#             # chunking_strategy=TimedChunkStrategy(chunk_duration_ms=60000),
#             # chunking_strategy=SilenceBasedChunkStrategy(),
#             chunking_strategy=SizeBasedChunkStrategy(overlap_ms=20000)
#         )
        
#         result = processor.process_audio(
#             file_path=args.file,
#             mode=args.mode,
#             model=args.model,
#             language=args.language,
#             prompt=args.prompt,
#             preprocess=args.preprocess
#         )
        
#         print("\nTranscription/Translation:")
#         print(result.text)
#         print("\nSegments:")
#         for segment in result.segments:
#             print(f"{segment.id:2} {segment.start:5.1f}-{segment.end:5.1f} -> {segment.text}")
            
#         # Print segments with timing
#         print("\nSegments:")
#         for segment in result.segments:
#             print(f"{segment.id:2} {segment.start:5.1f}-{segment.end:5.1f} -> {segment.text}")
            
#         # Save to output file if specified
#         if args.output:
#             with open(args.output, "w", encoding="utf-8") as f:
#                 f.write(result.text)
#             print(f"\nSaved to {args.output}")
            
#     except Exception as e:
#         print(f"Error: {e}")

