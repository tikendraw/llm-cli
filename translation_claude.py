import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from chunking_strategies import ChunkingStrategy, SizeBasedChunkStrategy, TimedChunkStrategy
from groq import Groq
from groq.types.audio import Transcription, Translation
from pydub import AudioSegment

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
    
    def merge_transcriptions(self, transcriptions: List[Transcription]) -> TranscriptionResult:
        """
        Merge multiple transcription results from chunks:
        1. First pass: combine all segments
        2. Second pass: identify and merge overlapping segments based on text similarity
        
        Args:
            transcriptions: List of Transcription objects from Groq API
            
        Returns:
            Merged TranscriptionResult
        """
        if not transcriptions:
            return TranscriptionResult(text="", segments=[], language="")
            
        # If there's only one chunk, no need to merge
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
                language=getattr(transcriptions[0], "language", "en")  # Default to English if not provided
            )
            
        # Combine all segments
        all_segments = []
        chunk_durations = [0]  # Starting offset for each chunk
        
        for i, trans in enumerate(transcriptions):
            # Skip first chunk's offset calculation
            if i > 0:
                # Calculate time offset for the current chunk
                # Subtract overlap duration from the current chunk's start time
                prev_chunk_duration = chunk_durations[-1]
                offset = prev_chunk_duration - self.OVERLAP_DURATION / 1000
                chunk_durations.append(offset + max([s["end"] for s in trans.segments], default=0))
                print(f'{i}-'*22)
                for j in trans.segments:
                    print(f"{j['id']:3} {j['start']:5}-{j['end']:5} -> {j['text']}")
                print()

                
            else:
                offset = 0
                chunk_durations.append(max([s["end"] for s in trans.segments], default=0))
                print(f'{i}-'*22)
                for j in trans.segments:
                    print(f"{j['id']:3} {j['start']:5}-{j['end']:5} -> {j['text']}")
                print()

                
            # Add segments with adjusted timing
            for s in trans.segments:
                # Skip segments that are completely in the overlap region for non-first chunks
                if i > 0 and s["end"] < self.OVERLAP_DURATION / 1000:
                    continue
                    
                segment = Segment(
                    id=len(all_segments),
                    seek=s["seek"],
                    start=s["start"] + offset if i > 0 else s["start"],
                    end=s["end"] + offset if i > 0 else s["end"],
                    text=s["text"],
                    tokens=s["tokens"],
                    temperature=s["temperature"],
                    avg_logprob=s["avg_logprob"],
                    compression_ratio=s["compression_ratio"],
                    no_speech_prob=s["no_speech_prob"]
                )
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
            while j < len(all_segments) and all_segments[j].start < current.end:
                # If significant overlap in timing, check for text similarity
                next_seg = all_segments[j]
                
                # Simple text similarity check (could be improved with more sophisticated methods)
                if len(current.text) > 20 and len(next_seg.text) > 20:
                    # Check if one is a substring of the other (with some flexibility)
                    if current.text in next_seg.text or next_seg.text in current.text:
                        # Merge by keeping the one with better confidence (higher avg_logprob)
                        if current.avg_logprob < next_seg.avg_logprob:
                            current = next_seg
                        # Skip this segment in further processing
                        j += 1
                        continue
                        
                j += 1
                
            merged_segments.append(current)
            i = j if j > i + 1 else i + 1
            
        # Recalculate IDs
        for i, segment in enumerate(merged_segments):
            segment.id = i
            
        # Combine text from all segments
        full_text = " ".join(s.text for s in merged_segments)
        
        # Determine language (use the most common language from all chunks)
        languages = [getattr(t, "language", "en") for t in transcriptions]
        language = max(set(languages), key=languages.count)
        
        return TranscriptionResult(
            text=full_text,
            segments=merged_segments,
            language=language
        )
    
    def process_audio(
        self, 
        file_path: str, 
        mode: Literal["transcribe", "translate"] = "transcribe",
        model: str = "whisper-large-v3",
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        preprocess: bool = False,
        cleanup: bool = True
    ) -> TranscriptionResult:
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
                chunk_files = self.chunk_audio(processed_file, temp_dir)
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
                            "response_format": "verbose_json",
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
                            "response_format": "verbose_json",
                        }
                        
                        if prompt:
                            kwargs["prompt"] = prompt
                            
                        result = self.client.audio.translations.create(**kwargs)
                        
                    transcriptions.append(result)
                    
            # Merge results
            final_result = self.merge_transcriptions(transcriptions)
            
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
    api_key: Optional[str] = None
) -> TranscriptionResult:
    """
    Transcribe audio file to text with optional preprocessing and chunking.
    
    Args:
        file_path: Path to the audio file
        model: Groq model to use
        language: Language code (e.g., "en", "fr")
        prompt: Optional prompt to guide transcription
        preprocess: Whether to preprocess the audio file
        api_key: Optional Groq API key
        
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
        preprocess=preprocess
    )

def translate_audio(
    file_path: str,
    model: str = "whisper-large-v3",
    prompt: Optional[str] = None,
    preprocess: bool = False,
    api_key: Optional[str] = None
) -> TranscriptionResult:
    """
    Translate audio file to English with optional preprocessing and chunking.
    
    Args:
        file_path: Path to the audio file
        model: Groq model to use
        prompt: Optional prompt to guide translation
        preprocess: Whether to preprocess the audio file
        api_key: Optional Groq API key
        
    Returns:
        TranscriptionResult with translated text and segments
    """
    processor = AudioProcessor(groq_api_key=api_key)
    return processor.process_audio(
        file_path=file_path,
        mode="translate",
        model=model,
        prompt=prompt,
        preprocess=preprocess
    )

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Transcribe or translate audio files using Groq API")
    parser.add_argument("file", help="Path to audio file")
    parser.add_argument("--mode", choices=["transcribe", "translate"], default="transcribe", 
                        help="Mode: transcribe or translate")
    parser.add_argument("--model", default="whisper-large-v3", 
                        help="Model to use (whisper-large-v3, whisper-large-v3-turbo, distil-whisper-large-v3-en)")
    parser.add_argument("--language", help="Language code for transcription (e.g., en, fr)")
    parser.add_argument("--prompt", help="Prompt to guide transcription/translation")
    parser.add_argument("--preprocess", action="store_true", help="Preprocess audio before processing")
    parser.add_argument("--output", help="Output file path for transcription/translation text")
    
    args = parser.parse_args()
    
    try:
        processor = AudioProcessor(
            chunking_strategy=TimedChunkStrategy(chunk_duration_ms=60000)
        )
        
        result = processor.process_audio(
            file_path=args.file,
            mode=args.mode,
            model=args.model,
            language=args.language,
            prompt=args.prompt,
            preprocess=args.preprocess
        )
        
        print("\nTranscription/Translation:")
        print(result.text)
        print("\nSegments:")
        for segment in result.segments:
            print(f"{segment.id:2} {segment.start:5.1f}-{segment.end:5.1f} -> {segment.text}")
            
        # Print segments with timing
        print("\nSegments:")
        for segment in result.segments:
            print(f"{segment.id:2} {segment.start:5.1f}-{segment.end:5.1f} -> {segment.text}")
            
        # Save to output file if specified
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(result.text)
            print(f"\nSaved to {args.output}")
            
    except Exception as e:
        print(f"Error: {e}")

