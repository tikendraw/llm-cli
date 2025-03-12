import logging
import os
import tempfile
from abc import ABC, abstractmethod
from typing import List

from pydub import AudioSegment

logger = logging.getLogger(__name__)

class ChunkingStrategy(ABC):
    """Abstract base class for audio chunking strategies"""
    
    @abstractmethod
    def chunk_audio(self, audio: AudioSegment, temp_dir: str) -> List[str]:
        """Chunk the audio according to the strategy"""
        pass

class TimedChunkStrategy(ChunkingStrategy):
    """Strategy to chunk audio into fixed-duration segments"""
    
    def __init__(self, chunk_duration_ms: int = 50000, overlap_ms: int = 10000):
        self.chunk_duration_ms = chunk_duration_ms
        self.overlap_ms = overlap_ms
    
    def chunk_audio(self, audio: AudioSegment, temp_dir: str) -> List[str]:
        chunk_files = []
        total_duration = len(audio)
        start_ms = 0
        
        while start_ms < total_duration:
            end_ms = min(start_ms + self.chunk_duration_ms, total_duration)
            chunk = audio[start_ms:end_ms]
            
            chunk_path = os.path.join(temp_dir, f"chunk_{len(chunk_files)}.mp3")
            chunk.export(chunk_path, format="mp3")
            chunk_files.append(chunk_path)
            
            start_ms = end_ms - self.overlap_ms
            start_ms = max(start_ms, end_ms - self.overlap_ms)
            
            if end_ms >= total_duration:
                break
                
        return chunk_files

class SizeBasedChunkStrategy(ChunkingStrategy):
    """Strategy to chunk audio to stay under a size limit"""
    
    def __init__(self, max_size_bytes: int = 25 * 1024 * 1024, overlap_ms: int = 10000):
        self.max_size_bytes = max_size_bytes
        self.overlap_ms = overlap_ms
    
    def chunk_audio(self, audio: AudioSegment, temp_dir: str) -> List[str]:
        chunk_files = []
        total_duration = len(audio)
        
        # Start with a large chunk size (10 minutes)
        chunk_size_ms = 10 * 60 * 1000
        
        # Find optimal chunk size
        while True:
            test_chunk = audio[:min(chunk_size_ms, total_duration)]
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_file:
                test_chunk.export(temp_file.name, format="mp3")
                temp_size = os.path.getsize(temp_file.name)
                os.unlink(temp_file.name)
                
            if temp_size < self.max_size_bytes:
                break
                
            chunk_size_ms = int(chunk_size_ms * 0.8)
            
            if chunk_size_ms < 30000:  # 30 seconds minimum
                raise ValueError("Cannot chunk file to stay under size limit")
        
        # Create chunks with the determined size
        start_ms = 0
        while start_ms < total_duration:
            end_ms = min(start_ms + chunk_size_ms, total_duration)
            chunk = audio[start_ms:end_ms]
            
            chunk_path = os.path.join(temp_dir, f"chunk_{len(chunk_files)}.mp3")
            chunk.export(chunk_path, format="mp3")
            chunk_files.append(chunk_path)
            
            start_ms = end_ms - self.overlap_ms
            start_ms = max(start_ms, end_ms - self.overlap_ms)
            
            if end_ms >= total_duration:
                break
                
        return chunk_files

class SilenceBasedChunkStrategy(ChunkingStrategy):
    """Strategy to chunk audio based on silent segments and merge into optimal-sized chunks"""
    
    def __init__(self, 
                 max_size_bytes: int = 25 * 1024 * 1024, 
                 overlap_ms: int = 10000,
                 min_silence_len: int = 1000,
                 silence_thresh: int = -16,
                 keep_silence: int = 100,
                 target_size_ratio: float = 0.95):  # Target 95% of max size
        self.max_size_bytes = max_size_bytes
        self.overlap_ms = overlap_ms
        self.min_silence_len = min_silence_len
        self.silence_thresh = silence_thresh
        self.keep_silence = keep_silence
        self.target_size = max_size_bytes * target_size_ratio
    
    def get_chunk_size(self, audio_chunk: AudioSegment) -> int:
        """Get the size of an audio chunk when exported as MP3"""
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=True) as temp_file:
            audio_chunk.export(temp_file.name, format="mp3")
            return os.path.getsize(temp_file.name)

    def chunk_audio(self, audio: AudioSegment, temp_dir: str) -> List[str]:
        from pydub.silence import split_on_silence
        
        # First split on silence
        initial_chunks = split_on_silence(
            audio,
            min_silence_len=self.min_silence_len,
            silence_thresh=self.silence_thresh,
            keep_silence=self.keep_silence
        )
        
        if not initial_chunks:
            initial_chunks = [audio]
        
        chunk_files = []
        current_merged = initial_chunks[0]
        pending_chunks = []
        
        # Helper function to export merged chunk
        def export_merged_chunk(merged_audio: AudioSegment, include_overlap: bool = True) -> None:
            if len(merged_audio) > 0:
                chunk_path = os.path.join(temp_dir, f"chunk_{len(chunk_files)}.mp3")
                merged_audio.export(chunk_path, format="mp3")
                chunk_files.append(chunk_path)
                
                if include_overlap and len(merged_audio) > self.overlap_ms:
                    # Keep overlap portion for next chunk
                    overlap_start = max(0, len(merged_audio) - self.overlap_ms)
                    return merged_audio[overlap_start:]
            return None

        # Process all chunks
        for chunk in initial_chunks[1:]:
            test_merged = current_merged + chunk
            test_size = self.get_chunk_size(test_merged)
            
            if test_size < self.target_size:
                # Can still add more chunks
                current_merged = test_merged
                pending_chunks.append(chunk)
            else:
                # Would exceed target size, export current and start new
                overlap_audio = export_merged_chunk(current_merged)
                current_merged = overlap_audio + chunk if overlap_audio else chunk
                pending_chunks = [chunk]
        
        # Export final chunk
        if len(current_merged) > 0:
            export_merged_chunk(current_merged, include_overlap=False)
        
        logger.info(f"Reduced chunks from {len(initial_chunks)} to {len(chunk_files)} chunks")
        return chunk_files
