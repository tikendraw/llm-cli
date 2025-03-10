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
