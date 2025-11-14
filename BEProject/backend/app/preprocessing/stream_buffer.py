"""
Stream buffer for real-time audio processing
Handles circular buffering with overlap for smooth audio streaming
"""

import numpy as np
from collections import deque
from typing import Optional, Union
import threading


class StreamBuffer:
    """
    Circular buffer for real-time audio streaming with overlap support.

    This buffer helps manage incoming audio chunks and provides:
    - Circular buffering (old data automatically removed)
    - Overlap between chunks for smooth processing
    - Efficient memory management
    - Thread-safety for producer/consumer usage
    """

    def __init__(self, max_size: int, overlap: int = 0):
        """
        Initialize stream buffer

        Args:
            max_size (int): Maximum size of buffer in samples
            overlap (int): Number of samples to overlap between chunks (default: 0)
        """
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        if overlap < 0:
            raise ValueError(f"overlap must be non-negative, got {overlap}")
        if overlap >= max_size:
            raise ValueError(f"overlap ({overlap}) must be less than max_size ({max_size})")

        self.max_size = max_size
        self.overlap = overlap

        # Main circular buffer (stores python scalars)
        self.buffer = deque(maxlen=max_size)

        # Overlap buffer to store data from previous chunk
        self.overlap_buffer = deque(maxlen=overlap)

        # Statistics
        self.total_added = 0
        self.total_retrieved = 0

        # Thread-safety
        self._lock = threading.Lock()

    def add(self, data: Union[np.ndarray, list]):
        """
        Add data to buffer

        Args:
            data: Audio samples as numpy array or list
        """
        if data is None:
            return

        # Normalize to 1D numpy array of float32
        if isinstance(data, np.ndarray):
            arr = data.flatten()
        else:
            # assume list-like
            arr = np.asarray(data).flatten()

        if arr.size == 0:
            return

        # Convert to python scalars for deque (keeps memory small)
        # use tolist() because deque.extend on numpy iterates numpy scalars which are fine,
        # but tolist() is explicit and portable.
        to_extend = arr.tolist()

        with self._lock:
            self.buffer.extend(to_extend)
            self.total_added += len(to_extend)

    def get_chunk(self, chunk_size: int) -> Optional[np.ndarray]:
        """
        Get chunk from buffer with overlap from previous chunk

        Args:
            chunk_size (int): Size of chunk to retrieve

        Returns:
            numpy array of audio samples (dtype=np.float32), or None if not enough data
        """
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        if not self.is_ready(chunk_size):
            return None

        new_data_size = chunk_size - self.overlap
        if new_data_size <= 0:
            raise ValueError(f"chunk_size ({chunk_size}) must be greater than overlap ({self.overlap})")

        with self._lock:
            # Pop new_data_size samples from buffer (they are python scalars)
            new_data = [self.buffer.popleft() for _ in range(new_data_size)]

            # Combine overlap from previous chunk with new data
            if len(self.overlap_buffer) > 0:
                chunk_list = list(self.overlap_buffer) + new_data
            else:
                chunk_list = new_data

            # If chunk shorter than requested (shouldn't happen because is_ready checked),
            # pad with zeros
            if len(chunk_list) < chunk_size:
                pad_len = chunk_size - len(chunk_list)
                chunk_list.extend([0.0] * pad_len)

            # Store overlap for next chunk (last 'overlap' samples)
            self.overlap_buffer.clear()
            if self.overlap > 0:
                self.overlap_buffer.extend(chunk_list[-self.overlap:])

            self.total_retrieved += new_data_size

        chunk = np.asarray(chunk_list, dtype=np.float32)
        return chunk

    def is_ready(self, chunk_size: int) -> bool:
        """
        Check if buffer has enough data for a chunk

        Args:
            chunk_size (int): Required chunk size

        Returns:
            bool: True if enough data available
        """
        if chunk_size <= 0:
            return False
        required_new_data = chunk_size - self.overlap
        with self._lock:
            return len(self.buffer) >= required_new_data

    def get_available(self) -> np.ndarray:
        """
        Get all available data without overlap handling

        Returns:
            numpy array of all buffer contents (dtype=np.float32)
        """
        with self._lock:
            data = np.array(list(self.buffer), dtype=np.float32)
            self.total_retrieved += len(data)
            self.buffer.clear()
        return data

    def peek(self, size: int) -> Optional[np.ndarray]:
        """
        Peek at data without removing from buffer

        Args:
            size (int): Number of samples to peek

        Returns:
            numpy array of peeked data (dtype=np.float32), or None if not enough data
        """
        if size <= 0:
            return None

        with self._lock:
            if len(self.buffer) < size:
                return None
            # Build list without removing
            return np.array([self.buffer[i] for i in range(size)], dtype=np.float32)

    def clear(self):
        """Clear all buffers"""
        with self._lock:
            self.buffer.clear()
            self.overlap_buffer.clear()

    def reset_stats(self):
        """Reset statistics counters"""
        with self._lock:
            self.total_added = 0
            self.total_retrieved = 0

    def __len__(self) -> int:
        """Return current buffer size"""
        with self._lock:
            return len(self.buffer)

    def __repr__(self) -> str:
        """String representation"""
        with self._lock:
            return (f"StreamBuffer(size={len(self.buffer)}/{self.max_size}, "
                    f"overlap={self.overlap}, added={self.total_added}, "
                    f"retrieved={self.total_retrieved})")

    @property
    def is_empty(self) -> bool:
        """Check if buffer is empty"""
        return len(self) == 0

    @property
    def is_full(self) -> bool:
        """Check if buffer is full"""
        return len(self) >= self.max_size

    @property
    def available_space(self) -> int:
        """Get available space in buffer"""
        return max(0, self.max_size - len(self))

    @property
    def fill_percentage(self) -> float:
        """Get buffer fill percentage"""
        return (len(self) / self.max_size) * 100.0


# Example usage and testing
if __name__ == "__main__":
    """Test the StreamBuffer class"""

    print("Testing StreamBuffer...")
    print("=" * 60)

    # Test 1: Basic functionality
    print("\n[Test 1] Basic Add and Get")
    buffer = StreamBuffer(max_size=1000, overlap=100)

    # Add some data
    data1 = np.random.randn(500).astype(np.float32)
    buffer.add(data1)
    print(f"Added 500 samples. Buffer: {buffer}")

    # Get a chunk
    if buffer.is_ready(300):
        chunk = buffer.get_chunk(300)
        print(f"Retrieved chunk: {chunk.shape}")
        print(f"Buffer after retrieval: {buffer}")
    else:
        print("Not ready for chunk (expected if not enough data)")

    # Test 2: Overlap functionality
    print("\n[Test 2] Overlap Between Chunks")
    buffer2 = StreamBuffer(max_size=2000, overlap=256)

    # Simulate streaming
    for i in range(5):
        # Add data
        new_data = np.random.randn(512).astype(np.float32)
        buffer2.add(new_data)

        # Get chunk if ready
        if buffer2.is_ready(512):
            chunk = buffer2.get_chunk(512)
            print(f"Chunk {i+1}: shape={chunk.shape}, buffer_size={len(buffer2)}")

    # Test 3: Buffer overflow
    print("\n[Test 3] Buffer Overflow Handling")
    buffer3 = StreamBuffer(max_size=100, overlap=10)

    # Add more than max_size
    large_data = np.random.randn(200).astype(np.float32)
    buffer3.add(large_data)
    print(f"Added 200 samples to buffer with max_size=100")
    print(f"Buffer size: {len(buffer3)} (old data automatically removed)")

    # Test 4: Statistics
    print("\n[Test 4] Statistics")
    print(f"Total added: {buffer3.total_added}")
    print(f"Total retrieved: {buffer3.total_retrieved}")
    print(f"Fill percentage: {buffer3.fill_percentage:.1f}%")

    # Test 5: Peek
    print("\n[Test 5] Peek Without Removing")
    peeked = buffer3.peek(20)
    if peeked is not None:
        print(f"Peeked 20 samples: {peeked.shape}")
        print(f"Buffer size unchanged: {len(buffer3)}")
    else:
        print("Not enough data to peek")

    print("\n" + "=" * 60)
    print("All tests completed! ✓")
