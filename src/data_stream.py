from typing import List, Iterator, Optional


class DataStream:
    """
    Simulates a live, event-driven market data feed.

    Instead of loading an entire dataset into memory at once (O(N)),
    this class emits returns one tick at a time (O(1)), enforcing
    the streaming constraint required by the recursive Bayesian engine.

    Usage:
        stream = DataStream([0.01, -0.03, 0.005, -0.02])

        # Option 1: Manual tick-by-tick
        while stream.has_next:
            tick = stream.next_tick()

        # Option 2: Python generator
        stream.reset()
        for tick in stream.stream():
            print(tick)
    """

    def __init__(self, data: List[float]):
        """
        Initialize the DataStream with a list of historical returns.

        :param data: A list of market returns (e.g., daily percentage changes).
                     Example: [0.01, -0.03, 0.005] represents +1%, -3%, +0.5%.
        """
        if not data:
            raise ValueError("DataStream requires a non-empty list of returns.")

        self._data = data
        self._index = 0
        self._total_points = len(data)

    def next_tick(self) -> Optional[float]:
        """
        Fetches the next market return from the stream.

        Returns:
            The next return value, or None if the stream has been fully consumed.
        """
        if self._index < self._total_points:
            tick = self._data[self._index]
            self._index += 1
            return tick
        return None

    def stream(self) -> Iterator[float]:
        """
        Python generator that yields returns one at a time.
        Allows usage in a for-loop: `for tick in stream.stream():`
        """
        while self._index < self._total_points:
            tick = self._data[self._index]
            self._index += 1
            yield tick

    def reset(self) -> None:
        """Resets the internal pointer back to the beginning of the data."""
        self._index = 0

    @property
    def has_next(self) -> bool:
        """Returns True if there are remaining ticks in the stream."""
        return self._index < self._total_points

    @property
    def progress(self) -> str:
        """Returns a human-readable progress string (e.g., '42/500 ticks')."""
        return f"{self._index}/{self._total_points} ticks"

    def __len__(self) -> int:
        """Returns the total number of data points in the stream."""
        return self._total_points

    def __repr__(self) -> str:
        return f"DataStream(total={self._total_points}, consumed={self._index})"
