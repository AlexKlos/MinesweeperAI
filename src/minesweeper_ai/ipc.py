from multiprocessing import shared_memory

from minesweeper_ai.types import State


class SharedFlag:
    def __init__(self, name: str, size: int = 1, create: bool = False) -> None:
        self.is_owner = create
        if create:
            self._shm = shared_memory.SharedMemory(name=name, create=True, size=size)
            self._buf = self._shm.buf
            self.set(State.IDLE.value)
        else:
            self._shm = shared_memory.SharedMemory(name=name, create=False)
            self._buf = self._shm.buf

    @property
    def name(self) -> str:
        return self._shm.name

    def get(self, idx: int = 0) -> int:
        return self._buf[idx]

    def set(self, value: int, idx: int = 0) -> None:
        if not self.is_owner:
            raise PermissionError("Only owner can modify this flag!")
        self._buf[idx] = value & 0xFF

    def close(self) -> None:
        self._buf.release()
        self._shm.close()

    def unlink(self) -> None:
        if not self.is_owner:
            raise PermissionError("Only owner can unlink the segment!")
        try:
            self._shm.unlink()
        except FileNotFoundError:
            pass
