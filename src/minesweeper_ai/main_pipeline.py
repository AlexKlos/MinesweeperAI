import logging
import os
import struct
import time

from datetime import datetime
from multiprocessing import Event
from multiprocessing.shared_memory import SharedMemory

import cv2
import mss
import numpy as np

from minesweeper_ai.core_types import Rectangle

if os.name == "nt":
    import ctypes


def capture_playground_sample(playground: Rectangle) -> np.ndarray:
    if os.name == "nt":
        cursor_x = playground.x + playground.w + 1
        cursor_y = playground.y
        ctypes.windll.user32.SetCursorPos(int(cursor_x), int(cursor_y))
        time.sleep(0.01)

    with mss.mss() as screenshot:
        monitor = {
            "left": playground.x,
            "top": playground.y,
            "width": 480,
            "height": 256,
        }
        img = np.array(screenshot.grab(monitor))[:, :, :3]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    raw_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../assets/dataset/raw_screenshot"))
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, f"{timestamp}.png")
    cv2.imwrite(raw_path, img)

    processed_array = np.zeros((16, 30), dtype=np.uint8)
    for i in range(16):
        for j in range(30):
            block = img[i*16:(i+1)*16, j*16:(j+1)*16, :]
            avg_val = int(np.round(block.mean()))
            processed_array[i, j] = avg_val

    processed_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../assets/dataset/30x16_screenshot"))
    os.makedirs(processed_dir, exist_ok=True)
    processed_path = os.path.join(processed_dir, f"{timestamp}.png")
    cv2.imwrite(processed_path, processed_array)

    npy_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../assets/dataset/numpy_array"))
    os.makedirs(npy_dir, exist_ok=True)
    npy_path = os.path.join(npy_dir, f"{timestamp}.npy")
    processed_expanded_array = np.expand_dims(processed_array.T, axis=-1)
    np.save(npy_path, processed_expanded_array)

    return processed_expanded_array


def predict_move() -> None:
    """Stub for prediction, to be implemented later."""
    logging.debug("predict_move() STUB called")


def click_cell(playground: Rectangle) -> tuple[int, int]:
    """Click at a random point inside the playground."""
    x, y = playground.random_point()
    logging.info("click_cell(): clicking at (%d, %d) inside %s", x, y, playground)
    if os.name == "nt":
        user32 = ctypes.windll.user32
        user32.SetCursorPos(int(x), int(y))
        time.sleep(0.02)
        MOUSEEVENTF_LEFTDOWN = 0x0002
        MOUSEEVENTF_LEFTUP = 0x0004
        user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(0.02)
        user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    else:
        logging.warning("click_cell(): non-Windows platform — simulated click only")
    return x, y


def pipeline_worker(
    playground_tuple: tuple[int, int, int, int],
    start_event: Event,
    shared_memory_name: str,
    process_sleep: float = 0.01,
) -> None:
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs"))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "main_pipeline.log")
    from logging.handlers import RotatingFileHandler
    handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(processName)s] %(levelname)s: %(message)s",
        handlers=[handler],
    )

    shared_buffer_size = 1 + 8 + 480
    shared_memory = SharedMemory(name=shared_memory_name, create=True, size=shared_buffer_size)
    shared_memory_buffer = shared_memory.buf
    shared_memory_buffer[0] = 0

    playground_rectangle = Rectangle(*playground_tuple)
    logging.info(
        "Main Pipeline started. Playground=%s, shared_memory_name=%s", playground_rectangle, shared_memory_name
    )

    try:
        while True:
            start_event.wait()
            start_event.clear()
            shared_memory_buffer[0] = 0

            processed_image = capture_playground_sample(playground_rectangle)
            logging.info("Screenshot captured and processed.")

            predict_move()
            logging.info("Predict_move() called.")

            click_x, click_y = click_cell(playground_rectangle)
            logging.info("Clicked at (%d, %d) in playground", click_x, click_y)

            processed_flat = processed_image.squeeze().T.flatten()
            timestamp_microseconds = int(datetime.now().timestamp() * 1_000_000)
            shared_memory_buffer[1:9] = struct.pack("<Q", timestamp_microseconds)
            shared_memory_buffer[9:489] = processed_flat.tobytes()
            shared_memory_buffer[0] = 1
            logging.info("Sample written to shared memory, timestamp=%d", timestamp_microseconds)

            time.sleep(process_sleep)
    except Exception as error:
        logging.exception("Main Pipeline crashed: %s", error)
    finally:
        shared_memory.close()
        shared_memory.unlink()
        logging.info("Main Pipeline stopped.")
