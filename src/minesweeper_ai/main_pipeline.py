import logging
import os
import time

from datetime import datetime
from multiprocessing import Event

import cv2
import mss
import numpy as np

from minesweeper_ai.ipc import SharedFlag
from minesweeper_ai.core_types import Rectangle, State

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
    flag_name: str,
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

    playground_rectangle = Rectangle(*playground_tuple)
    main_pipeline_flag = SharedFlag(flag_name, create=True)
    logging.info(
        "Main Pipeline started. Playground=%s, flag_name=%s", playground_rectangle, flag_name
    )
    try:
        main_pipeline_flag.set(State.IDLE.value)
        while True:
            start_event.wait()
            main_pipeline_flag.set(State.BUSY.value)
            capture_playground_sample(playground_rectangle)
            predict_move()
            click_cell(playground_rectangle)
            main_pipeline_flag.set(State.IDLE.value)
            start_event.clear()
            time.sleep(process_sleep)
    except KeyboardInterrupt:
        logging.info("Main Pipeline interrupted by KeyboardInterrupt")
    except Exception as e:
        main_pipeline_flag.set(State.ERROR.value)
        logging.exception("Main Pipeline crashed with exception: %s", e)
    finally:
        main_pipeline_flag.close()
        logging.info("Main Pipeline stopped.")

