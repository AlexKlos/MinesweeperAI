import logging
import os
import struct
import threading
import time

from logging.handlers import RotatingFileHandler
from multiprocessing import Event, Process
from multiprocessing.shared_memory import SharedMemory

import cv2
import keyboard
import mss
import numpy as np

from minesweeper_ai.core_types import Rectangle
from minesweeper_ai.main_pipeline import pipeline_worker


def grab_screen(monitor_idx=1) -> tuple[np.ndarray, dict]:
    with mss.mss() as sct:
        monitor = sct.monitors[monitor_idx]
        img = np.array(sct.grab(monitor))[:, :, :3]
    return img, monitor


def get_playground_coords(template_path: str) -> Rectangle:
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"Template file not found: {template_path}")
    h, w, _ = template.shape

    with mss.mss() as sct:
        for idx, monitor in enumerate(sct.monitors[1:], 1):
            screenshot = np.array(sct.grab(monitor))[:, :, :3]
            res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            threshold = 0.9
            if max_val >= threshold:
                top_left = max_loc
                abs_x = monitor["left"] + top_left[0]
                abs_y = monitor["top"] + top_left[1]
                logging.info(f"Playground found on monitor {idx} with confidence={max_val:.2f}")
                return Rectangle(x=abs_x, y=abs_y, w=w, h=h)
            else:
                logging.info(f"Monitor {idx}: not found, confidence={max_val:.2f}")

    raise ValueError("Playground not found on any monitor.")


def check_click_success(processed_sample: np.ndarray, timestamp_microseconds: int) -> bool:
    """Evaluate result of last click. Stub: always True."""
    return True


def spawn_main_pipeline(
    playground: Rectangle,
    start_event: Event,
    flag_name: str,
    name: str = "main_pipeline",
    daemon: bool = True,
) -> Process:
    main_pipeline_process = Process(
        target=pipeline_worker,
        name=name,
        args=((playground.x, playground.y, playground.w, playground.h), start_event, flag_name),
        daemon=daemon,
    )
    main_pipeline_process.start()
    return main_pipeline_process


def hotkey_listener(paused_flag: threading.Event, shutdown_flag: threading.Event) -> None:
    logging.info("Hotkey thread started. Press 'P' to pause/unpause the manager loop.")
    while not shutdown_flag.is_set():
        keyboard.wait('p')
        if shutdown_flag.is_set():
            break
        if not paused_flag.is_set():
            paused_flag.set()
            logging.info("PAUSE activated. Press 'P' again to continue.")
        else:
            paused_flag.clear()
            logging.info("PAUSE deactivated. Manager will resume.")


def shutdown_listener(shutdown_flag: threading.Event) -> None:
    logging.info("Shutdown thread started. Press 'Q' to exit gracefully.")
    keyboard.wait('q')
    shutdown_flag.set()
    logging.info("Shutdown requested via hotkey 'Q'.")


def main() -> None:
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs"))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "manager.log")
    handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(processName)s] %(levelname)s: %(message)s",
        handlers=[handler],
    )

    playground_template = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../assets/playground.png"))
    playground_rectangle = get_playground_coords(playground_template)
    logging.info("Playground detected at %s", playground_rectangle)

    shared_memory_name = "playground_shared_memory"
    start_event = Event()
    main_pipeline_process = spawn_main_pipeline(
        playground_rectangle, start_event, shared_memory_name, name="main_pipeline", daemon=True
    )

    shared_buffer_size = 1 + 8 + 480
    for _ in range(100):
        try:
            shared_memory = SharedMemory(name=shared_memory_name, create=False, size=shared_buffer_size)
            shared_memory_buffer = shared_memory.buf
            break
        except FileNotFoundError:
            time.sleep(0.05)
    else:
        raise RuntimeError("SharedMemory buffer not found! Pipeline process may not be running.")

    paused_flag = threading.Event()
    shutdown_flag = threading.Event()
    paused_flag.set()
    hotkey_thread = threading.Thread(target=hotkey_listener, args=(paused_flag, shutdown_flag), daemon=True)
    shutdown_thread = threading.Thread(target=shutdown_listener, args=(shutdown_flag,), daemon=True)
    hotkey_thread.start()
    shutdown_thread.start()

    try:
        step = 1
        while not shutdown_flag.is_set():
            if paused_flag.is_set():
                logging.info("Manager is PAUSED. Waiting for unpause...")
                while paused_flag.is_set() and not shutdown_flag.is_set():
                    time.sleep(0.1)
                if shutdown_flag.is_set():
                    break
                logging.info("Resuming main loop.")

            logging.info("Step %d: sending start_event to main_pipeline", step)
            start_event.set()

            wait_start = time.monotonic()
            while shared_memory_buffer[0] != 1 and not shutdown_flag.is_set():
                if time.monotonic() - wait_start > 5.0:
                    logging.error("Timeout waiting for pipeline sample.")
                    return
                time.sleep(0.01)
            if shutdown_flag.is_set():
                break

            timestamp_microseconds = struct.unpack("<Q", shared_memory_buffer[1:9])[0]
            processed_flat_bytes = bytes(shared_memory_buffer[9:489])
            processed_sample = np.frombuffer(processed_flat_bytes, dtype=np.uint8).reshape((30, 16, 1))

            logging.info(
                "Read playground sample, timestamp=%d, min=%d, max=%d",
                timestamp_microseconds, processed_sample.min(), processed_sample.max()
            )

            shared_memory_buffer[0] = 0

            if not check_click_success(processed_sample, timestamp_microseconds):
                logging.info("Click analysis failed. Stopping loop.")
                break

            step += 1

    except KeyboardInterrupt:
        logging.info("Manager interrupted by KeyboardInterrupt")
    finally:
        main_pipeline_process.terminate()
        main_pipeline_process.join(timeout=2.0)
        try:
            shared_memory.close()
        except Exception:
            pass
        logging.info("Manager stopped.")


if __name__ == "__main__":
    main()
