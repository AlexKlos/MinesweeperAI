import logging
import os
import threading
import time

from logging.handlers import RotatingFileHandler
from multiprocessing import Event, Process

import cv2
import keyboard
import mss
import numpy as np

from minesweeper_ai.core_types import Rectangle, State
from minesweeper_ai.ipc import SharedFlag
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


def check_click_success() -> bool:
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


def wait_for_flag(
    flag: SharedFlag,
    from_state: State,
    to_state: State,
    timeout: float = 5.0,
    poll_interval: float = 0.01,
) -> bool:
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout:
        if flag.get() != from_state.value:
            break
        time.sleep(poll_interval)
    else:
        return False

    while time.monotonic() - start_time < timeout:
        if flag.get() == to_state.value:
            return True
        time.sleep(poll_interval)
    return False


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

    flag_name = "main_pipeline_flag"
    start_event = Event()
    main_pipeline_process = spawn_main_pipeline(
        playground_rectangle, start_event, flag_name, name="main_pipeline", daemon=True
    )

    for _ in range(100):
        try:
            main_pipeline_flag = SharedFlag(flag_name, create=False)
            break
        except FileNotFoundError:
            time.sleep(0.05)
    else:
        raise RuntimeError("main_pipeline_flag not found, main_pipeline may not be running!")

    paused_flag = threading.Event()
    paused_flag.set()
    shutdown_flag = threading.Event()
    hotkey_thread = threading.Thread(target=hotkey_listener, args=(paused_flag, shutdown_flag), daemon=True)
    shutdown_thread = threading.Thread(target=shutdown_listener, args=(shutdown_flag,), daemon=True)
    hotkey_thread.start()
    shutdown_thread.start()

    try:
        logging.info("Waiting for main_pipeline to become IDLE...")
        while main_pipeline_flag.get() != State.IDLE.value:
            time.sleep(0.05)

        logging.info("main_pipeline is IDLE. Starting play loop...")
        step = 1
        while not shutdown_flag.is_set():
            if paused_flag.is_set():
                logging.info("Manager is PAUSED. Waiting for unpause...")
                while paused_flag.is_set() and not shutdown_flag.is_set():
                    time.sleep(0.1)
                if shutdown_flag.is_set():
                    break
                logging.info("Resuming main loop.")

            logging.info("Step %d: Sending start_event to main_pipeline", step)
            start_event.set()
            ok = wait_for_flag(
                main_pipeline_flag,
                from_state=State.IDLE,
                to_state=State.IDLE,
                timeout=5.0
            )
            if not ok:
                logging.error("Timeout waiting for main_pipeline to complete step!")
                break
            if not check_click_success():
                logging.info("Click failed (game probably lost). Exiting loop.")
                break
            step += 1

        if shutdown_flag.is_set():
            logging.info("Graceful shutdown requested. Exiting main loop.")

    except KeyboardInterrupt:
        logging.info("Manager interrupted by KeyboardInterrupt")
    finally:
        main_pipeline_process.terminate()
        main_pipeline_process.join(timeout=2.0)
        try:
            main_pipeline_flag.close()
        except Exception:
            pass
        logging.info("Manager stopped.")


if __name__ == "__main__":
    main()
