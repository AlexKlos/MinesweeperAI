"""
Functions screenshoter() and predictor() are currently stubs (to be implemented later).
"""
import logging
import os
import time

from logging.handlers import RotatingFileHandler
from multiprocessing import Event, Process

from minesweeper_ai.ipc import SharedFlag
from minesweeper_ai.types import Rectangle, State

if os.name == "nt":
    import ctypes


def screenshoter(playground: Rectangle) -> None:
    """Stub for screenshotting, to be implemented later."""
    logging.debug("screenshoter() STUB called with %s", playground)


def predictor() -> None:
    """Stub for prediction, to be implemented later."""
    logging.debug("predictor() STUB called")


def clicker(playground: Rectangle) -> tuple[int, int]:
    """Click at a random point inside the playground."""
    x, y = playground.random_point()
    logging.info("clicker(): clicking at (%d, %d) inside %s", x, y, playground)
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
        logging.warning("clicker(): non-Windows platform — simulated click only")
    return x, y


def pipeline_worker(
    playground_tuple: tuple[int, int, int, int],
    start_event: Event,
    flag_name: str,
    process_sleep: float = 0.01,
) -> None:
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
            screenshoter(playground_rectangle)
            predictor()
            clicker(playground_rectangle)
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


def main() -> None:
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs"))
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "main_pipeline.log")
    handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(processName)s] %(levelname)s: %(message)s",
        handlers=[handler],
    )

    from multiprocessing import Event as MPEvent

    flag_name = "main_pipeline_flag"
    pipeline_flag = SharedFlag(flag_name, create=True)
    try:
        playground_rectangle = Rectangle(200, 200, 400, 300)
        start_event = MPEvent()
        pipeline_process = spawn_main_pipeline(
            playground_rectangle, start_event, flag_name, name="main_pipeline", daemon=True
        )
        for step_number in range(3):
            logging.info("Sending start_event for step %d", step_number + 1)
            start_event.set()
            time.sleep(0.2)
        logging.info("Shutting down main_pipeline process.")
        pipeline_process.terminate()
        pipeline_process.join(timeout=2.0)
    finally:
        try:
            pipeline_flag.close()
            pipeline_flag.unlink()
        except Exception:
            pass


if __name__ == "__main__":
    main()
