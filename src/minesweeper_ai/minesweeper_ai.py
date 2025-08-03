import os
import time
import logging
import threading
import json
import shutil

from datetime import datetime
from logging.handlers import RotatingFileHandler

import cv2
import mss
import numpy as np
import keyboard
import ctypes

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets")
DATASET_BASE = os.path.join(ASSETS_DIR, "dataset")
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

MOVE_RESULTS_DIR = os.path.join(DATASET_BASE, "move_results")
MOVE_VISUALS_DIR = os.path.join(DATASET_BASE, "move_visuals")
NUMPY_DIR = os.path.join(DATASET_BASE, "numpy_array")
SCREENSHOT_RAW_DIR = os.path.join(DATASET_BASE, "raw_screenshot")
SCREENSHOT_30x16_DIR = os.path.join(DATASET_BASE, "30x16_screenshot")

GAME_WIDTH = 480
GAME_HEIGHT = 256
CELLS_X = 30
CELLS_Y = 16

GAME_PAUSE = 0.01


def setup_logging():
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, "minesweeper_ai.log")
    handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s]: %(message)s",
        handlers=[handler],
    )


def find_playground_coords(template_path):
    template = cv2.imread(template_path, cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"Template not found: {template_path}")
    h, w, _ = template.shape
    with mss.mss() as sct:
        for idx, monitor in enumerate(sct.monitors[1:], 1):
            screenshot = np.array(sct.grab(monitor))[:, :, :3]
            res = cv2.matchTemplate(screenshot, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            threshold = 0.995
            if max_val >= threshold:
                abs_x = monitor["left"] + max_loc[0]
                abs_y = monitor["top"] + max_loc[1]
                logging.info(f"Playground found on monitor {idx} (confidence={max_val:.2f})")
                return abs_x, abs_y, w, h
            else:
                logging.info(f"Monitor {idx}: not found, confidence={max_val:.2f}")
    raise RuntimeError("Playground not found on any monitor")


def get_face_rectangle(playground):
    x, y, w, h = playground
    fx = x + w + 227 - 480
    fy = y - 41
    return fx, fy, 26, 26


def get_face_center(face_rectangle):
    fx, fy, fw, fh = face_rectangle
    return (fx + fw // 2, fy + fh // 2)


def click(x, y):
    if os.name == "nt":
        user32 = ctypes.windll.user32
        user32.SetCursorPos(int(x), int(y))
        time.sleep(GAME_PAUSE)
        MOUSEEVENTF_LEFTDOWN = 0x0002
        MOUSEEVENTF_LEFTUP = 0x0004
        user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
        time.sleep(GAME_PAUSE)
        user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
    else:
        logging.warning("Simulated click (non-Windows): (%d, %d)", x, y)


def capture_face(face_rectangle):
    fx, fy, fw, fh = face_rectangle
    with mss.mss() as sct:
        monitor = {"left": fx, "top": fy, "width": fw, "height": fh}
        return np.array(sct.grab(monitor))[:, :, :3]


def compare_face(face_img, template_img, threshold=0.995):
    if face_img.shape != template_img.shape:
        return False
    sim = np.mean(face_img == template_img)
    return sim >= threshold


def save_numpy(timestamp, array):
    os.makedirs(NUMPY_DIR, exist_ok=True)
    np.save(os.path.join(NUMPY_DIR, f"{timestamp}.npy"), array)


def save_file(data, subfolder, timestamp, filetype="png"):
    save_dir = os.path.join(DATASET_BASE, subfolder)
    os.makedirs(save_dir, exist_ok=True)
    if filetype == "png":
        cv2.imwrite(os.path.join(save_dir, f"{timestamp}.png"), data)
    elif filetype == "npy":
        np.save(os.path.join(save_dir, f"{timestamp}.npy"), data)
    else:
        raise ValueError(f"Unsupported filetype: {filetype}")


def save_move_result(timestamp, best_x, best_y, success, step):
    os.makedirs(MOVE_RESULTS_DIR, exist_ok=True)
    result = {
        "timestamp": timestamp,
        "best_x": int(best_x),
        "best_y": int(best_y),
        "success": bool(success),
        "step": int(step)
    }
    with open(os.path.join(MOVE_RESULTS_DIR, f"{timestamp}.json"), "w") as f:
        json.dump(result, f, indent=2)


def save_move_visualization(timestamp, raw_img, best_x, best_y, success):
    os.makedirs(MOVE_VISUALS_DIR, exist_ok=True)
    img_vis = raw_img.copy()
    cell_w = GAME_WIDTH // CELLS_X
    cell_h = GAME_HEIGHT // CELLS_Y
    if best_x >= 0 and best_y >= 0:
        cx = int(best_x * cell_w + cell_w // 2)
        cy = int(best_y * cell_h + cell_h // 2)
        color = (0, 255, 0) if success else (0, 0, 255)
        cv2.circle(img_vis, (cx, cy), 10, color, 2)
    cv2.imwrite(os.path.join(MOVE_VISUALS_DIR, f"{timestamp}.png"), img_vis)


def capture_playground(playground):
    x, y, w, h = playground
    if os.name == "nt":
        ctypes.windll.user32.SetCursorPos(int(x + w + 1), int(y))
        time.sleep(GAME_PAUSE)
    with mss.mss() as sct:
        monitor = {"left": x, "top": y, "width": GAME_WIDTH, "height": GAME_HEIGHT}
        img = np.array(sct.grab(monitor))[:, :, :3]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    processed_array = np.zeros((CELLS_Y, CELLS_X), dtype=np.uint8)
    for i in range(CELLS_Y):
        for j in range(CELLS_X):
            block = img[i*16:(i+1)*16, j*16:(j+1)*16, :]
            processed_array[i, j] = int(np.round(block.mean()))
    processed_expanded = np.expand_dims(processed_array.T, axis=-1)
    return img, processed_expanded, timestamp


def predict_move(processed_image):
    if processed_image.ndim == 3:
        processed_image = processed_image.squeeze()
    assert processed_image.shape == (30, 16)
    heatmap = np.random.rand(30, 16).astype(np.float32)
    heatmap /= heatmap.sum()
    best_idx = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    return best_idx


def click_cell(playground, best_x, best_y):
    x, y, w, h = playground
    cell_w = w / 30.0
    cell_h = h / 16.0
    abs_x = int(x + best_x * cell_w + cell_w // 2)
    abs_y = int(y + best_y * cell_h + cell_h // 2)
    logging.info("Click cell: best_x=%d, best_y=%d -> abs=(%d, %d)", best_x, best_y, abs_x, abs_y)
    click(abs_x, abs_y)
    return abs_x, abs_y


def check_click_success(current_sample, previous_sample, happy_face, dead_face, face_rectangle):
    if not np.any(current_sample - previous_sample):
        return False
    face_img = capture_face(face_rectangle)
    if compare_face(face_img, happy_face):
        return True
    elif compare_face(face_img, dead_face):
        return False
    else:
        return False


def wait_for_hotkey(event, shutdown_event):
    while not shutdown_event.is_set():
        keyboard.wait('p')
        if event.is_set():
            event.clear()
            logging.info("PAUSE deactivated.")
        else:
            event.set()
            logging.info("PAUSE activated.")
        time.sleep(0.2)


def wait_for_shutdown(shutdown_event):
    keyboard.wait('q')
    shutdown_event.set()
    logging.info("Shutdown requested via hotkey 'Q'.")


def copy_first_step_assets(timestamp):
    raw_dst = os.path.join(SCREENSHOT_RAW_DIR, f"{timestamp}.png")
    scaled_dst = os.path.join(SCREENSHOT_30x16_DIR, f"{timestamp}.png")
    npy_dst = os.path.join(NUMPY_DIR, f"{timestamp}.npy")
    os.makedirs(os.path.dirname(raw_dst), exist_ok=True)
    os.makedirs(os.path.dirname(scaled_dst), exist_ok=True)
    os.makedirs(os.path.dirname(npy_dst), exist_ok=True)

    shutil.copyfile(
        os.path.join(ASSETS_DIR, "playground.png"),
        raw_dst,
    )
    shutil.copyfile(
        os.path.join(ASSETS_DIR, "playground_30_16.png"),
        scaled_dst,
    )
    shutil.copyfile(
        os.path.join(ASSETS_DIR, "start_playground.npy"),
        npy_dst,
    )


def main():
    setup_logging()
    playground_template = os.path.join(ASSETS_DIR, "playground.png")
    happy_face_path   = os.path.join(ASSETS_DIR, "happy_face.png")
    dead_face_path    = os.path.join(ASSETS_DIR, "dead_face.png")
    playground = find_playground_coords(playground_template)
    logging.info("Playground detected at %s", playground)
    face_rectangle = get_face_rectangle(playground)
    face_center = get_face_center(face_rectangle)
    happy_face = cv2.imread(happy_face_path, cv2.IMREAD_COLOR)
    dead_face  = cv2.imread(dead_face_path,  cv2.IMREAD_COLOR)

    paused_flag = threading.Event()
    shutdown_flag = threading.Event()
    paused_flag.set()
    threading.Thread(target=wait_for_hotkey, args=(paused_flag, shutdown_flag), daemon=True).start()
    threading.Thread(target=wait_for_shutdown, args=(shutdown_flag,), daemon=True).start()

    logging.info("Paused. Press 'P' to start.")
    while paused_flag.is_set() and not shutdown_flag.is_set():
        time.sleep(0.1)
    if shutdown_flag.is_set():
        return

    logging.info("First move: clicking restart face.")
    click(*face_center)
    step = 1

    current_raw_img = cv2.imread(os.path.join(ASSETS_DIR, "playground.png"))
    current_sample = np.load(os.path.join(ASSETS_DIR, "start_playground.npy"))
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    try:
        while not shutdown_flag.is_set():
            logging.info(f"Step {step}: start")
            best_x, best_y = predict_move(current_sample)
            click_cell(playground, best_x, best_y)

            next_raw_img, next_sample, next_timestamp = capture_playground(playground)
            click_success = check_click_success(next_sample, current_sample, happy_face, dead_face, face_rectangle)

            save_file(current_raw_img, "raw_screenshot", current_timestamp, "png")
            save_file(np.squeeze(current_sample.T), "30x16_screenshot", current_timestamp, "png")
            save_file(current_sample, "numpy_array", current_timestamp, "npy")
            save_move_result(current_timestamp, best_x, best_y, click_success, step)
            save_move_visualization(current_timestamp, current_raw_img, best_x, best_y, click_success)

            step += 1

            if paused_flag.is_set():
                logging.info("Paused. Waiting for unpause...")
                while paused_flag.is_set() and not shutdown_flag.is_set():
                    time.sleep(0.1)
                if shutdown_flag.is_set():
                    break
                logging.info("Unpaused. Restarting game.")
                click(*face_center)
                step = 1
                current_raw_img = cv2.imread(os.path.join(ASSETS_DIR, "playground.png"))
                current_sample = np.load(os.path.join(ASSETS_DIR, "start_playground.npy"))
                current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                continue

            if not click_success:
                logging.info("Click analysis failed. Restarting game.")
                click(*face_center)
                step = 1
                current_raw_img = cv2.imread(os.path.join(ASSETS_DIR, "playground.png"))
                current_sample = np.load(os.path.join(ASSETS_DIR, "start_playground.npy"))
                current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                continue

            current_raw_img = next_raw_img
            current_sample = next_sample
            current_timestamp = next_timestamp

    except KeyboardInterrupt:
        logging.info("Interrupted by KeyboardInterrupt.")
    finally:
        logging.info("Stopped.")


if __name__ == "__main__":
    main()
