import os
import re
import time

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel

PARAM_KEYS = [
    "Status",
    "Click Counter",
    "Average Step Time",
    "Average Steps",
    "Max Steps"
]
PARAM_COLOR = "#9C85CC"
LOGS_COLOR = "#62A6A8"


class LogMonitor:
    def __init__(self):
        self.console = Console()
        self.log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs"))
        self.log_file = os.path.join(self.log_dir, "minesweeper_ai.log")
        self.params = {k: "" for k in PARAM_KEYS}
        self.session_step_times = []
        self.session_clicks = 0
        self.game_steps = []
        self.current_game_steps = 0
        self.max_steps = 0
        self.last_status = "Pause"
        self.paused = True

    def get_max_log_lines(self):
        total_height = self.console.size.height
        params_panel_height = len(PARAM_KEYS) + 2
        reserved = params_panel_height + 3
        log_lines = max(total_height - reserved, 3)
        return log_lines

    def parse_runtime_params(self):
        if not os.path.exists(self.log_file):
            self.params = {k: "?" for k in PARAM_KEYS}
            return
    
        step_pattern = re.compile(r"Step (\d+): start", re.IGNORECASE)
        pause_pattern = re.compile(r"PAUSE activated|Paused\. Waiting for unpause", re.IGNORECASE)
        unpause_pattern = re.compile(r"PAUSE deactivated|Unpaused\. Restarting game", re.IGNORECASE)
        restart_pattern = re.compile(r"Click analysis failed\. Restarting game", re.IGNORECASE)
    
        with open(self.log_file, encoding="utf-8") as f:
            lines = f.readlines()
    
        self.session_step_times.clear()
        self.session_clicks = 0
        self.game_steps.clear()
        self.current_game_steps = 0
        self.max_steps = 0
        self.paused = False
        self.last_status = "Active"
        self.last_step_num = 0
    
        last_step_time = None
    
        for idx, line in enumerate(lines):
            if pause_pattern.search(line):
                self.paused = True
                self.last_status = "Pause"
                if self.current_game_steps > 0:
                    self.game_steps.append(self.current_game_steps)
                    if self.current_game_steps > self.max_steps:
                        self.max_steps = self.current_game_steps
                    self.current_game_steps = 0
    
            elif unpause_pattern.search(line) or restart_pattern.search(line):
                self.paused = False
                self.last_status = "Active"
                if self.current_game_steps > 0:
                    self.game_steps.append(self.current_game_steps)
                    if self.current_game_steps > self.max_steps:
                        self.max_steps = self.current_game_steps
                    self.current_game_steps = 0
    
            elif step_pattern.search(line):
                self.session_clicks += 1
                match = step_pattern.search(line)
                step_num = match.group(1)
                if step_num:
                    self.last_step_num = int(step_num)
                    self.current_game_steps += 1
    
                ts = self.extract_timestamp(line)
                if last_step_time is not None:
                    self.session_step_times.append(ts - last_step_time)
                last_step_time = ts
    
        if self.current_game_steps > 0:
            self.game_steps.append(self.current_game_steps)
            if self.current_game_steps > self.max_steps:
                self.max_steps = self.current_game_steps
    
        avg_step_time = "-"
        if self.session_step_times:
            avg_sec = sum(self.session_step_times) / len(self.session_step_times)
            avg_step_time = f"{avg_sec:.2f} sec"
    
        avg_steps = "-"
        if self.game_steps:
            avg_steps = f"{sum(self.game_steps) / len(self.game_steps):.2f}"
    
        max_steps = str(self.max_steps) if self.max_steps > 0 else "-"
    
        self.params["Status"] = self.last_status
        self.params["Click Counter"] = str(self.session_clicks)
        self.params["Average Step Time"] = avg_step_time
        self.params["Average Steps"] = avg_steps
        self.params["Max Steps"] = max_steps

    @staticmethod
    def extract_timestamp(line: str) -> float:
        m = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d+)", line)
        if not m:
            return 0.0
        time_str, ms_str = m.groups()
        ts = time.mktime(time.strptime(time_str, "%Y-%m-%d %H:%M:%S"))
        return ts + int(ms_str) / 1000.0

    def get_params_panel(self):
        params_lines = [
            f"[{PARAM_COLOR}]{k}[/{PARAM_COLOR}]: [bold]{self.params[k]}[/bold]"
            for k in PARAM_KEYS
        ]
        height = len(PARAM_KEYS) + 2
        return Panel(
            "\n".join(params_lines),
            title="Runtime Parameters",
            height=height,
            border_style=PARAM_COLOR
        )

    def get_log_panel(self, log_path: str, title: str) -> Panel:
        try:
            if not os.path.exists(log_path):
                return Panel("Log file not found", title=f"[{LOGS_COLOR}]{title}[/{LOGS_COLOR}]", border_style=LOGS_COLOR)
            with open(log_path, "r", encoding="utf-8") as f:
                log_content = f.read()
            max_log_lines = self.get_max_log_lines()
            lines = log_content.splitlines() if log_content else []
            shown = lines[-max_log_lines:] if lines else []
            return Panel(
                "\n".join(shown) if shown else "Log is empty",
                title=f"[{LOGS_COLOR}]{title}[/{LOGS_COLOR}]",
                border_style=LOGS_COLOR
            )
        except Exception as e:
            return Panel(
                f"Log read error: {str(e)}",
                title=f"[{LOGS_COLOR}]{title}[/{LOGS_COLOR}]",
                border_style=LOGS_COLOR
            )

    def run(self):
        layout = Layout()
        height_params = len(PARAM_KEYS) + 2
        layout.split(
            Layout(name="params", size=height_params),
            Layout(name="logs", ratio=1)
        )

        with Live(layout, refresh_per_second=4, console=self.console) as live:
            while True:
                try:
                    self.parse_runtime_params()
                    layout["params"].update(self.get_params_panel())
                    layout["logs"].update(
                        self.get_log_panel(self.log_file, "Logs")
                    )
                except Exception as e:
                    layout["logs"].update(Panel(f"Error: {str(e)}", title="Logs"))
                time.sleep(1)


if __name__ == "__main__":
    monitor = LogMonitor()
    monitor.run()
