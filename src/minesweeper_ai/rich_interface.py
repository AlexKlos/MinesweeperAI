import os
import re
import time

from rich.console import Console
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel

PARAM_KEYS = [
    "Status",
    "Step",
    "Average Step Time"
]
PARAM_COLOR = "#9C85CC"
LOGS_COLOR = "#62A6A8"


class LogMonitor:
    def __init__(self):
        self.console = Console()
        self.log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../logs"))
        self.manager_log = os.path.join(self.log_dir, "manager.log")
        self.pipeline_log = os.path.join(self.log_dir, "main_pipeline.log")
        self.params = {k: "" for k in PARAM_KEYS}
        self.step_times = []
        self.last_step_timestamp = None
        self.paused = True
        self.last_status = "Pause"
        self.last_step_num = 0

    def parse_runtime_params(self):
        if not os.path.exists(self.manager_log):
            self.params = {k: "?" for k in PARAM_KEYS}
            return

        step_time_pattern = re.compile(r"Step (\d+): Sending start_event|Step (\d+): sending start_event", re.IGNORECASE)
        pause_pattern = re.compile(r"PAUSE activated|Manager is PAUSED", re.IGNORECASE)
        unpause_pattern = re.compile(r"PAUSE deactivated|Resuming main loop", re.IGNORECASE)

        with open(self.manager_log, encoding="utf-8") as f:
            lines = f.readlines()

        self.step_times.clear()
        self.last_step_timestamp = None
        self.paused = False
        self.last_status = "Active"
        self.last_step_num = 0

        step_timestamps = []
        last_unpause_idx = 0

        for idx, line in enumerate(lines):
            if pause_pattern.search(line):
                self.paused = True
                self.last_status = "Pause"
                last_unpause_idx = idx + 1
                step_timestamps = []
            elif unpause_pattern.search(line):
                self.paused = False
                self.last_status = "Active"
                last_unpause_idx = idx
                step_timestamps = []
            elif step_time_pattern.search(line):
                try:
                    ts = self.extract_timestamp(line)
                    step_timestamps.append(ts)
                    match = step_time_pattern.search(line)
                    step_num = match.group(1) or match.group(2)
                    if step_num:
                        self.last_step_num = int(step_num)
                except Exception:
                    pass

        avg_step_time = ""
        if len(step_timestamps) >= 2:
            step_intervals = [
                step_timestamps[i + 1] - step_timestamps[i]
                for i in range(len(step_timestamps) - 1)
            ]
            avg_sec = sum(step_intervals) / len(step_intervals)
            avg_step_time = f"{avg_sec:.2f} sec"
        else:
            avg_step_time = "-"

        self.params["Status"] = self.last_status
        self.params["Step"] = str(self.last_step_num)
        self.params["Average Step Time"] = avg_step_time

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
            return Panel(
                log_content[-1800:] if log_content else "Log is empty",
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
        
        layout["logs"].split_row(
            Layout(name="manager_logs"),
            Layout(name="pipeline_logs")
        )

        with Live(layout, refresh_per_second=4, console=self.console) as live:
            while True:
                try:
                    self.parse_runtime_params()
                    layout["params"].update(self.get_params_panel())
                    layout["logs"]["manager_logs"].update(
                        self.get_log_panel(self.manager_log, "Manager Logs")
                    )
                    layout["logs"]["pipeline_logs"].update(
                        self.get_log_panel(self.pipeline_log, "Main Pipeline Logs")
                    )
                except Exception as e:
                    layout["logs"].update(Panel(f"Error: {str(e)}", title="Logs"))
                time.sleep(1)


if __name__ == "__main__":
    monitor = LogMonitor()
    monitor.run()
