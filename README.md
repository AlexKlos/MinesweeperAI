# MinesweeperAI

An experimental project exploring whether AI can learn to play Minesweeper using only visual information from the game screen—without access to the game’s rules, logic, or mathematics. The goal is to train a neural network to solve Minesweeper purely from images, making decisions without any built-in understanding of the underlying mechanics.

## Project structure

```
project_root/
├─ assets/
│  ├─ playground.png            # Screenshot of the game field
│  ├─ happy_face.png            # Positive indicator
│  ├─ scared_face.png           # Neutral/intermediate indicator
│  ├─ dead_face.png             # Negative indicator
│  └─ datasets/                 # Datasets for model training
│     └─ ...
├─ logs/                        # Logs
│  ├─ main_pipeline.log
│  ├─ manager.log
│  └─ ...
├─ src/
│  └─ minesweeper_ai/
│      ├─ __init__.py
│      ├─ main_pipeline.py      # Main pipeline (screenshot - inference - click)
│      ├─ manager.py            # Process and application manager
│      ├─ rich_interface.py     # TUI/CLI interface for monitoring the application state
│      ├─ ipc.py                # Inter-process communication utilities (shared memory, events, etc.)
│      └─ types.py              # Common data structures and enums for statuses
├─ project_structure.txt        # Project structure
├─ requirements.txt             # Project dependencies
├─ README.md                    # Project description
├─ scheme.png                   # Application architecture diagram
└─ .gitignore
```

## Installation and configuration



## Notes
### Authors

- [AlexKlos](https://github.com/AlexKlos)
