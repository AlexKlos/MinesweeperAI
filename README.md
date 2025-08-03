[![Status: Work in Progress](https://img.shields.io/badge/status-work%20in%20progress-orange)]()

# MinesweeperAI

An experimental project exploring whether AI can learn to play Minesweeper using only visual information from the game screen—without access to the game’s rules, logic, or mathematics. The goal is to train a neural network to solve Minesweeper purely from images, making decisions without any built-in understanding of the underlying mechanics.

## Project structure

```
project_root/
├─ assets/
│  ├─ playground.png            # Screenshot of the empty game field
│  ├─ playground_30_16.png      # Resized screenshot of the game field
│  ├─ happy_face.png            # Positive indicator
│  ├─ scared_face.png           # Neutral/intermediate indicator
│  ├─ dead_face.png             # Negative indicator
│  ├─ start_playground.npy      # Empty gamefield numpy-array
│  └─ datasets/                 # Datasets for model training
│     └─ ...
├─ logs/                        # Logs
│  ├─ minesweeper_ai.log
│  └─ ...
├─ src/
│  └─ minesweeper_ai/
│      ├─ __init__.py
│      ├─ minesweeper_ai.py     # Main script
│      └─ rich_interface.py     # TUI/CLI interface for monitoring the application state
├─ project_structure.txt        # Project structure
├─ requirements.txt             # Project dependencies
├─ README.md                    # Project description
└─ .gitignore
```

## Installation and configuration

## Notes

### Authors

- [AlexKlos](https://github.com/AlexKlos)
