# Maia AB test analysis
Run some analysis on the recent Maia chess bots AB test

## Requirements
You need to have Anaconda or Miniconda installed and available with the command `conda`

## Setup
Clone the repository, then run

```bash
make dep.install  # create a conda environment and install dependencies
conda activate maia-analysis  # activate the environment
conda develop src  # makes the source code importable
```

You should then symlink the bot_logs into the data folder with

```bash
ln -s /path/to/bot_logs ./data/bot_logs
```

## Usage
Documentation coming soon (in theory).
