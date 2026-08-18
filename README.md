# Hermes Autonomous Agent: TTS Optimization Tutorial

This repository contains a practical, hands-on tutorial for working with **[Hermes Agent](https://hermes-agent.nousresearch.com/docs/getting-started/installation)**, an open-source, autonomous AI framework developed by Nous Research. 

Unlike standard conversational chatbots, Hermes operates as a persistent, self-improving assistant capable of autonomous planning, tool execution, and dynamic problem-solving. In this tutorial, you will walk through a real-world engineering workflow: baseline TTS, performance bottleneck identification via MLflow observability, and hardware-accelerated pipeline optimization on AMD hardware.

---

## 🛠️ Quick Start Setup

Follow these steps to set up your local environment and launch the interactive tutorial notebook.

### 1. Clone the Repository
Open your terminal and clone this repository.
```bash
git clone https://github.com/Jereshea/hermes-audio-notebook.git
cd hermes-audio-notebook
git checkout tts
```
### 2. Create and Activate a Virtual Environment
Isolate your project dependencies by spinning up a clean Python virtual environment:
```bash
sudo apt install python3-venv  
python3 -m venv env
source env/bin/activate
```
### 3. Install Dependencies
Install all required frameworks, libraries, and tracking tools automatically from the requirements manifest:
```bash
python -m  pip install --upgrade pip
python -m pip install -r requirements.txt
```
## 🛠️ Running the Tutorial Notebook

JupyterLab defaults to a light theme, but the Hermes environment is designed with a dark theme. To ensure a consistent and comfortable experience, we recommend forcing JupyterLab into dark mode before launching.

Run the following commands to configure your environment and begin interacting with the codebase:

```bash
# 1. Ensure the setting directories exist
mkdir -p ./env/share/jupyter/lab/settings
mkdir -p ~/.jupyter/lab/user-settings/@jupyterlab/apputils-extension

# 2. Apply the dark theme to both the virtual environment and global settings
echo '{"@jupyterlab/apputils-extension:themes": {"theme": "JupyterLab Dark"}}' > ./env/share/jupyter/lab/settings/overrides.json
echo '{"theme": "JupyterLab Dark"}' > ~/.jupyter/lab/user-settings/@jupyterlab/apputils-extension/themes.jupyterlab-settings 

# 3. Launch JupyterLab
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

## 🛠️ Running the Test on terminal

1. In Terminal 1 run: `bash helper.sh`
2. In Terminal 2 run: `hermes` 