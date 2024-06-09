# music swap
Tool to replace music in music videos using song analysis

-----------------------------------------------

### Installation

```bash
# for windows
# install WSL
# https://www.omgubuntu.co.uk/how-to-install-wsl2-on-windows-10
# (I have confirmed it works with wsl2 + ubuntu)

git clone https://github.com/amon812/music_swap.git
cd music_swap
sudo apt-get update && sudo apt-get install -y libsndfile1 rubberband-cli ffmpeg
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip uninstall -y soundfile
pip install soundfile
```

### Usage
```python
source venv/bin/activate

# simplest usage
python main.py swap1 "PATH_TO_VIDEO_FILE" "PATH_TO_MUSIC_FILE"
```

```python
# This way you can fine-tune the output.

# This command outputs a json file.
python main.py generate_draft "PATH_TO_VIDEO_FILE" "PATH_TO_MUSIC_FILE"

# Execute command with json
python main.py swap2 "PATH_TO_JSON_FILE"

# You can get better results without changing anything, but you can also edit the json file to tweak it
```
