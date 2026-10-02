Currently included are the files left by the previous team
Planned:
-Capture At least 3 video feeds, at least 8 frames per second each.
-Merge the frames into one output file at 24 frames per second or more
-Align Common objects land on the same coordinates in every frame. Automatic, with a manual option.

## Setup

### Requirements
- Python 3.9 or newer (tested on 3.14)
- `opencv-contrib-python` 4.x. **Do not use OpenCV 5**, it removed `readNetFromDarknet`, which `tracking.py` needs.
- `numpy`
- `yolov3.weights` (~248 MB), needed by `tracking.py`. Too large for GitHub, so download it yourself from
  https://pjreddie.com/media/files/yolov3.weights and put it in `CSE485-486-CapstoneProject-main/`. It's gitignored.
- A webcam, only for `testgrid.py`

### Install
From the `CSE485-486-CapstoneProject-main` folder in a terminal:
```
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python check_env.py
```
`check_env.py` should print `Ready to run.` If PowerShell blocks the activate script, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once and try again.

### VS Code
1. Ctrl+Shift+P > **Python: Select Interpreter** > pick `CSE485-486-CapstoneProject-main\.venv\Scripts\python.exe`.
   If the Run button gives `No module named 'cv2'`, this is why.
2. Use the Run and Debug panel (Ctrl+Shift+D). The configs in `.vscode/launch.json` set the working folder for each script.

### Running the scripts
The scripts load files by relative path, so run them from `CSE485-486-CapstoneProject-main`, not the repo root.

| Script | What it does | Notes |
|---|---|---|
| `main.py` | Simulated ball bouncing on `soccer_field.png` with a broadcast frame following it | Any key closes it |
| `tracking.py` | YOLOv3 detects the ball in `soccer-ball.mp4` | Pauses after every frame, press a key to advance. Esc quits. |
| `testgrid.py` | Tracks an orange object on the webcam and reports which 3x3 grid cells it's in | `q` quits |

### Known issues in the previous team's code
- `tracking.py`: the KCF tracker setup is commented out, so it re-runs detection every frame and calls `waitKey(0)` each time.
- `tracking.py`: `post_process` returns `boxes[0]` instead of the best box after non-max suppression.
