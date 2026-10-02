import importlib.util
import sys
from pathlib import Path

PROJECT = Path(__file__).parent
ok = True


def report(passed, label, hint=""):
    global ok
    ok &= passed
    print(f"[{'OK' if passed else 'MISSING'}] {label}" + (f"  ->  {hint}" if hint and not passed else ""))


report(sys.version_info >= (3, 9), f"Python {sys.version.split()[0]}", "need 3.9+ (3.12 recommended)")

for pkg in ("cv2", "numpy"):
    report(importlib.util.find_spec(pkg) is not None, pkg, "pip install -r requirements.txt")

try:
    import cv2
    report(cv2.__version__.startswith("4."), f"OpenCV {cv2.__version__} is 4.x", 'pip install "opencv-contrib-python<5"')
    report(hasattr(cv2.dnn, "readNetFromDarknet"), "cv2.dnn.readNetFromDarknet")
    report(hasattr(cv2, "TrackerKCF"), "cv2.TrackerKCF (contrib)", "uninstall opencv-python, install opencv-contrib-python")
except ImportError:
    pass

for name in ("soccer_field.png", "soccer-ball.mp4", "yolov3.cfg", "coco.names"):
    report((PROJECT / name).exists(), name, "should be in the repo, try git pull")

weights = PROJECT / "yolov3.weights"
report(weights.exists() and weights.stat().st_size > 200_000_000, "yolov3.weights",
       f"download https://pjreddie.com/media/files/yolov3.weights into this folder")

if "cv2" in sys.modules:
    cam = cv2.VideoCapture(0)
    report(cam.isOpened(), "webcam 0 (testgrid.py only)", "plug in or allow camera access")
    cam.release()

print("\nReady to run." if ok else "\nFix the MISSING items above.")
