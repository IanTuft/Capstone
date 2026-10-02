import cv2
import numpy as np

from typing import Optional, Sequence


class ObjectRecognizer:
    def __init__(self):
        self.mode: str = "Detecting"
        self.state: str = "Tracking not started"
        self.tracker: Optional[cv2.TrackerKCF] = None


def get_objectness_threshold() -> float:
    # Objectness threshold
    return 0.5


def get_conf_threshold() -> float:
    # Confidence threshold
    return 0.5


def get_nms_threshold() -> float:
    # Non-maximum suppression threshold
    return 0.4


def main() -> None:
    # Give the configuration and weight files for the model and load the network using them.
    model_configuration: str = "yolov3.cfg"

    # The .weights file stores the trained parameters.
    # The network can load these pre-trained weights to save time and computational resources
    # instead of training the model from scratch
    model_weights: str = "yolov3.weights"

    # Takes the configuration and weights and builds a neural network model.
    # It initializes the model with the pre-trained weights,
    # so you can use it directly for object detection without retraining.
    net: cv2.dnn.Net = cv2.dnn.readNetFromDarknet(model_configuration, model_weights)

    # Process inputs
    video_path: str = "soccer-ball.mp4"

    # Open the video file
    cap: cv2.VideoCapture = cv2.VideoCapture(video_path)

    # Read the first frame of the video
    has_frame, frame = cap.read()

    object_recognizer: ObjectRecognizer = ObjectRecognizer()

    while has_frame:
        if object_recognizer.tracker is None:
            identity_object(object_recognizer, frame, net)
        else:
            track_object(object_recognizer, frame)

        draw_object_state_and_mode(object_recognizer, frame)

        # Display the frame
        cv2.imshow("frame", frame)

        # Check for escape key
        key: int = cv2.waitKey(1)

        if key == 27:
            break

        # Read the next frame
        has_frame, frame = cap.read()

    # Close the video file
    cap.release()

    # Close all windows
    cv2.destroyAllWindows()


def identity_object(object_recognizer: ObjectRecognizer, frame: np.ndarray, net: cv2.dnn.Net) -> None:
    input_frame_width: int = 416  # Width of network's input image
    input_frame_height: int = 416  # Height of network's input image

    # Load names of classes
    classes_file: str = "coco.names"
    with open(classes_file, 'rt') as f:
        classes: list[str] = f.read().rstrip('\n').split('\n')

    # Create a 4D blob from a frame. The 4Ds are:
    # 1. Batch size (N): This refers to the number of images processed at once.
    #    Since YOLOv3 processes one image at a time, this is typically set to 1.
    # 2. Each image consists of three color channels: Red, Green, and Blue (RGB).
    #    In most color images, there are 3 channels.
    # 3. The height of the image.
    # 4. The width of the image.
    blob: np.ndarray = cv2.dnn.blobFromImage(frame, 1 / 255, (input_frame_width, input_frame_height),
                                             [0, 0, 0], True, crop=False)

    # Sets the input by feeding the processed frame into the YOLO neural network.
    net.setInput(blob)

    # Performing a forward pass of the neural network (YOLO model) i.e. explicitly specifying which layers'
    # outputs you want to retrieve.
    # Each element in outs contains a tuple that contains:
    # 1. Bounding Box Coordinates (center coordinates, width, height)
    # 2. Class Scores, the probability scores for each detected object belonging to a specific class.
    # 3. Confidence Scores, the confidence that the detected object actually belongs to the predicted class.
    outs: Sequence[Sequence[np.ndarray]] = net.forward(get_output_layer_names(net))

    # Get the "best" object that represents a ball.
    ball_bounding_box: Optional[tuple[int, ...]] = post_process(frame, outs, classes)

    if ball_bounding_box is not None:
        # Draw a blue bounding box to indicate detection mode
        p1: tuple[int, int] = (ball_bounding_box[0], ball_bounding_box[1])
        p2: tuple[int, int] = (ball_bounding_box[0] + ball_bounding_box[2],
                               ball_bounding_box[1] + ball_bounding_box[3])
        cv2.rectangle(frame, p1, p2, (255, 178, 50), 3)

        # Initialize tracker with frame and bounding box
        # object_recognizer.tracker = cv2.TrackerKCF.create()
        # object_recognizer.tracker.init(frame, ball_bounding_box)

        # Switch to tracking mode
        object_recognizer.mode = "Tracking"
        object_recognizer.state = "Tracking started"

        # Display the frame
        cv2.imshow("frame", frame)

        cv2.waitKey(0)


# Get the names of the output layers
def get_output_layer_names(net: cv2.dnn.Net) -> list[str]:
    # A layer is a collection of nodes (or neurons) that perform a transformation on the input data.
    # Each layer takes inputs, processes them, and passes the output to the next layer.
    # Get the names of all the layers in the network.
    layers_names: Sequence[str] = net.getLayerNames()

    # Get the names of the output layers, i.e. the layers with unconnected outputs
    # The method net.getUnconnectedOutLayers() returns the indices of the layers that are considered "output layers"
    # in the network. These are layers that have no other layers connected to their outputs,
    # meaning they produce the final results of the neural network.
    return [layers_names[i - 1] for i in net.getUnconnectedOutLayers()]


# Remove the bounding boxes with low confidence using non-maxima suppression
def post_process(frame: np.ndarray,
                 outs: Sequence[Sequence[np.ndarray]],
                 classes: list[str]) -> Optional[tuple[int, ...]]:
    frame_height: int = frame.shape[0]
    frame_width: int = frame.shape[1]

    # Scan through all the bounding boxes output from the neural network and keep only the
    # ones with high confidence scores.
    confidences: list[float] = []
    boxes: list[list[int]] = []

    # We are only interested in the "sports ball" class
    sports_ball_class_id: int = classes.index("sports ball")

    for out in outs:
        for detection in out:
            # Is this an object of some kind?
            if detection[4] > get_objectness_threshold():
                scores: np.ndarray = detection[5:]
                class_id: np.int64 = np.argmax(scores)
                confidence: float = float(scores[class_id])
                # How confidant are we that this object is a soccer ball?
                if (confidence > get_conf_threshold()) and (class_id == sports_ball_class_id):
                    center_x: int = int(detection[0] * frame_width)
                    center_y: int = int(detection[1] * frame_height)
                    width = int(detection[2] * frame_width)
                    height = int(detection[3] * frame_height)
                    left: int = center_x - (width // 2)
                    top: int = center_y - (height // 2)
                    confidences.append(confidence)
                    boxes.append([left, top, width, height])

    # Performs Non-Maximum Suppression (NMS) to remove redundant overlapping bounding
    # boxes based on their confidence scores.
    indices: Sequence[int] = cv2.dnn.NMSBoxes(boxes, confidences, get_conf_threshold(), get_nms_threshold())

    if len(indices) > 0:
        # Return the first result
        box: list[int] = boxes[0]
        left: int = box[0]
        top: int = box[1]
        width: int = box[2]
        height: int = box[3]

        return left, top, width, height

    return None


def track_object(object_recognizer: ObjectRecognizer, frame: np.ndarray):
    # Update tracker
    success, ball_bounding_box = object_recognizer.tracker.update(frame)

    if success:
        # Tracking success
        object_recognizer.state = "Tracking success"
        # Draw bounding box
        p1: tuple[int, int] = (int(ball_bounding_box[0]), int(ball_bounding_box[1]))
        p2: tuple[int, int] = (int(ball_bounding_box[0] + ball_bounding_box[2]),
                               int(ball_bounding_box[1] + ball_bounding_box[3]))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 3)
    else:
        # Tracking failure
        object_recognizer.state = "Tracking failure"
        object_recognizer.mode = "Detecting"
        object_recognizer.tracker = None


def draw_object_state_and_mode(object_recognizer: ObjectRecognizer, frame: np.ndarray) -> None:
    # Draw a black box behind text
    cv2.rectangle(frame, (0, 0), (300, 40), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, "Mode: " + object_recognizer.mode, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255))

    color: tuple[int, int, int] = (255, 255, 255)

    if "failure" in object_recognizer.state:
        color = (0, 0, 255)

    cv2.putText(frame, "State: " + object_recognizer.state, (0, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                color)


if __name__ == "__main__":
    main()
