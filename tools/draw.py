from time import time
import cv2

_fps_last_time = time()
_fps_count = 0
_fps = 0

def fpsmeter(frame, update_rate=1.0):
    global _fps_last_time, _fps_count, _fps
    _fps_count += 1
    if update_rate < time() - _fps_last_time:
        _fps = _fps_count/(time()-_fps_last_time)
        _fps_count = 0
        _fps_last_time = time()

    fps_frame = cv2.putText(
            frame,
            f"{int(_fps)}FPS",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
        )
    return fps_frame, _fps

def boundingBox(frame, xmin, ymin, xmax, ymax, name, confidence):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = 255
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
    cv2.putText(
        frame,
        f"{name} {round(confidence * 100, 2)}%",
        (xmin + 5, ymin + 15),
        font,
        0.8,
        0,
        2,
    )
    return frame