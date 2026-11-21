import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst
import numpy as np, cv2

Gst.init(None)
pipeline = Gst.parse_launch(
    "rtspsrc location=rtsp://192.168.2.9/depth latency=30 ! rtpgstdepay ! "
    "videoconvert ! appsink name=sink emit-signals=false max-buffers=1 drop=true sync=false"
)
sink = pipeline.get_by_name("sink")
pipeline.set_state(Gst.State.PLAYING)

try:
    while True:
        sample = sink.emit("try-pull-sample", 2 * Gst.SECOND)
        if sample is None:
            continue
        caps = sample.get_caps().get_structure(0)
        w, h, fmt = caps.get_value("width"), caps.get_value("height"), caps.get_value("format")
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            continue
        # print format once to confirm what the arm actually sends
        if "printed" not in globals():
            print("depth caps:", caps.to_string()); printed = True
        depth = np.frombuffer(info.data, dtype=np.uint16 if "16" in fmt else np.uint8).reshape(h, w).copy()
        buf.unmap(info)

        # replace the vis/imshow lines with:
        h2, w2 = depth.shape
        center = depth[h2//2 - 5:h2//2 + 5, w2//2 - 5:w2//2 + 5]
        valid = center[center > 0]                      # 0 usually means "no depth"
        print("center median:", np.median(valid) if valid.size else "invalid")

        FIXED_MAX = 5000                                # try the units below, then adjust
        vis = np.clip(depth.astype(np.float32) / FIXED_MAX * 255, 0, 255).astype(np.uint8)
        cv2.imshow("Kinova depth", cv2.applyColorMap(vis, cv2.COLORMAP_JET))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    pipeline.set_state(Gst.State.NULL)
    cv2.destroyAllWindows()