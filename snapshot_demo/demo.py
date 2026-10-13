"""
Demo tải ảnh snapshot - chạy ĐỘC LẬP, không đụng vào code FireGuard.

Chạy:
    python demo.py                 # dùng webcam 0 (không có thì tự dùng khung hình giả)
    python demo.py 1               # webcam số 1
    python demo.py video.mp4       # dùng file video
    python demo.py rtsp://...      # dùng camera IP

Rồi mở http://localhost:8001

LƯU TRỮ ẢNH:
  - Có MinIO đang chạy + đã `pip install minio`  -> ảnh lưu vào MinIO (bucket "snapshots"),
    tải bằng presigned URL.
  - Không có MinIO                                -> tự lưu vào thư mục snapshots/ (dự phòng).

Cấu hình MinIO bằng biến môi trường (có giá trị mặc định cho MinIO chạy local):
    MINIO_ENDPOINT   (mặc định localhost:9000)
    MINIO_ACCESS_KEY (mặc định minioadmin)
    MINIO_SECRET_KEY (mặc định minioadmin)
    MINIO_BUCKET     (mặc định snapshots)
    MINIO_SECURE     (mặc định false, đặt true nếu dùng https)

Cần: opencv-python, numpy (+ minio nếu dùng MinIO).
"""
import io
import os
import sys
import json
import time
import threading
from datetime import timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

import cv2
import numpy as np

PORT = 8001
HERE = os.path.dirname(os.path.abspath(__file__))
SNAP_DIR = os.path.join(HERE, "snapshots")
os.makedirs(SNAP_DIR, exist_ok=True)

SOURCE = sys.argv[1] if len(sys.argv) > 1 else "0"
try:
    SOURCE = int(SOURCE)
except ValueError:
    pass


# ----------------------------------------------------------------- lưu trữ
IMG_EXT = (".jpg", ".jpeg", ".png")


class LocalStore:
    """Dự phòng: lưu ảnh vào thư mục snapshots/ trên máy."""
    kind = "local"

    def info(self):
        return {"kind": "local", "where": SNAP_DIR}

    def put(self, name, data):
        with open(os.path.join(SNAP_DIR, name), "wb") as f:
            f.write(data)

    def list(self):
        files = [f for f in os.listdir(SNAP_DIR) if f.lower().endswith(IMG_EXT)]
        return sorted(files, reverse=True)

    def get(self, name):
        full = os.path.join(SNAP_DIR, name)
        if not os.path.isfile(full):
            return None
        with open(full, "rb") as f:
            return f.read()

    def download_url(self, name):
        # Cùng origin nên dùng thẳng route của server này
        return "/download/snapshot/" + name


class MinioStore:
    """Lưu ảnh vào MinIO; tải bằng presigned URL (link tạm thời có chữ ký)."""
    kind = "minio"

    def __init__(self, client, bucket, endpoint):
        self.client, self.bucket, self.endpoint = client, bucket, endpoint

    def info(self):
        return {"kind": "minio", "where": "%s / bucket '%s'" % (self.endpoint, self.bucket)}

    def put(self, name, data):
        self.client.put_object(self.bucket, name, io.BytesIO(data), len(data),
                               content_type="image/jpeg")

    def list(self):
        objs = [o for o in self.client.list_objects(self.bucket)
                if o.object_name.lower().endswith(IMG_EXT)]
        objs.sort(key=lambda o: o.last_modified, reverse=True)
        return [o.object_name for o in objs]

    def get(self, name):
        try:
            resp = self.client.get_object(self.bucket, name)
        except Exception:
            return None
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()

    def download_url(self, name):
        # Link hết hạn sau 5 phút; header ép trình duyệt TẢI VỀ thay vì mở ảnh
        return self.client.presigned_get_object(
            self.bucket, name, expires=timedelta(minutes=5),
            response_headers={"response-content-disposition": 'attachment; filename="%s"' % name},
        )


def make_store():
    endpoint = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
    access = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
    secret = os.environ.get("MINIO_SECRET_KEY", "minioadmin")
    bucket = os.environ.get("MINIO_BUCKET", "snapshots")
    secure = os.environ.get("MINIO_SECURE", "false").lower() == "true"
    try:
        from minio import Minio
    except ImportError:
        print("[!] Chưa cài thư viện minio (pip install minio) -> lưu vào thư mục snapshots/.")
        return LocalStore()
    try:
        client = Minio(endpoint, access_key=access, secret_key=secret, secure=secure)
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            print("[MinIO] Đã tạo bucket '%s'" % bucket)
        print("[MinIO] Kết nối OK: %s, bucket '%s'" % (endpoint, bucket))
        return MinioStore(client, bucket, endpoint)
    except Exception as e:
        print("[!] Không kết nối được MinIO (%s): %s" % (endpoint, e))
        print("    -> lưu vào thư mục snapshots/ thay thế.")
        return LocalStore()


store = make_store()


# ----------------------------------------------------------------- camera
class Camera:
    """Một luồng đọc khung hình liên tục, giữ khung mới nhất (giống camera_manager.py)."""

    def __init__(self, source):
        self.source = source
        self.frame = None
        self.frame_id = 0
        self.lock = threading.Lock()
        self.cap = cv2.VideoCapture(source)
        self.is_fake = not self.cap.isOpened()
        if self.is_fake:
            print("[!] Không mở được camera -> dùng khung hình giả để demo.")
        threading.Thread(target=self._loop, daemon=True).start()

    def _fake_frame(self):
        img = np.full((480, 640, 3), (60, 40, 20), np.uint8)
        t = time.time()
        x = int(320 + 200 * np.sin(t))
        cv2.circle(img, (x, 240), 40, (0, 120, 255), -1)
        cv2.putText(img, "DEMO FRAME", (180, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(img, time.strftime("%Y-%m-%d %H:%M:%S"), (190, 440),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 255, 200), 2)
        return img

    def _loop(self):
        while True:
            if self.is_fake:
                frame = self._fake_frame()
                time.sleep(1 / 30)
            else:
                ok, frame = self.cap.read()
                if not ok:  # hết video -> quay lại đầu
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.05)
                    continue
                time.sleep(1 / 30)
            with self.lock:
                self.frame, self.frame_id = frame, self.frame_id + 1

    def jpeg(self):
        """Mã hóa khung mới nhất thành JPEG bytes (giống Camera.get_jpeg)."""
        with self.lock:
            frame = None if self.frame is None else self.frame.copy()
        if frame is None:
            return None
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return buf.tobytes() if ok else None


camera = Camera(SOURCE)


# ----------------------------------------------------------------- server
class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.address_string(), fmt % args))

    def _send(self, code, body=b"", ctype="text/plain; charset=utf-8", headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj).encode(), "application/json")

    def do_GET(self):
        path = unquote(self.path.split("?")[0])

        if path == "/":
            with open(os.path.join(HERE, "index.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")

        # Luồng video trực tiếp (MJPEG) - để xem camera trên trang
        if path == "/video_feed":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            last = -1
            try:
                while True:
                    if camera.frame_id == last:
                        time.sleep(0.01)
                        continue
                    last = camera.frame_id
                    jpg = camera.jpeg()
                    if jpg:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                return

        # (1) CHỤP NGAY khung hiện tại và trả về dạng file tải xuống
        if path == "/snapshot":
            jpg = camera.jpeg()
            if jpg is None:
                return self._json(503, {"detail": "Chưa có khung hình"})
            name = "cam_%s.jpg" % time.strftime("%Y%m%d_%H%M%S")
            return self._send(200, jpg, "image/jpeg", {
                "Content-Disposition": 'attachment; filename="%s"' % name,
                "Cache-Control": "no-store",
            })

        # Thông tin nơi lưu ảnh (local hay MinIO)
        if path == "/storage":
            return self._json(200, store.info())

        # Danh sách ảnh đã lưu
        if path == "/snapshots":
            try:
                return self._json(200, {"files": store.list()})
            except Exception as e:
                return self._json(500, {"detail": "Lỗi đọc danh sách: %s" % e})

        # Xem ảnh đã lưu (hiển thị trong trang) - server đọc từ kho rồi trả về
        if path.startswith("/snapshots/"):
            return self._serve_saved(path[len("/snapshots/"):], download=False)

        # Lấy link tải: MinIO -> presigned URL, local -> route của server
        if path.startswith("/api/download-url/"):
            name = self._safe_name(path[len("/api/download-url/"):])
            if name is None:
                return self._json(400, {"detail": "Tên file không hợp lệ"})
            try:
                return self._json(200, {"url": store.download_url(name)})
            except Exception as e:
                return self._json(500, {"detail": "Không tạo được link: %s" % e})

        # Tải ảnh dạng attachment (dùng cho chế độ local)
        if path.startswith("/download/snapshot/"):
            return self._serve_saved(path[len("/download/snapshot/"):], download=True)

        self._json(404, {"detail": "Not Found"})

    def do_POST(self):
        # Lưu snapshot vào kho (MinIO hoặc thư mục)
        if self.path == "/snapshot/save":
            jpg = camera.jpeg()
            if jpg is None:
                return self._json(503, {"detail": "Chưa có khung hình"})
            name = "event_%s.jpg" % time.strftime("%Y%m%d_%H%M%S")
            try:
                store.put(name, jpg)
            except Exception as e:
                return self._json(500, {"detail": "Lưu thất bại: %s" % e})
            return self._json(200, {"saved": name, "storage": store.kind})
        self._json(404, {"detail": "Not Found"})

    @staticmethod
    def _safe_name(filename):
        """Chặn path traversal (../../file) và chỉ nhận file ảnh."""
        safe = os.path.basename(filename)
        if safe != filename or not safe.lower().endswith(IMG_EXT):
            return None
        return safe

    def _serve_saved(self, filename, download):
        safe = self._safe_name(filename)
        if safe is None:
            return self._json(400, {"detail": "Tên file không hợp lệ"})
        data = store.get(safe)
        if data is None:
            return self._json(404, {"detail": "Không tìm thấy ảnh"})
        headers = {"Content-Disposition": 'attachment; filename="%s"' % safe} if download else {}
        self._send(200, data, "image/jpeg", headers)


if __name__ == "__main__":
    print("Demo chạy tại http://localhost:%d  (Ctrl+C để dừng)" % PORT)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
