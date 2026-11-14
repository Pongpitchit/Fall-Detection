import cv2
from ultralytics import YOLO
import cvzone
import requests
from datetime import datetime
import os
import tempfile

# ========= CONFIG =========
LINE_ACCESS_TOKEN = os.environ["LINE_ACCESS_TOKEN"]
LINE_USER_ID = os.environ["LINE_USER_ID"]
IMGUR_CLIENT_ID = os.environ["IMGUR_CLIENT_ID"]

# ========= UPLOAD TO IMGUR =========
def upload_image_to_imgur(image_path):
    headers = {"Authorization": f"Client-ID {IMGUR_CLIENT_ID}"}
    with open(image_path, 'rb') as img:
        files = {'image': img}
        response = requests.post("https://api.imgur.com/3/image", headers=headers, files=files)
        if response.status_code == 200:
            return response.json()['data']['link']
        else:
            print("❌ Imgur Upload Failed:", response.text)
            return None

# ========= SEND LINE NOTIFICATION =========
def send_line_alert(frame, track_id):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fall_{track_id}_{timestamp}.jpg"
    temp_dir = tempfile.gettempdir()
    filepath = os.path.join(temp_dir, filename)

    print("📁 Saving image to:", filepath)
    cv2.imwrite(filepath, frame)

    image_url = upload_image_to_imgur(filepath)
    if not image_url:
        return

    message_text = f"❗ ตรวจพบการล้มของบุคคล (Track ID: {track_id})"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {LINE_ACCESS_TOKEN}"
    }

    data = {
        "to": LINE_USER_ID,
        "messages": [
            {"type": "text", "text": message_text},
            {"type": "image", "originalContentUrl": image_url, "previewImageUrl": image_url}
        ]
    }

    try:
        response = requests.post("https://api.line.me/v2/bot/message/push", headers=headers, json=data)
        print("📲 LINE Sent:", response.status_code, response.text)
    except Exception as e:
        print("❌ LINE Error:", e)

    if os.path.exists(filepath):
        os.remove(filepath)

# ========= YOLO + FALL DETECTION =========
model = YOLO("best.pt")  # โหลดโมเดล YOLO ที่เทรนไว้แล้ว
names = model.model.names  # รายชื่อคลาสในโมเดล

# 📹 เปิดวิดีโอจากไฟล์ หรือเปลี่ยนเป็นกล้องได้โดยใช้ cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('fall.mp4')  # ใช้ไฟล์วิดีโอ fall.mp4 ในการตรวจจับ  # หรือใส่ 0 เพื่อใช้กล้อง: cap = cv2.VideoCapture(0)

cv2.namedWindow("Fall Detection")
count = 0
fall_alerts = {}  # บันทึก Track ID ที่แจ้งเตือนแล้ว เพื่อไม่ส่งซ้ำ

while True:
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue  # ประมวลผลทุก 3 เฟรม เพื่อลดโหลด

    frame = cv2.resize(frame, (1020, 600))  # ปรับขนาดเฟรม
    results = model.track(frame, persist=True)  # ตรวจจับ + ติดตามด้วย YOLOv8

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        confidences = results[0].boxes.conf.cpu().tolist()

        for box, class_id, track_id, conf in zip(boxes, class_ids, track_ids, confidences):
            x1, y1, x2, y2 = box
            h = y2 - y1
            w = x2 - x1
            thresh = h - w  # เช็คว่าคนล้มโดยดูว่า "นอน" (กว้างมากกว่าสูง)

            if thresh <= 0:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # สีแดง = ล้ม
                cvzone.putTextRect(frame, f"Fall ({track_id})", (x1, y1 - 10), 1, 1, (0, 0, 255), 2)

                if track_id not in fall_alerts:
                    send_line_alert(frame, track_id)
                    fall_alerts[track_id] = count
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # สีเขียว = ปกติ
                cvzone.putTextRect(frame, f"Person ({track_id})", (x1, y1 - 10), 1, 1, (0, 255, 0), 2)

    cv2.imshow("Fall Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
