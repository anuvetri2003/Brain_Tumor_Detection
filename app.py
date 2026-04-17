import os
from datetime import datetime
from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from pymongo import MongoClient
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
GRADCAM_FOLDER = "static/gradcam"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

model = YOLO("best.pt")

# ===== MongoDB Connection =====
client = MongoClient("mongodb://localhost:27017/")
db = client["brain_tumor_db"]
reports_collection = db["reports"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/detection")
def detection():
    return render_template("detection.html")


@app.route("/reports")
def reports():
    reports_data = list(reports_collection.find().sort("timestamp", -1))
    return render_template("reports.html", reports=reports_data)


@app.route("/database")
def database():
    reports_data = list(reports_collection.find().sort("timestamp", -1))
    return render_template("database.html", reports=reports_data)


# ================= DETECT =================
@app.route("/detect", methods=["POST"])
def detect():
    patient_name = request.form.get("patient_name", "Not Provided")
    age = request.form.get("age", "Not Provided")
    scan_date_raw = request.form.get("scan_date", "")

    if scan_date_raw:
        scan_date = datetime.strptime(scan_date_raw, "%Y-%m-%d").strftime("%d-%m-%Y")
    else:
        scan_date = "Not Provided"

    file = request.files.get("mri_image")
    if not file or file.filename == "":
        return "No file uploaded", 400

    filename = secure_filename(file.filename)
    image_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(image_path)

    results = model(image_path, conf=0.05)
    boxes = results[0].boxes

    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    gradcam_img = original_img.copy()

    img_h, img_w = original_img.shape[:2]
    image_area = img_w * img_h

    detected_regions = 0
    tumor_type_list = []
    confidence_list = []

    estimated_risk_level = "Normal"
    max_percentage = 0.0

    if boxes is not None and len(boxes) > 0:
        for box, cls_id, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(original_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            tumor_crop = gradcam_img[y1:y2, x1:x2]

            if tumor_crop.size != 0:
                gray = cv2.cvtColor(tumor_crop, cv2.COLOR_RGB2GRAY)
                heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(tumor_crop, 0.6, heatmap, 0.4, 0)
                gradcam_img[y1:y2, x1:x2] = overlay

            detected_regions += 1
            tumor_type_list.append(model.names[int(cls_id)])
            confidence_list.append(conf * 100)

            tumor_width = x2 - x1
            tumor_height = y2 - y1
            tumor_area = tumor_width * tumor_height
            percentage = (tumor_area / image_area) * 100

            if percentage > max_percentage:
                max_percentage = percentage

        status = "Tumor Detected"

        # Estimated risk level based on largest detected tumor region
        if max_percentage < 5:
            estimated_risk_level = "Low"
        elif max_percentage < 15:
            estimated_risk_level = "Medium"
        else:
            estimated_risk_level = "High"

    else:
        status = "No Tumor Detected"

    result_filename = "result_" + filename
    gradcam_filename = "gradcam_" + filename

    Image.fromarray(original_img).save(os.path.join(RESULT_FOLDER, result_filename))
    Image.fromarray(gradcam_img).save(os.path.join(GRADCAM_FOLDER, gradcam_filename))

    # Remove duplicate tumor types but keep original order
    unique_tumor_types = list(dict.fromkeys(tumor_type_list))
    tumor_type_text = ", ".join(unique_tumor_types) if unique_tumor_types else "None"

    confidence_text = ", ".join([f"{c:.2f}" for c in confidence_list]) if confidence_list else "0.00"
    estimated_tumor_region_percentage = round(max_percentage, 2)

    # ===== SAVE DATA TO MONGODB =====
    report_data = {
        "patient_name": patient_name,
        "age": age,
        "scan_date": scan_date,
        "status": status,
        "tumor_type": tumor_type_text,
        "confidence": confidence_text,
        "detected_regions": detected_regions,
        "estimated_risk_level": estimated_risk_level,
        "estimated_tumor_region_percentage": estimated_tumor_region_percentage,
        "original_image": "uploads/" + filename,
        "result_image": "results/" + result_filename,
        "gradcam_image": "gradcam/" + gradcam_filename,
        "timestamp": datetime.now()
    }

    reports_collection.insert_one(report_data)

    return render_template(
        "result.html",
        original_image="uploads/" + filename,
        result_image="results/" + result_filename,
        gradcam_image="gradcam/" + gradcam_filename,
        patient_name=patient_name,
        age=age,
        scan_date=scan_date,
        status=status,
        tumor_type=tumor_type_text,
        confidence=confidence_text,
        regions=detected_regions,
        estimated_risk_level=estimated_risk_level,
        estimated_tumor_region_percentage=f"{estimated_tumor_region_percentage:.2f}"
    )


# ================= PDF DOWNLOAD =================
@app.route("/download_report")
def download_report():
    patient_name = request.args.get("patient_name", "Not Provided")
    age = request.args.get("age", "")
    scan_date = request.args.get("scan_date", "")
    status = request.args.get("status", "")
    tumor_type = request.args.get("tumor_type", "")
    confidence = request.args.get("confidence", "")
    regions = request.args.get("regions", "")
    estimated_risk_level = request.args.get("estimated_risk_level", "")
    estimated_tumor_region_percentage = request.args.get("estimated_tumor_region_percentage", "")

    original_image = request.args.get("original_image")
    result_image = request.args.get("result_image")
    gradcam_image = request.args.get("gradcam_image")

    rows = [
        ("Name", patient_name),
        ("Age", age),
        ("Scan Date", scan_date),
        ("Status", status),
        ("Tumor Type", tumor_type),
        ("Confidence", f"{confidence}%"),
        ("Detected Regions", regions),
        ("Estimated Risk Level", estimated_risk_level),
        ("Estimated Tumor Region (%)", f"{estimated_tumor_region_percentage}%"),
    ]

    pdf_path = os.path.abspath("Brain_Tumor_Report.pdf")
    c = canvas.Canvas(pdf_path, pagesize=A4)

    width, height = A4

    # ===== TITLE =====
    c.setFillColorRGB(0.05, 0.15, 0.45)
    c.rect(0, height - 60, width, 60, fill=1)

    c.setFillColorRGB(1, 1, 1)
    c.setFont("Times-Bold", 22)
    c.drawCentredString(width / 2, height - 40, "BRAIN TUMOR DETECTION REPORT")
    c.setFillColorRGB(0, 0, 0)

    # ===== TABLE =====
    table_top = height - 100
    row_height = 28
    table_width = 400

    left_margin = (width - table_width) / 2
    right_margin = left_margin + table_width
    col_split = left_margin + table_width / 2

    c.setFont("Times-Roman", 12)

    y = table_top

    for label, value in rows:
        c.line(left_margin, y, right_margin, y)
        c.line(col_split, y, col_split, y - row_height)

        c.drawString(left_margin + 10, y - 18, label)
        c.drawString(col_split + 10, y - 18, str(value))

        y -= row_height

    c.line(left_margin, y, right_margin, y)
    c.line(left_margin, table_top, left_margin, y)
    c.line(right_margin, table_top, right_margin, y)

    # ===== IMAGES =====
    y -= 50

    c.setFont("Times-Bold", 14)
    c.drawCentredString(width / 2, y, "MRI Images")

    y -= 30

    img_w = 160
    img_h = 160

    total_width = (img_w * 3) + 40
    start_x = (width - total_width) / 2

    if original_image:
        path = os.path.abspath(os.path.join("static", original_image))
        if os.path.exists(path):
            c.drawImage(path, start_x, y - img_h, width=img_w, height=img_h)
            c.drawCentredString(start_x + img_w / 2, y - img_h - 15, "Original")

    if result_image:
        path = os.path.abspath(os.path.join("static", result_image))
        if os.path.exists(path):
            c.drawImage(path, start_x + img_w + 20, y - img_h, width=img_w, height=img_h)
            c.drawCentredString(start_x + img_w + 20 + img_w / 2, y - img_h - 15, "Detected")

    if gradcam_image:
        path = os.path.abspath(os.path.join("static", gradcam_image))
        if os.path.exists(path):
            c.drawImage(path, start_x + (img_w + 20) * 2, y - img_h, width=img_w, height=img_h)
            c.drawCentredString(start_x + (img_w + 20) * 2 + img_w / 2, y - img_h - 15, "Grad-CAM")

    # ===== DOCTOR DETAILS =====
    c.setFont("Times-Roman", 12)

    y_position = 150

    col1 = width * 1 / 6
    col2 = width * 3 / 6
    col3 = width * 5 / 6

    c.drawCentredString(col1, y_position, "Doctor Name")
    c.drawCentredString(col2, y_position, "Doctor ID")
    c.drawCentredString(col3, y_position, "Doctor Signature")

    # ===== DISCLAIMER =====
    c.setFont("Times-Italic", 10)
    c.drawCentredString(
        width / 2,
        35,
        "Disclaimer: This report is generated using an AI-based YOLOv8 + Grad-CAM system."
    )
    c.drawCentredString(
        width / 2,
        20,
        "For academic demonstration only. This is not a clinical diagnosis."
    )

    c.save()

    return send_file(pdf_path, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)