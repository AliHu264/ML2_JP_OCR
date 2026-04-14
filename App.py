import gradio as gr
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from manga_ocr import MangaOcr
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util

# ── Load models once at startup ────────────────────────────────────────────
print("Loading models...")
yolo       = YOLO("runs/detect/manga_runs/mangaseer_detector2/weights/best.pt")
mocr       = MangaOcr()
embedder   = SentenceTransformer("LaBSE")
translator = GoogleTranslator(source="ja", target="en")
print("Models ready.")

# ── Class config ────────────────────────────────────────────────────────────
OCR_CLASSES = {"clean_text", "messy_text"}  # text_bubble is skipped

COLORS = {
    "clean_text"  : (0,   200,   0),   # green  (RGB)
    "messy_text"  : (255, 140,   0),   # orange (RGB)
    "text_bubble" : (180,   0, 255),   # purple (RGB)
}

# ── Bubble grouping helpers ─────────────────────────────────────────────────
def find_parent_bubble(text_box, bubble_boxes):
    tx1, ty1, tx2, ty2 = text_box
    for i, (bx1, by1, bx2, by2) in enumerate(bubble_boxes):
        if tx1 >= bx1 and ty1 >= by1 and tx2 <= bx2 and ty2 <= by2:
            return i
    return None

def group_by_bubble(detections, bubble_boxes):
    bubble_groups = {}
    ungrouped     = []
    for d in detections:
        parent = find_parent_bubble(d["bbox"], bubble_boxes)
        if parent is not None:
            bubble_groups.setdefault(parent, []).append(d)
        else:
            ungrouped.append(d)

    grouped = []
    for bubble_idx, group in bubble_groups.items():
        group.sort(key=lambda d: d["bbox"][1])  # top-to-bottom
        combined  = "\u3000".join(d["text"] for d in group)
        avg_conf  = round(sum(d["confidence"] for d in group) / len(group), 2)
        grouped.append({
            "text"      : combined,
            "bbox"      : bubble_boxes[bubble_idx],
            "confidence": avg_conf,
            "class"     : group[0]["class"],
        })
    return grouped + ungrouped

# ── Shared: translate + similarity scoring ──────────────────────────────────
def score_detections(detections):
    """Translate each detection and compute embedding similarity. Returns table rows."""
    table_rows = []
    for d in detections:
        japanese = d["text"]
        if not japanese.strip():
            continue
        english    = translator.translate(japanese)
        emb_ja     = embedder.encode(japanese, convert_to_tensor=True)
        emb_en     = embedder.encode(english,  convert_to_tensor=True)
        similarity = round(util.cos_sim(emb_ja, emb_en).item(), 4)
        quality    = "🟢" if similarity >= 0.85 else "🟡" if similarity >= 0.70 else "🔴"
        table_rows.append([
            d.get("class", "ocr_only"), japanese, english,
            f"{d.get('confidence', '—')}", f"{similarity:.4f}", quality
        ])
    return table_rows

# ── Mode A: Full pipeline (YOLO → crop → OCR) ──────────────────────────────
def run_full_pipeline(pil_image, conf_threshold):
    results      = yolo.predict(source=pil_image, conf=conf_threshold, verbose=False)
    img_cv       = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    bubble_boxes = []
    detections   = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = [int(c) for c in box.xyxy[0].tolist()]
            conf            = round(box.conf.item(), 2)
            class_name      = result.names[int(box.cls)]
            color_rgb       = COLORS.get(class_name, (255, 255, 255))
            color_bgr       = color_rgb[::-1]

            cv2.rectangle(img_cv, (x1, y1), (x2, y2), color_bgr, 2)
            label = f"{class_name} {conf}"
            cv2.rectangle(img_cv, (x1, y1 - 18), (x1 + len(label) * 7, y1), color_bgr, -1)
            cv2.putText(img_cv, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            if class_name == "text_bubble":
                bubble_boxes.append([x1, y1, x2, y2])
                continue

            if class_name not in OCR_CLASSES:
                continue

            cropped = pil_image.crop((x1, y1, x2, y2))
            text    = mocr(cropped)
            if not text.strip():
                continue

            detections.append({
                "text"      : text,
                "bbox"      : [x1, y1, x2, y2],
                "confidence": conf,
                "class"     : class_name,
            })

    grouped    = group_by_bubble(detections, bubble_boxes)
    table_rows = score_detections(grouped)
    annotated  = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    return annotated, table_rows

# ── Mode B: OCR-only (no YOLO, full image straight to MangaOCR) ────────────
def run_ocr_only(pil_image):
    """
    Pass the entire image (or a pasted crop) directly to MangaOCR.
    """
    raw_text = mocr(pil_image)

    # Split on Japanese full-stop / reading-order markers to get segments.
    # Keep the delimiter attached to the preceding segment (like splitlines).
    import re
    segments = re.split(r'(?<=[。！？\n])', raw_text)
    segments = [s.strip() for s in segments if s.strip()]

    # If MangaOCR returned nothing at all, surface a clear message.
    if not segments:
        return np.array(pil_image), [["ocr_only", "(no text detected)", "", "—", "—", "—"]]

    detections = [{"text": seg, "class": "ocr_only", "confidence": "—"} for seg in segments]
    table_rows = score_detections(detections)

    # Return the original image unchanged (no bounding boxes in this mode)
    return np.array(pil_image), table_rows

# ── Dispatcher ──────────────────────────────────────────────────────────────
def run_pipeline(pil_image, conf_threshold, mode):
    if pil_image is None:
        return None, []
    if mode == "OCR Only (no detection)":
        return run_ocr_only(pil_image)
    return run_full_pipeline(pil_image, conf_threshold)

# ── Gradio UI ────────────────────────────────────────────────────────────────
css = """
.results-table {
    height: 500px !important;
    overflow-y: auto !important;
}
.results-table table {
    table-layout: fixed;
    width: 100%;
}
.results-table td, .results-table th {
    overflow-x: auto;
    overflow-y: hidden;
    white-space: nowrap;
    max-width: 200px;
    display: table-cell;
}
.results-table td:nth-child(3),
.results-table td:nth-child(4) {
    max-width: 300px;
}
"""

with gr.Blocks(title="Manga Skimmer", theme=gr.themes.Soft(), css=css) as app:

    gr.Markdown("# 🈳 Manga Text Detection & Translation")
    gr.Markdown("Upload or paste a manga page to detect, OCR, translate, and similarity score.")

    with gr.Row():
        # ── Left column: image + controls ──────────────────────────────────
        with gr.Column(scale=1):
            image_panel = gr.Image(
                type="pil",
                label="Manga Page",
                sources=["upload", "clipboard"],
                elem_classes=["image-upload-container"],
            )

            mode_radio = gr.Radio(
                choices=["Full Pipeline (YOLO + OCR)", "OCR Only (no detection)"],
                value="Full Pipeline (YOLO + OCR)",
                label="Processing Mode",
            )

            conf_slider = gr.Slider(
                0.1, 0.9, value=0.4, step=0.05,
                label="Detection Confidence Threshold",
                visible=True,           # hidden in OCR-only mode
            )

            run_btn = gr.Button("▶ Run Pipeline", variant="primary")

            gr.Markdown(
                "**OCR Only mode** feeds the whole image directly to MangaOCR — "
                "no bounding boxes are drawn. Useful when YOLO misses regions or "
                "you paste a pre-cropped speech bubble."
            )

        # ── Right column: results table ────────────────────────────────────
        with gr.Column(scale=1):
            results_table = gr.Dataframe(
                headers=["Class", "Japanese", "English", "Conf", "Similarity", "Quality"],
                label="Results",
                wrap=True,
                elem_classes=["results-table"],
            )
            gr.Markdown(
                "**Quality:** 🟢 ≥ 0.85 (high) · 🟡 0.70-0.84 (medium) · 🔴 < 0.70 (low / SFX)"
            )

    # Hide confidence slider when OCR-only is selected
    mode_radio.change(
        fn=lambda m: gr.update(visible=(m == "Full Pipeline (YOLO + OCR)")),
        inputs=mode_radio,
        outputs=conf_slider,
    )

    run_btn.click(
        fn      = run_pipeline,
        inputs  = [image_panel, conf_slider, mode_radio],
        outputs = [image_panel, results_table],
    )

if __name__ == "__main__":
    app.launch(share=False)