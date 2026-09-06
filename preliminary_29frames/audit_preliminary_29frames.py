"""Read-only audit of saved PNG/JPEG HPE outputs; writes only to --out.

Replays the current detector/filter (without changing the production pipeline),
checks image pairing, and plots the saved predictions rather than invented data.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from rtmlib import RTMDet, RTMPose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.source, args.out
    out.mkdir(parents=True, exist_ok=True)
    font_path = Path('/mnt/c/Windows/Fonts/malgun.ttf')
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
        plt.rcParams['font.family'] = 'Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] = False
    data, inputs = {}, {}
    for fmt in ("png", "jpeg"):
        folder = root / "run/hpe_compare" / fmt
        records = json.loads((folder / "outputs/pose_predictions.json").read_text())["frames"]
        data[fmt] = {Path(f["image_path"]).stem: f for f in records}
        inputs[fmt] = {p.stem: p for p in sorted((folder / "inputs").iterdir())
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg")}
    common = sorted(data["png"].keys() & data["jpeg"].keys())
    def primary(frame):
        return next((p for p in frame["people"] if p.get("track_id") == 0), frame["people"][0])
    rows = []
    for name in common:
        if not data["png"][name]["people"] or not data["jpeg"][name]["people"]:
            continue
        a, b = primary(data["png"][name]), primary(data["jpeg"][name])
        dist = np.linalg.norm(np.array(a["keypoints"]) - np.array(b["keypoints"]), axis=1)
        for k, d in enumerate(dist):
            rows.append(dict(frame=name, keypoint=k, distance_px=float(d),
                             png_conf=a["keypoint_scores"][k], jpeg_conf=b["keypoint_scores"][k],
                             png_xy=a["keypoints"][k], jpeg_xy=b["keypoints"][k]))
    worst = max(rows, key=lambda r: r["distance_px"])
    print("SAVED OUTPUT AUDIT", json.dumps({"input_counts": {k: len(v) for k,v in inputs.items()},
        "json_counts": {k:len(v) for k,v in data.items()}, "common": len(common), "worst": worst}), flush=True)
    with (out / "keypoint_differences.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    # Validate same decoded source frame and controlled JPEG quality, not just filenames.
    image_checks = []
    for name in sorted(inputs["png"].keys() & inputs["jpeg"].keys()):
        a, b = (cv2.imread(str(inputs[fmt][name])) for fmt in ("png", "jpeg"))
        if a.shape != b.shape:
            raise ValueError(f"Image shape mismatch at {name}")
        _, encoded = cv2.imencode(".jpg", a, [cv2.IMWRITE_JPEG_QUALITY, 95])
        regenerated = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        image_checks.append(dict(frame=name, width=a.shape[1], height=a.shape[0],
            decoded_mae=float(np.abs(a.astype(float)-b).mean()),
            q95_reencoded_decoded_exact=bool(np.array_equal(regenerated, b))))
    print("PAIRING CHECK", sum(x["q95_reencoded_decoded_exact"] for x in image_checks),
          "/", len(image_checks), "exact decoded q95 reconstructions", flush=True)

    detector = RTMDet(onnx_model=root / "models/detectors/rtmdet-nano-person-320x320/end2end.onnx",
        model_input_size=(320,320), det_mode="human", score_thr=0.3, nms_thr=0.45,
        backend="onnxruntime", device="cpu")
    traces, retained, all_boxes = {}, {}, {}
    for fmt in ("png", "jpeg"):
        metadata = json.loads((root / "run/hpe_compare" / fmt / "outputs/details.json").read_text())
        width = metadata["video"]["width"]
        trace, kept, stopped = [], [], False
        all_boxes[fmt] = {}
        for name, path in inputs[fmt].items():
            im = cv2.imread(str(path))
            boxes = np.asarray(detector(im))
            all_boxes[fmt][name] = boxes
            outside = bool(len(boxes) and (boxes[0][0] < 10 or boxes[0][2] > width - 10))
            row = dict(frame=name, detection_count=len(boxes), actual_width=im.shape[1],
                       configured_width=width, first_bbox=boxes[0].tolist() if len(boxes) else None,
                       outside_range=outside, cleared=[])
            if stopped:
                row["action"] = "not_reached_after_break"
            elif not len(boxes):
                row["action"] = "skip_no_detection"
            elif outside:
                if len(kept) < 20:
                    row["action"] = "clear_previous_and_skip"
                    row["cleared"] = kept.copy()
                    kept.clear()
                else:
                    row["action"] = "break"
                    stopped = True
            else:
                row["action"] = "append"
                kept.append(name)
            trace.append(row)
        traces[fmt], retained[fmt] = trace, kept
        print(fmt, "FILTER REPLAY", len(kept), "saved-output match:", kept == list(data[fmt]), flush=True)

    # Repeat the largest-discrepancy frame with shared and independent boxes.
    # A fixed PNG bbox isolates pose sensitivity from detector crop changes.
    pose = RTMPose(onnx_model=root / "models/pose/rtmpose-m-halpe26-384x288/end2end.onnx",
        model_input_size=(288,384), backend="onnxruntime", device="cpu", to_openpose=False)
    name = worst["frame"]
    images = {fmt: cv2.imread(str(inputs[fmt][name])) for fmt in ("png", "jpeg")}
    people = {fmt: primary(data[fmt][name]) for fmt in ("png", "jpeg")}
    predictions = {}
    for label, fmt, boxfmt in (("png_own", "png", "png"), ("png_repeat", "png", "png"),
                              ("jpeg_own", "jpeg", "jpeg"), ("jpeg_fixed_png_box", "jpeg", "png")):
        k, s = pose(images[fmt], bboxes=np.asarray([people[boxfmt]["bbox"]], dtype=np.float32))
        predictions[label] = {"keypoints": k[0].tolist(), "scores": s[0].tolist()}
    repeat = {}
    for label in ("png_repeat", "jpeg_own", "jpeg_fixed_png_box"):
        d = np.linalg.norm(np.array(predictions[label]["keypoints"]) - predictions["png_own"]["keypoints"], axis=1)
        repeat[label] = {"mean_px": float(d.mean()), "max_px": float(d.max()),
                         "outlier_keypoint_px": float(d[worst["keypoint"]])}
    for fmt in ("png", "jpeg"):
        d = np.linalg.norm(np.array(predictions[fmt+"_own"]["keypoints"]) - people[fmt]["keypoints"], axis=1)
        repeat[fmt+"_saved_reproduction_max_px"] = float(d.max())

    # Three-panel scientific overlay: original pixels, saved predictions, one shared crop.
    points = {fmt: np.asarray(p["keypoints"]) for fmt,p in people.items()}
    k = worst["keypoint"]
    colors = {"png":"#00d7ff", "jpeg":"#ff9300"}
    fig, axes = plt.subplots(1,3,figsize=(15,6))
    box = people["png"]["bbox"]
    xmin, ymin = max(0, box[0]-70), max(0, box[1]-60)
    # Include out-of-bbox predictions: the outlier itself is outside the runner box.
    xmax = min(images["png"].shape[1], max(box[2], *(p[:,0].max() for p in points.values())) + 130)
    ymax = min(images["png"].shape[0], max(box[3], *(p[:,1].max() for p in points.values())) + 60)
    edges = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
    for ax,fmt in zip(axes[:2], ("png","jpeg")):
        ax.imshow(cv2.cvtColor(images[fmt], cv2.COLOR_BGR2RGB))
        p = points[fmt]
        for a,b in edges:
            ax.plot(p[[a,b],0],p[[a,b],1],color=colors[fmt],linewidth=1)
        ax.scatter(p[:,0],p[:,1],s=12,c=colors[fmt])
        ax.scatter(*p[k],s=180,facecolors="none",edgecolors="red",linewidths=2)
        ax.annotate(f"왼쪽 귀 추정점\n신뢰도 {people[fmt]['keypoint_scores'][k]:.4f}",p[k],
                    xytext=(8,-25),textcoords="offset points",color="red",fontsize=10,
                    bbox=dict(facecolor="white",alpha=.85,edgecolor="none"))
        ax.set_title(fmt.upper()+" 이미지와 HPE 좌표")
    ax=axes[2]; ax.imshow(cv2.cvtColor(images["png"],cv2.COLOR_BGR2RGB))
    for fmt in ("png","jpeg"):
        ax.scatter(*points[fmt][k],s=70,c=colors[fmt],label=fmt.upper())
    ax.plot([points["png"][k,0],points["jpeg"][k,0]],
            [points["png"][k,1],points["jpeg"][k,1]],"r--",linewidth=2)
    ax.legend(); ax.set_title(f"한 배경에 겹친 두 추정점 | {worst['distance_px']:.3f}px 차이")
    for ax in axes:
        ax.set_xlim(xmin,xmax); ax.set_ylim(ymax,ymin); ax.axis("off")
    fig.suptitle(f"235번 프레임 · 왼쪽 귀(인덱스 {k}) | 두 점 모두 모델 예측이며 정답 좌표가 아님")
    fig.tight_layout(); fig.savefig(out/"outlier_overlay.png",dpi=150); plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(15,6))
    for ax,n in zip(axes,("00000226","00000235","00000244")):
        im=cv2.imread(str(inputs["png"][n])); ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        for idx,p in enumerate(data["png"][n]["people"]):
            x1,y1,x2,y2=p["bbox"]
            selected=p.get("track_id")==0
            color="#00aaff" if selected else "#ff9300"
            ax.add_patch(plt.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor=color,linewidth=2))
            text=f"검출 {idx}"+(" / 추적 대상" if selected else "")
            ax.text(max(x1,4),y2+18,text,color=color,fontsize=9,
                    bbox=dict(facecolor="white",alpha=.8,edgecolor="none"))
        ax.set_xlim(0,750);ax.set_ylim(1000,170);ax.axis("off")
        ax.set_title({"00000226":"226번: 러너와 유리 반사상", "00000235":"235번: 러너가 화면 밖으로 나감", "00000244":"244번: 반사상이 추적 대상으로 선택됨"}[n])
    fig.suptitle("PNG 결과의 검출·추적 상황 | 파랑: 비교에 사용된 track_id=0 | 주황: 다른 검출")
    fig.tight_layout(rect=(0,0,1,.91));fig.savefig(out/"tracking_context.png",dpi=150);plt.close(fig)

    # Contact sheet makes the actual frame retention sequence visible.
    fig,axes=plt.subplots(5,6,figsize=(18,10))
    for ax in axes.flat: ax.axis("off")
    for ax,row in zip(axes.flat,traces["png"]):
        im=cv2.imread(str(inputs["png"][row["frame"]]))
        ax.imshow(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
        h,w=im.shape[:2]
        ax.axvline(10,color="red",linewidth=.5); ax.axvline(w-10,color="red",linewidth=.5)
        if row["first_bbox"]:
            x1,y1,x2,y2=row["first_bbox"][:4]
            ax.add_patch(plt.Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor="orange",linewidth=1))
        kept=row["frame"] in data["png"]
        title="저장됨" if kept else row["action"].replace("clear_previous_and_skip","가장자리 필터로 제외").replace("skip_no_detection","사람 검출 없음")
        if not kept and row["action"]=="append": title="APPENDED, LATER CLEARED"
        ax.set_title(row["frame"]+" | "+title,fontsize=8,color="green" if kept else "red")
    fig.suptitle("균등 추출한 29장 | 초록: JSON 저장 | 빨강: 제외 | 주황 박스: 첫 번째 검출")
    fig.tight_layout(); fig.savefig(out/"frame_filter_contact_sheet.png",dpi=140); plt.close(fig)

    fig,ax=plt.subplots(figsize=(10,3.5))
    for i,name in enumerate(common):
        values=[r["distance_px"] for r in rows if r["frame"]==name]
        ax.scatter([i]*len(values), values, s=12, alpha=.65,color="#176b9b")
    ax.set_xticks(range(len(common)),[str(int(n)) for n in common]); ax.set_xlabel("Frame number")
    ax.set_ylabel("PNG vs JPEG keypoint distance (px)")
    ax.set_title("All 26 keypoints per matched frame; low-confidence points included")
    ax.grid(alpha=.2); fig.tight_layout();fig.savefig(out/"keypoint_error_plot.png",dpi=150);plt.close(fig)
    high=[r for r in rows if r["png_conf"]>=.5 and r["jpeg_conf"]>=.5]
    result={"source":str(root),"opencv_version":cv2.__version__,
        "model_code_sha256":hashlib.sha256((root/"scripts/hpe/hpe_model.py").read_bytes()).hexdigest(),
        "input_frames":{k:list(v) for k,v in inputs.items()},"saved_frames":{k:list(v) for k,v in data.items()},
        "only_png_saved":sorted(data["png"].keys()-data["jpeg"].keys()),
        "only_jpeg_saved":sorted(data["jpeg"].keys()-data["png"].keys()),
        "worst":worst,"top10":sorted(rows,key=lambda r:r["distance_px"],reverse=True)[:10],
        "image_pairing_checks":image_checks,"filter_trace":traces,"replayed_retained":retained,
        "outlier_saved_people":people,"repeat_tests":repeat,"repeat_predictions":predictions,
        "tracking_context":{n:data["png"][n] for n in ("00000226","00000235","00000244")},
        "confidence_ge_0_5_both":{"count":len(high),"total":len(rows),
            "mean_px":float(np.mean([r["distance_px"] for r in high])),
            "max_px":max(r["distance_px"] for r in high)},
        "caution":"No ground truth: these are prediction differences, not accuracy errors. Sparse samples cannot validate cadence or stance features."}
    (out/"audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print("FINAL",json.dumps({"repeat_tests":repeat,"high_confidence":result["confidence_ge_0_5_both"],
                              "output":str(out)}),flush=True)


if __name__=="__main__":
    main()
