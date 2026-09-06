#!/usr/bin/env python3
"""plot_test7_top5_worst_error_frames.py: test7 255장 검증 결과 중 최대 픽셀 오차 발생 Top 5 프레임 종합 시각화 플롯 생성."""

import json
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

POC_DIR = Path("/home/jwp/Oracle_Project2/Oracle_Project/PoC")
OUTPUT_DIR = POC_DIR / "diagnostics/test7_png_vs_jpeg_255frames_validation"
PNG_DIR = POC_DIR / "run/test7/inputs"
JPG_DIR = POC_DIR / "run/speed_test/opencv_jpg"

HALPE26_NAMES = [
    "코(Nose)", "왼눈(LEye)", "오른눈(REye)", "왼귀(LEar)", "오른귀(REar)",
    "왼어깨(LShoulder)", "오른어깨(RShoulder)", "왼팔꿈치(LElbow)", "오른팔꿈치(RElbow)",
    "왼손목(LWrist)", "오른손목(RWrist)", "왼골반(LHip)", "오른골반(RHip)",
    "왼무릎(LKnee)", "오른무릎(RKnee)", "왼발목(LAnkle)", "오른발목(RAnkle)",
    "머리(Head)", "목(Neck)", "엉덩이(Hip)",
    "왼엄지발(LBigToe)", "오른엄지발(RBigToe)", "왼새끼발(LSmallToe)", "오른새끼발(RSmallToe)",
    "왼발뒤꿈치(LHeel)", "오른발뒤꿈치(RHeel)"
]

SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (15, 20), (15, 22), (15, 24),
    (16, 21), (16, 23), (16, 25),
    (17, 18), (18, 19),
]

def setup_font():
    for p in [Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"), Path("/mnt/c/Windows/Fonts/malgun.ttf")]:
        if p.exists():
            font_manager.fontManager.addfont(str(p))
            name = font_manager.FontProperties(fname=str(p)).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return None

def main():
    setup_font()
    
    # 1. 모델 로드하여 Top 5 프레임 재추론 (완전한 키포인트 획득)
    from rtmlib import RTMDet, RTMPose
    detector = RTMDet(
        onnx_model=POC_DIR / "models/detectors/rtmdet-nano-person-320x320/end2end.onnx",
        model_input_size=(320, 320), det_mode="human", score_thr=0.4, nms_thr=0.45,
        backend="onnxruntime", device="cpu"
    )
    pose_model = RTMPose(
        onnx_model=POC_DIR / "models/pose/rtmpose-m-halpe26-384x288/end2end.onnx",
        model_input_size=(288, 384), backend="onnxruntime", device="cpu", to_openpose=False
    )

    summary_path = OUTPUT_DIR / "full255_validation_summary.json"
    summary_data = json.loads(summary_path.read_text())
    top5_frames_info = sorted(summary_data["frames"], key=lambda x: x["max_dist_px"], reverse=True)[:5]

    print("Top 5 최대 오차 프레임:", [f["frame"] for f in top5_frames_info])

    top5_data = []
    for rank, info in enumerate(top5_frames_info, start=1):
        f_id = info["frame"]
        im_png = cv2.imread(str(PNG_DIR / f"{f_id}.png"))
        
        jpg_p = JPG_DIR / f"{f_id}.jpg"
        if jpg_p.exists():
            im_jpg = cv2.imread(str(jpg_p))
        else:
            _, enc = cv2.imencode(".jpg", im_png, [cv2.IMWRITE_JPEG_QUALITY, 95])
            im_jpg = cv2.imdecode(enc, cv2.IMREAD_COLOR)

        b_png = detector(im_png)
        b_jpg = detector(im_jpg)
        k_png, _ = pose_model(im_png, bboxes=b_png[:1])
        k_jpg, _ = pose_model(im_jpg, bboxes=b_jpg[:1])

        kpts_p = np.array(k_png[0])
        kpts_j = np.array(k_jpg[0])
        dists = np.linalg.norm(kpts_p - kpts_j, axis=1)

        top5_data.append({
            "rank": rank,
            "frame_id": f_id,
            "im_rgb": cv2.cvtColor(im_png, cv2.COLOR_BGR2RGB),
            "kpts_png": kpts_p,
            "kpts_jpeg": kpts_j,
            "bbox": b_png[0][:4],
            "dists": dists,
            "mean_dist": float(np.mean(dists)),
            "max_dist": float(np.max(dists)),
            "worst_kpt": int(np.argmax(dists)),
            "worst_kpt_name": HALPE26_NAMES[int(np.argmax(dists))]
        })

    # ========================================================
    # 플롯 1: 5행 2열 상세 분석 대시보드 (오버레이 + 막대그래프)
    # ========================================================
    fig, axes = plt.subplots(5, 2, figsize=(16, 26), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.suptitle("test7 255장 전체 중 PNG vs JPEG 픽셀 오차 최대 발생 Top 5 프레임 정밀 분석", fontsize=18, y=0.995, fontweight="bold")

    for i, item in enumerate(top5_data):
        ax_img = axes[i, 0]
        ax_bar = axes[i, 1]

        im_rgb = item["im_rgb"]
        img_h, img_w = im_rgb.shape[:2]
        kp_p = item["kpts_png"]
        kp_j = item["kpts_jpeg"]
        box = item["bbox"]
        dists = item["dists"]

        all_xs = np.concatenate([kp_p[:, 0], kp_j[:, 0], [box[0], box[2]]])
        all_ys = np.concatenate([kp_p[:, 1], kp_j[:, 1], [box[1], box[3]]])
        pad_x, pad_y = 60, 60
        xmin = max(0, int(np.min(all_xs) - pad_x))
        ymin = max(0, int(np.min(all_ys) - pad_y))
        xmax = min(img_w, int(np.max(all_xs) + pad_x))
        ymax = min(img_h, int(np.max(all_ys) + pad_y))

        # 좌측 이미지
        ax_img.imshow(im_rgb)
        for a, b in SKELETON_EDGES:
            ax_img.plot(kp_p[[a, b], 0], kp_p[[a, b], 1], color="#00e5ff", linewidth=2.2, alpha=0.85, zorder=2)
            ax_img.plot(kp_j[[a, b], 0], kp_j[[a, b], 1], color="#ff9100", linewidth=1.8, linestyle="--", alpha=0.85, zorder=2)

        ax_img.scatter(kp_p[:, 0], kp_p[:, 1], c="#00e5ff", s=36, edgecolors="black", linewidths=0.5, label="PNG", zorder=3)
        ax_img.scatter(kp_j[:, 0], kp_j[:, 1], c="#ff9100", s=28, edgecolors="black", linewidths=0.5, label="JPEG (q95)", zorder=3)

        for k in range(len(dists)):
            if dists[k] >= 2.0:
                ax_img.plot([kp_p[k, 0], kp_j[k, 0]], [kp_p[k, 1], kp_j[k, 1]], color="#ff1744", linewidth=2.5, zorder=4)

        # 이상치 관절 텍스트 강조
        w_k = item["worst_kpt"]
        ax_img.annotate(
            f"{item['worst_kpt_name']}\n차이: {item['max_dist']:.1f}px",
            xy=(kp_j[w_k, 0], kp_j[w_k, 1]),
            xytext=(15, -15), textcoords="offset points",
            color="yellow", fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.75, edgecolor="red"),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="yellow", lw=1.5)
        )

        ax_img.set_xlim(xmin, xmax)
        ax_img.set_ylim(ymax, ymin)
        ax_img.set_title(f"[Rank {item['rank']}] 프레임 {item['frame_id']} | 최대 오차: {item['max_dist']:.2f}px [{item['worst_kpt_name']}] | 평균: {item['mean_dist']:.2f}px", fontsize=11, pad=8)
        ax_img.axis("off")
        ax_img.legend(loc="upper right", fontsize=8, framealpha=0.85)

        # 우측 막대그래프
        y_pos = np.arange(26)
        bar_colors = ["#ff5252" if d >= 2.0 else "#42a5f5" for d in dists]
        bars = ax_bar.barh(y_pos, dists, color=bar_colors, height=0.7)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels([f"Kpt {idx:02d} ({HALPE26_NAMES[idx]})" for idx in range(26)], fontsize=7.5)
        ax_bar.invert_yaxis()
        ax_bar.set_xlabel("픽셀 좌표 오차 (px)", fontsize=9)
        ax_bar.set_title(f"관절별 오차 분포 (2px 이상: {int(np.sum(dists >= 2.0))}개 관절)", fontsize=10, pad=8)
        ax_bar.grid(axis="x", linestyle="--", alpha=0.5)
        ax_bar.axvline(2.0, color="#d32f2f", linestyle=":", linewidth=1.2)
        ax_bar.bar_label(bars, fmt="%.1f", padding=3, fontsize=7, color="#333333")

    fig.tight_layout(rect=[0, 0, 1, 0.99])
    out_path1 = OUTPUT_DIR / "top5_worst_discrepancy_frames.png"
    fig.savefig(out_path1, dpi=130)
    plt.close(fig)
    print(f"Top 5 상세 분석 대시보드 저장 완료: {out_path1.name}")

    # ========================================================
    # 플롯 2: 1행 5열 가로 비교 스트립 카드 뷰 (Top 5 빠른 훑어보기용)
    # ========================================================
    fig_strip, axes_strip = plt.subplots(1, 5, figsize=(22, 6.5))
    fig_strip.suptitle("test7 픽셀 차이 최대 발생 Top 5 프레임 러너 크롭 & 포즈 비교 스트립", fontsize=15, y=0.98, fontweight="bold")

    for i, item in enumerate(top5_data):
        ax = axes_strip[i]
        im_rgb = item["im_rgb"]
        img_h, img_w = im_rgb.shape[:2]
        kp_p = item["kpts_png"]
        kp_j = item["kpts_jpeg"]
        box = item["bbox"]
        dists = item["dists"]

        all_xs = np.concatenate([kp_p[:, 0], kp_j[:, 0], [box[0], box[2]]])
        all_ys = np.concatenate([kp_p[:, 1], kp_j[:, 1], [box[1], box[3]]])
        pad_x, pad_y = 60, 60
        xmin = max(0, int(np.min(all_xs) - pad_x))
        ymin = max(0, int(np.min(all_ys) - pad_y))
        xmax = min(img_w, int(np.max(all_xs) + pad_x))
        ymax = min(img_h, int(np.max(all_ys) + pad_y))

        ax.imshow(im_rgb)
        for a, b in SKELETON_EDGES:
            ax.plot(kp_p[[a, b], 0], kp_p[[a, b], 1], color="#00e5ff", linewidth=2.2, alpha=0.85, zorder=2)
            ax.plot(kp_j[[a, b], 0], kp_j[[a, b], 1], color="#ff9100", linewidth=1.8, linestyle="--", alpha=0.85, zorder=2)

        ax.scatter(kp_p[:, 0], kp_p[:, 1], c="#00e5ff", s=32, edgecolors="black", linewidths=0.5, label="PNG", zorder=3)
        ax.scatter(kp_j[:, 0], kp_j[:, 1], c="#ff9100", s=24, edgecolors="black", linewidths=0.5, label="JPEG (q95)", zorder=3)

        for k in range(len(dists)):
            if dists[k] >= 2.0:
                ax.plot([kp_p[k, 0], kp_j[k, 0]], [kp_p[k, 1], kp_j[k, 1]], color="#ff1744", linewidth=2.5, zorder=4)

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymax, ymin)
        ax.set_title(f"[Rank {item['rank']}] Frame {item['frame_id']}\n최대: {item['max_dist']:.1f}px ({item['worst_kpt_name']})\n평균: {item['mean_dist']:.1f}px", fontsize=10, pad=6)
        ax.axis("off")
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

    fig_strip.tight_layout(rect=[0, 0, 1, 0.96])
    out_path2 = OUTPUT_DIR / "top5_worst_frames_comparison_strip.png"
    fig_strip.savefig(out_path2, dpi=130)
    plt.close(fig_strip)
    print(f"Top 5 가로 비교 스트립 저장 완료: {out_path2.name}")

if __name__ == "__main__":
    main()
