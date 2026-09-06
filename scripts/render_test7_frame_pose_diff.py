#!/usr/bin/env python3
"""results_05.py: 유효 프레임 전체의 러너 크롭 및 PNG vs JPEG 관절 비교 이미지 생성기.

results_04 (또는 results_03)의 예측 JSON 및 원본 이미지를 읽어
러너 크롭 영역의 포즈 오버레이와 26개 관절별 오차 막대그래프를 생성하여 results_05 폴더에 저장합니다.
"""

import argparse
import json
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

# Halpe26 관절 명칭 정의 (0~25)
HALPE26_NAMES = [
    "코(Nose)", "왼눈(LEye)", "오른눈(REye)", "왼귀(LEar)", "오른귀(REar)",
    "왼어깨(LShoulder)", "오른어깨(RShoulder)", "왼팔꿈치(LElbow)", "오른팔꿈치(RElbow)",
    "왼손목(LWrist)", "오른손목(RWrist)", "왼골반(LHip)", "오른골반(RHip)",
    "왼무릎(LKnee)", "오른무릎(RKnee)", "왼발목(LAnkle)", "오른발목(RAnkle)",
    "머리(Head)", "목(Neck)", "엉덩이(Hip)",
    "왼엄지발(LBigToe)", "오른엄지발(RBigToe)", "왼새끼발(LSmallToe)", "오른새끼발(RSmallToe)",
    "왼발뒤꿈치(LHeel)", "오른발뒤꿈치(RHeel)"
]

# Halpe26 주요 스켈레톤 뼈대 연결선 정의
SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),      # 상체 / 팔
    (5, 11), (6, 12), (11, 12),                  # 몸통
    (11, 13), (13, 15), (12, 14), (14, 16),      # 다리
    (15, 20), (15, 22), (15, 24),                # 왼발
    (16, 21), (16, 23), (16, 25),                # 오른발
    (17, 18), (18, 19),                          # 척추/머리
]


def setup_korean_font():
    """matplotlib 한글 깨짐 방지를 위한 시스템 폰트 설정."""
    font_candidates = [
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/mnt/c/Windows/Fonts/malgun.ttf"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            try:
                font_manager.fontManager.addfont(str(font_path))
                font_name = font_manager.FontProperties(fname=str(font_path)).get_name()
                plt.rcParams["font.family"] = font_name
                plt.rcParams["axes.unicode_minus"] = False
                return font_name
            except Exception:
                continue
    return None


def resolve_json_paths(poc_dir: Path, target: str):
    """지정된 대상(results_04 또는 results_03)의 PNG/JPEG JSON 경로를 탐색."""
    base_diag = poc_dir / "diagnostics/test7_png_vs_jpeg_preliminary_29frames"
    target_dir = base_diag / target

    png_json = target_dir / "png_pose_predictions.json"
    jpeg_json = target_dir / "jpeg_pose_predictions.json"

    # 만약 타겟 디렉토리에 없으면 run/hpe_compare 디렉토리에서 fallback
    if not (png_json.exists() and jpeg_json.exists()):
        hpe_dir = poc_dir / ("run/hpe_compare_04" if "04" in target else "run/hpe_compare")
        alt_png = hpe_dir / "png/outputs/pose_predictions.json"
        alt_jpeg = hpe_dir / "jpeg/outputs/pose_predictions.json"
        if alt_png.exists() and alt_jpeg.exists():
            png_json, jpeg_json = alt_png, alt_jpeg

    return png_json, jpeg_json


def find_frame_image(poc_dir: Path, frame_id: str, frame_data: dict, target: str) -> Path | None:
    """프레임 원본 이미지 파일 경로 탐색."""
    candidates = []
    # 1. JSON에 기록된 image_path
    if "image_path" in frame_data:
        candidates.append(poc_dir / frame_data["image_path"])
        candidates.append(Path(frame_data["image_path"]))

    # 2. hpe_compare_04 / hpe_compare 입력 경로
    if "04" in target:
        candidates.append(poc_dir / f"run/hpe_compare_04/png/inputs/{frame_id}.png")
        candidates.append(poc_dir / f"run/hpe_compare/png/inputs/{frame_id}.png")
    else:
        candidates.append(poc_dir / f"run/hpe_compare/png/inputs/{frame_id}.png")
        candidates.append(poc_dir / f"run/hpe_compare_04/png/inputs/{frame_id}.png")

    # 3. speed_test 전체 프레임 경로
    candidates.append(poc_dir / f"run/speed_test/opencv_frames/{frame_id}.png")
    candidates.append(poc_dir / f"run/speed_test/opencv_jpg/{frame_id}.jpg")

    for path in candidates:
        if path.exists():
            return path
    return None


def select_primary_person(people_list: list) -> dict | None:
    """트래킹 대상(track_id=0) 우선 선택, 없으면 첫 번째 인물 반환."""
    if not people_list:
        return None
    for p in people_list:
        if p.get("track_id") == 0:
            return p
    return people_list[0]


def process_target(poc_dir: Path, target: str, output_dir: Path, diff_thresh: float = 2.0):
    """지정된 대상의 비교 시각화 실행."""
    png_json_path, jpeg_json_path = resolve_json_paths(poc_dir, target)
    if not png_json_path.exists() or not jpeg_json_path.exists():
        print(f"[{target}] JSON 파일을 찾을 수 없습니다: {png_json_path} / {jpeg_json_path}")
        return

    print(f"\n==========================================")
    print(f"[{target}] 시각화 작업 시작")
    print(f" - PNG JSON : {png_json_path}")
    print(f" - JPEG JSON: {jpeg_json_path}")
    print(f" - 출력 폴더: {output_dir}")
    print(f"==========================================")

    output_dir.mkdir(parents=True, exist_ok=True)

    png_records = json.loads(png_json_path.read_text()).get("frames", [])
    jpeg_records = json.loads(jpeg_json_path.read_text()).get("frames", [])

    png_dict = {Path(f["image_path"]).stem: f for f in png_records}
    jpeg_dict = {Path(f["image_path"]).stem: f for f in jpeg_records}

    common_frames = sorted(png_dict.keys() & jpeg_dict.keys())
    print(f"총 {len(common_frames)}개 공통 유효 프레임 발견: {common_frames}")

    saved_count = 0
    summary_stats = []

    for frame_id in common_frames:
        frame_png = png_dict[frame_id]
        frame_jpeg = jpeg_dict[frame_id]

        p_png = select_primary_person(frame_png.get("people", []))
        p_jpeg = select_primary_person(frame_jpeg.get("people", []))

        if p_png is None or p_jpeg is None:
            print(f"[프레임 {frame_id}] 인물 검출 데이터가 없어 건너뜁니다.")
            continue

        kpts_png = np.array(p_png["keypoints"])   # (26, 2)
        kpts_jpeg = np.array(p_jpeg["keypoints"]) # (26, 2)

        # 관절별 유클리드 거리 (오차)
        dists = np.linalg.norm(kpts_png - kpts_jpeg, axis=1)
        mean_dist = float(np.mean(dists))
        max_dist = float(np.max(dists))
        worst_kpt = int(np.argmax(dists))
        worst_kpt_name = HALPE26_NAMES[worst_kpt] if worst_kpt < len(HALPE26_NAMES) else f"Kpt {worst_kpt}"

        summary_stats.append({
            "frame_id": frame_id,
            "mean_dist_px": round(mean_dist, 4),
            "max_dist_px": round(max_dist, 4),
            "worst_kpt_index": worst_kpt,
            "worst_kpt_name": worst_kpt_name,
            "diff_over_threshold_count": int(np.sum(dists >= diff_thresh)),
            "per_keypoint_diffs": [round(float(d), 4) for d in dists]
        })

        # 원본 이미지 로드
        img_path = find_frame_image(poc_dir, frame_id, frame_png, target)
        if img_path is None:
            print(f"[프레임 {frame_id}] 원본 이미지를 찾을 수 없어 건너뜁니다.")
            continue

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[프레임 {frame_id}] 이미지 로드 실패: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_rgb.shape[:2]

        # 러너 및 모든 키포인트를 포함하는 넉넉한 크롭 영역(BBox) 계산
        box = p_png.get("bbox", [0, 0, img_w, img_h])
        all_xs = np.concatenate([kpts_png[:, 0], kpts_jpeg[:, 0], [box[0], box[2]]])
        all_ys = np.concatenate([kpts_png[:, 1], kpts_jpeg[:, 1], [box[1], box[3]]])

        pad_x, pad_y = 60, 60
        xmin = max(0, int(np.min(all_xs) - pad_x))
        ymin = max(0, int(np.min(all_ys) - pad_y))
        xmax = min(img_w, int(np.max(all_xs) + pad_x))
        ymax = min(img_h, int(np.max(all_ys) + pad_y))

        # 시각화 플롯 생성 (1행 2열: 좌측 오버레이, 우측 오차 막대그래프)
        fig, axes = plt.subplots(1, 2, figsize=(15, 7.5), gridspec_kw={"width_ratios": [1.25, 1]})

        # [좌측 서브플롯] 원본 배경 + 관절 비교 오버레이
        ax_img = axes[0]
        ax_img.imshow(img_rgb)

        # 1) PNG 스켈레톤 (밝은 하늘색 실선)
        for a, b in SKELETON_EDGES:
            if a < len(kpts_png) and b < len(kpts_png):
                ax_img.plot(kpts_png[[a, b], 0], kpts_png[[a, b], 1],
                            color="#00e5ff", linewidth=2.2, alpha=0.85, zorder=2)
        ax_img.scatter(kpts_png[:, 0], kpts_png[:, 1], c="#00e5ff",
                       s=36, edgecolors="black", linewidths=0.5, label="PNG 예측", zorder=3)

        # 2) JPEG 스켈레톤 (주황색 점선)
        for a, b in SKELETON_EDGES:
            if a < len(kpts_jpeg) and b < len(kpts_jpeg):
                ax_img.plot(kpts_jpeg[[a, b], 0], kpts_jpeg[[a, b], 1],
                            color="#ff9100", linewidth=1.8, linestyle="--", alpha=0.85, zorder=2)
        ax_img.scatter(kpts_jpeg[:, 0], kpts_jpeg[:, 1], c="#ff9100",
                       s=28, edgecolors="black", linewidths=0.5, label="JPEG (q95) 예측", zorder=3)

        # 3) 차이 강조선 (임계값 이상 오차 관절은 빨간색 실선 연결)
        diff_count = 0
        for i in range(len(dists)):
            if dists[i] >= diff_thresh:
                diff_count += 1
                ax_img.plot([kpts_png[i, 0], kpts_jpeg[i, 0]],
                            [kpts_png[i, 1], kpts_jpeg[i, 1]],
                            color="#ff1744", linewidth=2.5, zorder=4)

        ax_img.set_xlim(xmin, xmax)
        ax_img.set_ylim(ymax, ymin)  # 이미지 좌표계(y축 반전)
        ax_img.set_title(
            f"프레임 {frame_id} 오버레이\n(평균 오차: {mean_dist:.2f}px | 최대 오차: {max_dist:.2f}px [{worst_kpt_name}])",
            fontsize=12, pad=10
        )
        ax_img.axis("off")
        ax_img.legend(loc="upper right", framealpha=0.9, fontsize=9)

        # [우측 서브플롯] 26개 관절별 오차 막대그래프
        ax_bar = axes[1]
        y_pos = np.arange(len(dists))
        bar_colors = ["#ff5252" if d >= diff_thresh else "#42a5f5" for d in dists]
        bars = ax_bar.barh(y_pos, dists, color=bar_colors, height=0.7)
        ax_bar.set_yticks(y_pos)
        ytick_labels = [f"Kpt {i:02d} ({HALPE26_NAMES[i]})" for i in range(len(dists))]
        ax_bar.set_yticklabels(ytick_labels, fontsize=8)
        ax_bar.invert_yaxis()  # 0번 관절(코)이 위에 오도록
        ax_bar.set_xlabel("픽셀 좌표 오차 (px)", fontsize=10)
        ax_bar.set_title(f"26개 관절별 PNG vs JPEG 차이 (오차 ≥ {diff_thresh}px: {diff_count}개 관절)", fontsize=11, pad=10)
        ax_bar.grid(axis="x", linestyle="--", alpha=0.5)
        ax_bar.axvline(diff_thresh, color="#d32f2f", linestyle=":", linewidth=1.2, label=f"임계치 ({diff_thresh}px)")
        ax_bar.bar_label(bars, fmt="%.1f", padding=3, fontsize=7.5, color="#333333")
        ax_bar.legend(loc="lower right", fontsize=8)

        fig.tight_layout()
        save_path = output_dir / f"frame_{frame_id}_diff.png"
        fig.savefig(save_path, dpi=130)
        plt.close(fig)
        saved_count += 1
        print(f"[{target}] 저장 완료: {save_path.name} (평균: {mean_dist:.2f}px, 최대: {max_dist:.2f}px [{worst_kpt_name}])")

    # 통계 요약 JSON 저장
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({
        "target": target,
        "total_frames": saved_count,
        "diff_threshold_px": diff_thresh,
        "keypoint_names": HALPE26_NAMES,
        "frames": summary_stats
    }, indent=2, ensure_ascii=False))
    print(f"[{target}] 통계 요약 파일 저장: {summary_path.name}")

    print(f"[{target}] 총 {saved_count}/{len(common_frames)}장 생성 완료 -> {output_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="PNG vs JPEG Halpe26 관절 비교 오버레이 생성기")
    parser.add_argument(
        "--target",
        type=str,
        default="results_04",
        choices=["results_04", "results_03", "all"],
        help="비교 대상 지정 (기본값: results_04, results_03 또는 all 선택 가능)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="저장할 디렉토리 경로 (기본값: diagnostics/test7_png_vs_jpeg_255frames_validation)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="빨간색 강조선 표시 기준 오차 (픽셀 단위, 기본값: 2.0px)"
    )
    args = parser.parse_args()

    setup_korean_font()

    poc_dir = Path("/home/jwp/Oracle_Project2/Oracle_Project/PoC")
    base_diag = poc_dir / "diagnostics/test7_png_vs_jpeg_preliminary_29frames"

    targets = ["results_04", "results_03"] if args.target == "all" else [args.target]

    for t in targets:
        if args.output_dir is not None:
            out_dir = args.output_dir if len(targets) == 1 else args.output_dir / t
        else:
            # 기본 경로: results_05 디렉토리
            if t == "results_04":
                out_dir = poc_dir / "diagnostics/test7_png_vs_jpeg_255frames_validation"
            else:
                out_dir = base_diag / f"results_05_{t}"

        process_target(poc_dir, t, out_dir, diff_thresh=args.threshold)


if __name__ == "__main__":
    main()

def generate_top5_plot_standalone(poc_dir: Path, output_dir: Path):
    """결과 요약 데이터를 바탕으로 최대 픽셀 오차 발생 Top 5 프레임 플롯 자동 생성."""
    summary_path = output_dir / "full255_validation_summary.json"
    if not summary_path.exists():
        print(f"[{output_dir.name}] full255_validation_summary.json 파일이 없어 Top 5 플롯을 건너뜁니다.")
        return

    import subprocess
    script = poc_dir / "diagnostics/plot_test7_top5_worst_error_frames.py"
    if script.exists():
        subprocess.run([poc_dir / ".venv/bin/python", str(script)], check=True)
