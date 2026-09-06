#!/usr/bin/env python3
"""plot_test7_keypoint_error_scatter.py: test7 255장 전체 매칭 프레임에 대한 26개 관절 오차 산점도 플롯 생성.

test7_png_vs_jpeg_preliminary_29frames/results_03/keypoint_error_plot.png의 스타일을
test7 255장 전체 전수 검증 데이터(111개 유효 프레임 × 26개 관절 = 2,886개 키포인트)로 완벽 확장 구현합니다.
"""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

POC_DIR = Path("/home/jwp/Oracle_Project2/Oracle_Project/PoC")
OUTPUT_DIR = POC_DIR / "diagnostics/test7_png_vs_jpeg_255frames_validation"
SUMMARY_FILE = OUTPUT_DIR / "full255_validation_summary.json"

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
    summary = json.loads(SUMMARY_FILE.read_text())
    frames_data = summary["frames"]

    print(f"총 {len(frames_data)}개 유효 프레임 산점도 데이터 로드 완료")

    # =========================================================================
    # 버전 1: 원본 keypoint_error_plot.png와 100% 동일한 학술 스타일 (순서형 인덱스)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(15, 4.5))

    frame_numbers = [int(f["frame"]) for f in frames_data]
    
    for i, f in enumerate(frames_data):
        dists = f["dists"]
        # 각 프레임 위치에 26개 키포인트 점을 찍음
        ax.scatter([i] * len(dists), dists, s=14, alpha=0.65, color="#176b9b", edgecolors="none")

    # 10프레임 간격으로 x축 라벨링
    tick_step = 10
    tick_indices = list(range(0, len(frames_data), tick_step))
    if (len(frames_data) - 1) not in tick_indices:
        tick_indices.append(len(frames_data) - 1)
    
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([str(frame_numbers[idx]) for idx in tick_indices], fontsize=10)
    ax.set_xlabel("Frame number", fontsize=11)
    ax.set_ylabel("PNG vs JPEG keypoint distance (px)", fontsize=11)
    ax.set_title("All 26 keypoints per matched frame; low-confidence points included (test7 255 frames)", fontsize=12, pad=10)
    ax.grid(True, alpha=0.25, linestyle="-")
    ax.set_ylim(-5, max(f["max_dist_px"] for f in frames_data) + 15)

    fig.tight_layout()
    out_file1 = OUTPUT_DIR / "keypoint_error_plot.png"
    fig.savefig(out_file1, dpi=160)
    plt.close(fig)
    print(f"1. 순서형 산점도 저장 완료: {out_file1}")

    # =========================================================================
    # 버전 2: 전체 255 타임라인 기준 산점도 (미검출 구간 포함 타임라인 시각화)
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(15, 4.8))

    for f in frames_data:
        frame_num = int(f["frame"])
        dists = f["dists"]
        ax2.scatter([frame_num] * len(dists), dists, s=14, alpha=0.6, color="#176b9b", edgecolors="none")

    ax2.axhline(2.0, color="#e53935", linestyle=":", linewidth=1.2, label="허용 기준선 (2.0px)")
    ax2.set_xlim(1, 255)
    ax2.set_xlabel("프레임 번호 (전체 1~255)", fontsize=11)
    ax2.set_ylabel("키포인트 좌표 거리 오차 (px)", fontsize=11)
    ax2.set_title("test7 255장 전체 구간 키포인트 오차 산점도 (26개 관절 전수 표시)", fontsize=12, pad=10)
    ax2.grid(True, alpha=0.25, linestyle="--")
    ax2.legend(loc="upper right", fontsize=10)

    fig2.tight_layout()
    out_file2 = OUTPUT_DIR / "keypoint_error_plot_timeline255.png"
    fig2.savefig(out_file2, dpi=160)
    plt.close(fig2)
    print(f"2. 전체 255 타임라인 산점도 저장 완료: {out_file2}")

if __name__ == "__main__":
    main()
