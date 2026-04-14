import cv2
import numpy as np
import os
import json
import argparse


def build_object_points(pattern_size, square_size):
    """
    pattern_size: (cols, rows) = 체커보드 '내부 코너 수'
    square_size: 실제 한 칸의 크기 (예: mm 단위)
    """
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def calibrate_from_video(
    video_path,
    pattern_size,
    square_size,
    output_dir,
    frame_step=10,
    max_frames=120,
    visualize_detection=True,
    save_detection_video=True
):
    """
    video_path: 입력 영상
    pattern_size: (cols, rows), 내부 코너 개수
    square_size: 체커보드 한 칸 실제 크기
    output_dir: 결과 저장 폴더
    frame_step: 몇 프레임마다 한 번씩 검사할지
    max_frames: 최대 몇 개의 성공 프레임을 사용할지
    """

    ensure_dir(output_dir)
    detection_img_dir = os.path.join(output_dir, "detected_frames")
    undistort_img_dir = os.path.join(output_dir, "undistorted_samples")
    ensure_dir(detection_img_dir)
    ensure_dir(undistort_img_dir)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"비디오를 열 수 없습니다: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detection_video_path = os.path.join(output_dir, "detection_preview.mp4")
    undistorted_video_path = os.path.join(output_dir, "undistorted_video.mp4")

    detection_writer = None
    if save_detection_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        detection_writer = cv2.VideoWriter(
            detection_video_path, fourcc, fps, (width, height)
        )

    objp = build_object_points(pattern_size, square_size)

    objpoints = []   # 3D points in world space
    imgpoints = []   # 2D points in image plane

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )

    frame_idx = 0
    success_count = 0
    image_size = None

    print("[INFO] 체커보드 검출 시작...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step != 0:
            if detection_writer is not None:
                detection_writer.write(frame)
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]  # (width, height)

        # OpenCV의 최신/기본 플래그
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK
        )

        found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

        vis = frame.copy()

        if found:
            corners2 = cv2.cornerSubPix(
                gray,
                corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=criteria
            )

            objpoints.append(objp.copy())
            imgpoints.append(corners2)
            success_count += 1

            cv2.drawChessboardCorners(vis, pattern_size, corners2, found)

            save_path = os.path.join(
                detection_img_dir, f"detected_{success_count:03d}.png"
            )
            cv2.imwrite(save_path, vis)

            print(f"[INFO] detected frame {frame_idx} -> total {success_count}")

        if visualize_detection:
            text = f"Detected: {success_count}"
            cv2.putText(
                vis, text, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA
            )

        if detection_writer is not None:
            detection_writer.write(vis)

        frame_idx += 1

        if success_count >= max_frames:
            break

    cap.release()
    if detection_writer is not None:
        detection_writer.release()

    if success_count < 5:
        raise RuntimeError(
            f"체커보드 검출 성공 프레임이 너무 적습니다: {success_count}개\n"
            f"조명, 보드 크기, pattern_size, frame_step을 확인하세요."
        )

    print("[INFO] calibrateCamera 실행 중...")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None
    )

    # RMSE 직접 계산
    total_error = 0.0
    total_points = 0

    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        error = cv2.norm(imgpoints[i], projected, cv2.NORM_L2)
        n = len(projected)
        total_error += error * error
        total_points += n

    rmse = np.sqrt(total_error / total_points)

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    dist_flat = dist_coeffs.flatten().tolist()

    result = {
        "video_path": video_path,
        "image_width": width,
        "image_height": height,
        "pattern_size_inner_corners": {
            "cols": pattern_size[0],
            "rows": pattern_size[1]
        },
        "square_size": square_size,
        "num_valid_frames": success_count,
        "rms_returned_by_opencv": float(ret),
        "rmse_reprojection": float(rmse),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_flat,
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy
    }

    result_json_path = os.path.join(output_dir, "calibration_result.json")
    with open(result_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print("[INFO] calibration_result.json 저장 완료")
    print(json.dumps(result, indent=4, ensure_ascii=False))

    # 왜곡 보정 샘플 이미지 저장
    print("[INFO] 왜곡 보정 샘플 저장 중...")
    cap = cv2.VideoCapture(video_path)
    sample_indices = []
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frame_count > 0:
        sample_indices = [
            int(total_frame_count * 0.1),
            int(total_frame_count * 0.5),
            int(total_frame_count * 0.9),
        ]
    else:
        sample_indices = [0, 30, 60]

    current_idx = 0
    saved_sample = 0

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (width, height), 1, (width, height)
    )

    while True:
        ret_read, frame = cap.read()
        if not ret_read:
            break

        if current_idx in sample_indices:
            undistorted = cv2.undistort(
                frame, camera_matrix, dist_coeffs, None, new_camera_matrix
            )

            x, y, w_roi, h_roi = roi
            if w_roi > 0 and h_roi > 0:
                undistorted_crop = undistorted[y:y+h_roi, x:x+w_roi]
            else:
                undistorted_crop = undistorted

            cv2.imwrite(
                os.path.join(undistort_img_dir, f"sample_{saved_sample+1}_original.png"),
                frame
            )
            cv2.imwrite(
                os.path.join(undistort_img_dir, f"sample_{saved_sample+1}_undistorted.png"),
                undistorted
            )
            cv2.imwrite(
                os.path.join(undistort_img_dir, f"sample_{saved_sample+1}_undistorted_crop.png"),
                undistorted_crop
            )

            # 비교 이미지
            h1, w1 = frame.shape[:2]
            und_resized = cv2.resize(undistorted, (w1, h1))
            comp = np.hstack([frame, und_resized])
            cv2.putText(
                comp, "Original", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA
            )
            cv2.putText(
                comp, "Undistorted", (w1 + 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA
            )
            cv2.imwrite(
                os.path.join(undistort_img_dir, f"sample_{saved_sample+1}_comparison.png"),
                comp
            )

            saved_sample += 1
            if saved_sample >= len(sample_indices):
                break

        current_idx += 1

    cap.release()

    # 전체 영상 왜곡 보정
    print("[INFO] 왜곡 보정 영상 생성 중...")
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(
        undistorted_video_path, fourcc, fps, (width, height)
    )

    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix,
        (width, height),
        cv2.CV_16SC2
    )

    while True:
        ret_read, frame = cap.read()
        if not ret_read:
            break

        undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)

        # ROI 밖을 잘라내지 않고 그대로 저장
        out_writer.write(undistorted)

    cap.release()
    out_writer.release()

    print("[INFO] 완료")
    print(f"[INFO] detection preview video: {detection_video_path}")
    print(f"[INFO] undistorted video:      {undistorted_video_path}")
    print(f"[INFO] result json:           {result_json_path}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Chessboard video based camera calibration and distortion correction"
    )
    parser.add_argument("--video", type=str, required=True, help="입력 동영상 파일 (.mp4, .avi)")
    parser.add_argument(
        "--cols", type=int, required=True,
        help="체커보드 내부 코너 수 (가로)"
    )
    parser.add_argument(
        "--rows", type=int, required=True,
        help="체커보드 내부 코너 수 (세로)"
    )
    parser.add_argument(
        "--square_size", type=float, default=1.0,
        help="체커보드 한 칸의 실제 크기 (예: mm 단위면 25.0)"
    )
    parser.add_argument(
        "--frame_step", type=int, default=10,
        help="몇 프레임마다 검출할지"
    )
    parser.add_argument(
        "--max_frames", type=int, default=80,
        help="최대 검출 성공 프레임 수"
    )
    parser.add_argument(
        "--output_dir", type=str, default="calibration_output",
        help="결과 저장 폴더"
    )

    args = parser.parse_args()

    pattern_size = (args.cols, args.rows)

    calibrate_from_video(
        video_path=args.video,
        pattern_size=pattern_size,
        square_size=args.square_size,
        output_dir=args.output_dir,
        frame_step=args.frame_step,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main()