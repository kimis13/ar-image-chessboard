import numpy as np
import cv2 as cv
import json

# =========================================================
# 1. 입력 데이터 / 캘리브레이션 파라미터
# =========================================================
video_file = "./result/undistorted_video.mp4"
ar_image_file = "./ar_image.png"   # 덧씌울 이미지 (PNG/JPG)
save_file = "./data/chessboard_ar_output.mp4"
json_file = "./result/calibration_result.json"

def load_param(json_file, use_zero_distortion=False):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        
        K = np.array(data["camera_matrix"], dtype=np.float32)
        
        raw_dist = np.array(data["dist_coeffs"], dtype=np.float32).reshape(-1, 1)
        if use_zero_distortion:
            raw_dist = np.zeros_like(raw_dist)
    return K, raw_dist, data

K, dist_coeff, calibration_data = load_param(json_file, use_zero_distortion=True)

board_pattern = (10, 7)
board_cellsize = 0.025
board_criteria = (
    cv.CALIB_CB_ADAPTIVE_THRESH
    + cv.CALIB_CB_NORMALIZE_IMAGE
    + cv.CALIB_CB_FAST_CHECK
)


# =========================================================
# 2. 보조 함수들
# =========================================================
def overlay_warped_image(frame, overlay_img, dst_quad):
    """
    frame      : (H, W, 3) BGR
    overlay_img: (h, w, 3) 또는 (h, w, 4) BGR/BGRA
    dst_quad   : (4, 2) float32, 영상 위 목적지 사각형 꼭짓점
                 순서: 좌상, 우상, 우하, 좌하
    """
    h, w = overlay_img.shape[:2]

    src_quad = np.array([
        [0, 0],
        [w - 1, 0],
        [w - 1, h - 1],
        [0, h - 1]
    ], dtype=np.float32)

    H = cv.getPerspectiveTransform(src_quad, dst_quad.astype(np.float32))

    # 이미지 warp
    warped = cv.warpPerspective(
        overlay_img,
        H,
        (frame.shape[1], frame.shape[0]),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0) if overlay_img.shape[2] == 4 else (0, 0, 0)
    )

    # 알파 채널 있는 경우
    if overlay_img.shape[2] == 4:
        warped_rgb = warped[:, :, :3]
        alpha = warped[:, :, 3] / 255.0
        alpha = alpha[..., None]

        out = frame.astype(np.float32) * (1 - alpha) + warped_rgb.astype(np.float32) * alpha
        return np.clip(out, 0, 255).astype(np.uint8)

    # 알파 채널 없는 경우: 검은 배경 대신 폴리곤 마스크 사용
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv.fillConvexPoly(mask, np.int32(dst_quad), 255)

    mask_3 = (mask / 255.0)[..., None]
    out = frame.astype(np.float32) * (1 - mask_3) + warped[:, :, :3].astype(np.float32) * mask_3
    return np.clip(out, 0, 255).astype(np.uint8)


def project_plane_points(points_3d, rvec, tvec, K, dist_coeff):
    """
    3D 점들을 영상에 투영
    points_3d: (N, 3)
    return   : (N, 2)
    """
    img_pts, _ = cv.projectPoints(points_3d, rvec, tvec, K, dist_coeff)
    return img_pts.reshape(-1, 2)


# =========================================================
# 3. 비디오 / 이미지 열기
# =========================================================
video = cv.VideoCapture(video_file)
assert video.isOpened(), f"Cannot read the given input: {video_file}"

fps = video.get(cv.CAP_PROP_FPS)
width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

fourcc = cv.VideoWriter_fourcc(*"mp4v")
writer = cv.VideoWriter(save_file, fourcc, fps, (width, height))

# IMREAD_UNCHANGED: PNG 알파채널까지 읽기 위해 사용
overlay_img = cv.imread(ar_image_file, cv.IMREAD_UNCHANGED)
assert overlay_img is not None, f"Cannot read overlay image: {ar_image_file}"

# 3채널 이미지면 BGRA로 바꿀 수도 있음 (선택)
if overlay_img.ndim == 2:
    overlay_img = cv.cvtColor(overlay_img, cv.COLOR_GRAY2BGRA)
elif overlay_img.shape[2] == 3:
    # 알파 없는 이미지는 그대로 둬도 되지만, 필요하면 아래처럼 alpha 추가 가능
    pass


# =========================================================
# 4. 체스보드 3D 기준점 준비
# =========================================================
obj_points = board_cellsize * np.array(
    [[c, r, 0] for r in range(board_pattern[1]) for c in range(board_pattern[0])],
    dtype=np.float32
)

# ---------------------------------------------------------
# AR 이미지를 붙일 "보드 위 평면"의 3D 좌표
# 예: 체스보드 위 (col,row) = (2,2)에서 시작해서
#     가로 3칸, 세로 2칸 크기의 직사각형
#
# 순서 매우 중요: 좌상, 우상, 우하, 좌하
# ---------------------------------------------------------
c0, r0 = 2, 2
w_cells, h_cells = 3, 2

plane_3d = board_cellsize * np.array([
    [c0,           r0,           0],
    [c0 + w_cells, r0,           0],
    [c0 + w_cells, r0 + h_cells, 0],
    [c0,           r0 + h_cells, 0]
], dtype=np.float32)


# =========================================================
# 5. 메인 루프
# =========================================================
while True:
    valid, frame = video.read()
    if not valid:
        break

    success, corners = cv.findChessboardCorners(frame, board_pattern, board_criteria)

    if success:
        # 코너 refinement 하면 좀 더 안정적임
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners = cv.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )

        ret, rvec, tvec = cv.solvePnP(obj_points, corners, K, dist_coeff)
        if ret:
            # 3D 사각형 꼭짓점을 영상으로 투영
            dst_quad = project_plane_points(plane_3d, rvec, tvec, K, dist_coeff)

            # 이미지 합성
            frame = overlay_warped_image(frame, overlay_img, dst_quad)

            # 보기 좋게 테두리 표시
            cv.polylines(frame, [np.int32(dst_quad)], True, (0, 255, 0), 2, cv.LINE_AA)

            # 카메라 위치 표시 (기존 예제 유지)
            R, _ = cv.Rodrigues(rvec)
            p = (-R.T @ tvec).flatten()
            info = f"XYZ: [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}]"
            cv.putText(frame, info, (10, 25), cv.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 0), 1, cv.LINE_AA)

    writer.write(frame)
    cv.imshow("AR Image on Chessboard", frame)

    key = cv.waitKey(10)
    if key == ord(' '):
        key = cv.waitKey()
    if key == 27:  # ESC
        break

video.release()
writer.release()
cv.destroyAllWindows()

print(f"[INFO] Saved output video to: {save_file}")