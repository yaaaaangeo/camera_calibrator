"""Calibrate every camera against the LiDAR in a calibration room, from one bag.

    ../venv/bin/python gui/room_calib.py BAG --lidar TOPIC[,TOPIC] --cameras TOPIC[=NAME[:방향]][,...]
        [--vehicle NAME] [--roi x0,x1,y0,y1,z0,z1[,yaw][;...]] [--camera PRESET]
        [--target PRESET] [--sweeps 1] [--frames 1] [--at 0.5] [--out DIR]

One moment of the recording -- by default one LiDAR sweep and one picture per
camera; everything stands still, so more only averages noise. The LiDAR boards
are found once (inside the ROI if one is given, or the vehicle profile has
one) and every camera is paired against them, one after another, so a board
one camera helped place is there for the next.

Each camera's lens is ranked over all presets unless --camera names one (or the
vehicle profile remembers one for that topic).

Results go to calib_result/<vehicle>/<recording date>/<camera>/ (or --out):
    extrinsic.txt  result.yaml  boards.yaml  overlay.png  report.txt
plus session.yaml with every camera's numbers, and the vehicle's profile is
updated with the topics, names and ROI used.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gui.core import presets, room, session  # noqa: E402
from gui.core.project import FilterBox, Target  # noqa: E402


def parse_layouts(text: str | None):
    """'1,2,4,3;5,6,8,7' -> ((1,2,4,3), (5,6,8,7)). Quadrant order TL, TR, BR, BL."""
    if not text:
        return (room.STOCK_LAYOUT,)
    out = []
    for part in text.split(";"):
        ids = tuple(int(x) for x in part.split(","))
        if len(ids) != 4:
            raise SystemExit(f"레이아웃은 마커 4개씩입니다: {part!r}")
        out.append(ids)
    return tuple(out)


def parse_roi(text: str | None) -> list[FilterBox]:
    """'x0,x1,y0,y1,z0,z1[,yaw];...' -> FilterBoxes."""
    if not text:
        return []
    out = []
    for part in text.split(";"):
        v = [float(x) for x in part.split(",")]
        if len(v) not in (6, 7):
            raise SystemExit(f"ROI 는 x0,x1,y0,y1,z0,z1[,yaw] 입니다: {part!r}")
        out.append(FilterBox(*v[:6], yaw=v[6] if len(v) == 7 else 0.0))
    return out


def parse_cameras(text: str, profile) -> list[tuple[str, str, str]]:
    """'T1=FRONT:정면,T2' -> [(T1, FRONT, 정면), (T2, <profile or topic name>, <profile or guessed>)]."""
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        topic, _, rest = part.partition("=")
        name, _, facing = rest.partition(":")
        name = session.safe_name(name) if name else profile.camera_name(topic)
        if facing and facing not in room.DIRECTIONS:
            guessed = room.guess_direction(facing)
            if not guessed:
                raise SystemExit(f"방향 '{facing}' 을 모릅니다. 쓸 수 있는 것: {' '.join(room.DIRECTIONS)} "
                                 f"(또는 front, left, rear_right …)")
            facing = guessed
        out.append((topic, name, facing or profile.camera_direction(topic, name)))
    names = [n for _, n, _ in out]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        raise SystemExit(f"카메라 이름이 겹칩니다: {', '.join(sorted(dup))} — TOPIC=NAME 으로 정하세요")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("--lidar", default=None, help="쉼표로 여러 개. 생략하면 차량 프로필의 토픽")
    ap.add_argument("--cameras", "--camera-topic", dest="cameras", required=True,
                    help="TOPIC[=NAME[:방향]] 을 쉼표로. NAME 은 결과 폴더명, 방향은 정면/좌전/좌/좌후/후/우후/우/우전 "
                         "(예: /cam/a=FRONT_LEFT:좌전). 방향을 안 주면 프로필, 그것도 없으면 이름에서 짐작")
    ap.add_argument("--vehicle", default="", help="차량 이름 — 결과는 calib_result/<차량>/<날짜>/ 에")
    ap.add_argument("--roi", default=None, help="x0,x1,y0,y1,z0,z1[,yaw] 를 ';' 로. 생략하면 차량 프로필의 ROI, 그것도 없으면 전체")
    ap.add_argument("--no-roi", action="store_true", help="프로필에 ROI 가 있어도 전체 클라우드 탐색")
    ap.add_argument("--camera", default=None, help="모든 카메라에 쓸 내부 파라미터 프리셋. 생략하면 카메라마다 자동 순위")
    ap.add_argument("--target", default=None, help="gui/config/target_specs.yaml 의 이름 (기본: 프로필 또는 원본 CAD)")
    ap.add_argument("--layouts", default=None,
                    help="보드별 마커 id (TL,TR,BR,BL), ';' 로 구분. 생략하면 모든 보드가 1,2,4,3")
    ap.add_argument("--method", default="occupancy", help="LiDAR 구멍 검출 방식: occupancy(기본, 실패 시 template 재시도) / template")
    ap.add_argument("--sweeps", type=int, default=0,
                    help="쌓을 LiDAR 스윕 수. 0(기본)=자동: 보통 1, 다운샘플된 점군이면 10")
    ap.add_argument("--frames", type=int, default=1)
    ap.add_argument("--at", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="세션 폴더를 직접 지정")
    ap.add_argument("--no-profile", action="store_true", help="차량 프로필을 갱신하지 않음")
    ap.add_argument("-v", "--verbose", action="count", default=0, help="자세한 로그 (-vv 는 디버그)")
    ap.add_argument("--log-file", default=None, help="로그를 파일에도 남김")
    args = ap.parse_args(argv)
    from calibration.log_utils import setup_logging

    setup_logging(verbosity=args.verbose, log_file=args.log_file)

    profile = session.load_profile(args.vehicle) if args.vehicle else session.VehicleProfile(name="")
    presets_by_name = {p.name: p.camera for p in presets.load()}
    if args.camera and args.camera not in presets_by_name:
        raise SystemExit(f"카메라 '{args.camera}' 없음. 있는 것: {', '.join(presets_by_name)}")
    targets = {p.name: p.target for p in presets.load_targets()}
    target_name = args.target or profile.target or "원본 CAD"
    target = targets.get(target_name) or Target()
    layouts = parse_layouts(args.layouts or profile.layouts or None)
    lidar_topics = args.lidar.split(",") if args.lidar else list(profile.lidar_topics)
    if not lidar_topics:
        raise SystemExit("--lidar 를 주세요 (차량 프로필에도 없음)")
    cams = parse_cameras(args.cameras, profile)
    roi = [] if args.no_roi else (parse_roi(args.roi) if args.roi else list(profile.roi))

    t0 = time.time()
    cap = room.load_capture_multi(args.bag, lidar_topics, [t for t, _, _ in cams], args.at, args.sweeps, args.frames)
    sid = session.session_id(cap.stamp_ns)
    print(f"입력: 스윕 {cap.n_sweeps}개 {len(cap.cloud):,}점, 카메라 {len(cams)}대 "
          f"({', '.join(f'{n} {len(cap.images[t])}장' for t, n, _ in cams)})  ({time.time() - t0:.1f}s)")
    for line in cap.note.splitlines():
        print("  " + line)

    t0 = time.time()
    lidar, stats = room.find_lidar_boards(cap.cloud, cap.single, target, args.method, roi=roi or None)
    where = f"ROI {len(roi)}개 안 {stats['searched']:,}점" if roi else f"전체 {stats['searched']:,}점"
    print(f"LiDAR: 보드 {len(lidar)}개 ({where}, {time.time() - t0:.1f}s)")
    for lb in lidar:
        c = lb.centers.mean(axis=0)
        print(f"  {lb.label}  [{c[0]:6.2f} {c[1]:6.2f} {c[2]:5.2f}]  {lb.det.method}")

    prepared = []
    for topic, name, facing in cams:
        images = cap.images.get(topic) or []
        if not images:
            print(f"\n=== {name} ({topic}): 이미지 없음")
            continue
        h, w = images[0].shape[:2]
        preset = args.camera or profile.lens_for(topic, name)  # 지난번 렌즈, 없으면 Camera Intrinsic 이 등록한 렌즈
        line = ""
        if preset not in presets_by_name:
            ranking = room.rank_cameras(images[len(images) // 2], presets_by_name, target, layouts, w, h)
            if not ranking or ranking[0][1] == 0:
                print(f"\n=== {name} ({topic}): 맞는 내부 파라미터 프리셋 없음")
                continue
            preset = ranking[0][0]
            line = f"내부 파라미터 자동 선택: {preset} (보드 {ranking[0][1]}, {ranking[0][2]:.2f}px)"
        camera = presets_by_name[preset]
        cam_boards = room.detect_boards_multi(images, camera, target, layouts)
        prepared.append((topic, name, facing, preset, camera, images, cam_boards, line))

    # Cameras that see several boards first; their directions give the vehicle's
    # forward, and that tells a one-board camera which of the identical boards it sees.
    jobs = [room.CameraJob(cb, camera, None, facing) for _, _, facing, _, camera, _, cb, _ in prepared]
    solved, forward = room.solve_cameras(jobs, lidar, cap.cloud, cap.single, target, args.method,
                                         known_forward=profile.forward_deg)
    if forward is not None:
        print(f"\n차량 정면: LiDAR 좌표에서 방위 {forward:+.1f}°")
    results = []
    for (topic, name, facing, preset, camera, images, cam_boards, line), (res, _) in zip(prepared, solved):
        print(f"\n=== {name} ({topic})  방향 {facing or '?'}")
        if line:
            print("  " + line)
        print(f"  카메라 보드 {len(cam_boards)}개")
        if res.ok:
            print(f"  보드 쌍 {len(res.scenes)}개 · RMSE {res.solution.rmse * 1000:.1f} mm · 이미지 {res.image_rmse:.2f} px"
                  f"  ({', '.join(f'{res.cams[j].label}={res.lidar[i].label}' for i, j, _ in res.pairs)})")
            print("  " + session.pose_text(res.solution).replace("\n", "\n  "))
            for n in res.notes:
                print("  · " + n)
        else:
            print("  실패: " + "; ".join(res.notes))
        results.append((topic, name, preset, camera, images, res, facing))

    out = Path(args.out) if args.out else (session.vehicle_dir(args.vehicle) / sid if args.vehicle else None)
    if out is None:
        out = session.vehicle_dir("_no_vehicle") / sid
    summary = []
    for topic, name, preset, camera, images, res, _ in results:
        if not res.ok:
            continue
        meta = {"vehicle": args.vehicle, "session": sid, "date": session.session_label(sid), "camera_name": name,
                "camera_topic": topic, "source": str(args.bag), "lidar_topics": lidar_topics,
                "sweeps": cap.n_sweeps, "frames": len(images), "target": target_name}
        session.save_camera(out / name, res=res, camera=camera, camera_preset=preset, image=images[len(images) // 2],
                            cloud=cap.cloud, intensity=cap.intensity, meta=meta)
        summary.append({"name": name, "topic": topic, "preset": preset, "ok": True, "boards": len(res.scenes),
                        "rmse_m": round(float(res.solution.rmse), 5), "image_rmse_px": round(float(res.image_rmse), 3),
                        "T_camera_lidar": np.round(res.solution.matrix, 9).tolist()})
    session.write_summary(out, vehicle=args.vehicle, sid=sid, source=str(args.bag), lidar_topics=lidar_topics,
                          n_sweeps=cap.n_sweeps, roi=roi, target=target_name, cameras=summary)
    if args.vehicle and not args.no_profile:
        profile.lidar_topics = lidar_topics
        for topic, name, preset, *_, facing in results:
            profile.cameras[topic] = session.CameraEntry(name, preset, facing)
        if forward is not None:
            profile.forward_deg = forward
        profile.roi = roi or profile.roi
        profile.target = target_name
        profile.save()
    print(f"\n저장: {out}  (카메라 {len(summary)}/{len(cams)}대)")
    return 0 if len(summary) == len(cams) else 1


if __name__ == "__main__":
    sys.exit(main())
