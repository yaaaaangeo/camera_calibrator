# Output Sessions

사용자 결과물은 기본적으로 소스 저장소 밖의
`~/CameraCalibratorOutputs/<session>/`에 저장됩니다. Session 이름은
`YYYY-MM-DD_HHMMSS[_camera]`이며 충돌하면 `_02`, `_03` suffix가 붙습니다.

```text
<session>/
├── manifest.json
├── project/calibration.ccproj
├── intrinsic/
│   ├── camera_<model>.yaml
│   ├── camera_info.yaml
│   ├── kalibr/
│   └── subset/camera_subset_<model>.yaml
├── validation/subset_comparison/
├── paper/
├── windshield/
│   ├── geometry/<geometry-model>.yaml
│   ├── reflection/
│   └── ghost/
└── reports/
    ├── report.html
    ├── calibration.json
    └── dataset.csv
```

GUI 상단의 **Output Session**에서 Root 변경, 현재 Session 열기, **Export
All**을 사용할 수 있습니다. 개별 Export 버튼도 같은 Session을 사용하며
Project의 **다른 이름으로 저장**만 명시적 외부 경로를 선택합니다.

CLI에서 `--output-dir`를 생략하면 같은 Session 구조를 사용합니다. 기존 CI나
스크립트가 `--output-dir ./out`을 명시하면 하위 구조를 강제하지 않고 종전의
flat 파일명 계약을 유지합니다.

저장소의 `calibration_output/`은 과거 CLI 기본 실행이 만든 tracked 결과
snapshot입니다. 새 실행의 기본 목적지가 아니며, serializer 회귀 확인을 위한
legacy sample로만 취급합니다. 새 사용자 결과를 이 디렉터리에 추가하지
마십시오.

## 통합 전 저장 진입점 감사

| 진입점 | 기존 목적지 선택 | Session 목적지 |
|---|---|---|
| OpenCV YAML | Result View Save dialog | `intrinsic/camera_<model>.yaml` |
| Best Subset YAML | Result View Save dialog | `intrinsic/subset/camera_subset_<model>.yaml` |
| Paper Metrics | directory dialog | `paper/` |
| Project | `.ccproj` Save dialog | `project/calibration.ccproj` |
| HTML/JSON/CSV/ROS/Kalibr | CLI `--output-dir` | `reports/`, `intrinsic/kalibr/` |
| Subset hold-out comparison | GUI directory dialog / CLI `--output` | `validation/subset_comparison/` (GUI) |
| Windshield Geometry | panel Save dialog | `windshield/geometry/<model>.yaml` |
| Reflection Evaluation | panel Save dialog | `windshield/reflection/` |
| Ghost model | panel Save dialog | `windshield/ghost/` |

Library 저장, `~/.camera_calibrator/autosave.ccproj`, live/rosbag 이미지 추출은
각각 내부 이력·복구·입력 획득 기능이므로 사용자 Export Session과 분리되어
있습니다.

