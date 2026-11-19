"""
camera_calibrator.ui.help_view
==============================

상단 [설명] 메뉴가 여는 읽기 전용 사용 설명서. 홈 화면, 상단 메뉴,
Camera Intrinsic, Camera–LiDAR 순서로 안내한다.
"""

from __future__ import annotations

from PySide6.QtWidgets import QTextBrowser, QVBoxLayout, QWidget

from ui.theme import Theme


class HelpView(QWidget):
    """Calibration Tool 사용 설명서."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setStyleSheet(
            f"QTextBrowser {{ background: {Theme.BG_PRIMARY}; color: {Theme.TEXT_PRIMARY}; "
            f"border: 1px solid {Theme.BORDER}; padding: 14px; }}"
        )
        browser.setHtml(self._html())
        self.browser = browser
        layout.addWidget(browser)

    @staticmethod
    def _html() -> str:
        css = (
            _CSS.replace("@BORDER@", Theme.BORDER)
            .replace("@HEAD@", Theme.BG_TERTIARY)
            .replace("@NOTE@", Theme.BG_SECONDARY)
            .replace("@ACCENT@", Theme.ACCENT)
            .replace("@GOOD@", Theme.GOOD)
            .replace("@WARN@", Theme.WARNING)
        )
        return css + _OVERVIEW + _INTRINSIC + _ROOM


_CSS = """
        <style>
            body { font-family: 'Segoe UI', 'Malgun Gothic', sans-serif; font-size: 13px; line-height: 1.55; }
            h1 { font-size: 22px; margin: 0 0 10px 0; color: @ACCENT@; }
            h2 { font-size: 17px; margin: 22px 0 8px 0; color: @ACCENT@; }
            h3 { font-size: 14px; margin: 16px 0 6px 0; }
            p { margin: 6px 0; }
            table { border-collapse: collapse; width: 100%; margin: 8px 0 14px 0; }
            th, td { border: 1px solid @BORDER@; padding: 8px; vertical-align: top; }
            th { background: @HEAD@; font-weight: 700; }
            .note { background: @NOTE@; border: 1px solid @BORDER@; padding: 10px; margin: 10px 0; }
            .good { color: @GOOD@; font-weight: 700; }
            .warn { color: @WARN@; font-weight: 700; }
        </style>
"""

_OVERVIEW = """
        <h1>Calibration Tool 사용 설명서</h1>
        <p>
            카메라 <b>내부파라미터</b>(K, D)와 <b>카메라·LiDAR 외부파라미터</b>(R, t)를 구하는 도구입니다.
            홈 화면에서 작업을 고르고, 각 화면 왼쪽 위의 <b>← Calibration Home</b> / <b>← Home</b> 버튼으로 돌아옵니다.
        </p>
        <table>
            <tr><th>홈 카드</th><th>하는 일</th><th>결과</th></tr>
            <tr><td><b>CAMERA INTRINSIC</b></td>
                <td>보드 사진(이미지 · rosbag · 실시간 ROS 토픽)으로 카메라 모델 4종을 계산·비교하고 추천합니다.</td>
                <td>K, D (OpenCV YAML, ROS camera_info, Kalibr, HTML 리포트)</td></tr>
            <tr><td><b>CAMERA–LiDAR</b></td>
                <td>보드가 놓인 공간을 한 번 캡처(실시간 토픽 또는 bag)해서 LiDAR와 여러 카메라의 extrinsic을 구합니다.</td>
                <td>카메라별 R, t (extrinsic.txt, result.yaml, 오버레이 이미지)</td></tr>
        </table>

        <h2>상단 메뉴</h2>
        <table>
            <tr><th>메뉴</th><th>항목</th><th>설명</th></tr>
            <tr><td><b>설명</b></td><td>사용 설명서 열기 <i>(F1)</i></td><td>지금 보고 있는 이 문서입니다.</td></tr>
            <tr><td><b>도구</b></td><td>Intrinsic 계산 엔진 자체 진단</td>
                <td>정답을 미리 아는 합성 ChArUco 데이터로 4개 모델을 계산해, 복원된 fx·fy·cx·cy가 정답과 맞는지 확인합니다.
                    설치 직후나 환경을 바꾼 뒤 계산 엔진이 정상인지 볼 때 씁니다. 불러온 데이터와는 무관하며 수 초~수십 초 걸립니다.</td></tr>
        </table>
        <div class="note">
            내부파라미터 작업은 <b>Export All</b>을 누르면 결과 폴더의 <code>project/calibration.ccproj</code>에 함께 저장됩니다.
            프로그램이 비정상 종료되면 다음 실행 때 마지막 계산 결과를 복구할지 묻습니다.
            Camera–LiDAR 결과는 7단계 <b>저장</b> 화면에서 저장합니다.
        </div>
        <div class="note">
            <b>Git 업로드</b>: 결과를 결과 전용 Git 저장소에 올립니다. Intrinsic 은 Export All 옆 버튼,
            Camera–LiDAR 는 7단계 저장 화면의 버튼입니다. 처음 한 번 저장소 주소·브랜치·작성자를 입력하면 기억하고,
            인증은 이 PC 의 git 설정(ssh 키 또는 credential helper)을 그대로 씁니다. 올라갈 파일 목록을 보여준 뒤
            [업로드]를 눌러야 올립니다. 결과값 파일만 올립니다 - Intrinsic 은 <code>camera_&lt;model&gt;.yaml</code>·<code>camera_info.yaml</code>,
            Camera–LiDAR 는 카메라별 <code>extrinsic.txt</code>·<code>result.yaml</code>.
            차량을 골랐으면 차량 정보(<code>vehicles/&lt;차량&gt;/vehicle.yaml</code>·<code>lenses.yaml</code>)도 같이 올라갑니다.
        </div>
        <div class="note">
            <b>화면 테마</b>: 메뉴 <b>보기 → 화면 테마</b>에서 다크(검은 바탕) / 라이트(흰 바탕에 검은 글씨)를 고릅니다.
            고르면 바로 다시 시작할지 묻고, 다음 실행부터도 기억합니다.
        </div>
        <div class="note">
            <b>차량 선택</b>: 홈 화면에서 차량을 먼저 고릅니다. 두 기능이 같은 차량 프로필을 씁니다 -
            Camera Intrinsic 의 Camera Name 은 그 차량의 카메라 목록에서 고르고, <b>CAMERA–LiDAR 렌즈로 등록</b>하면
            그 차량 카메라의 렌즈로 지정되어 CAMERA–LiDAR 3단계에서 자동으로 골라집니다.
            차량에 달지 않은 카메라는 <b>차량 없음 (카메라 단품)</b> (Camera Intrinsic 만).
            <b>GitHub에서 가져오기</b>는 결과 저장소의 차량 정보를 이 PC 로 받아옵니다.
        </div>
"""

_INTRINSIC = """
        <h2>Camera Intrinsic</h2>

        <div class="note">
            <b>Camera Intrinsic 추천 흐름</b><br>
            Camera Setup에서 이미지 불러오기 → ① Dataset에서 Coverage/Diversity/Batch 확인 →
            캘리브레이션 실행 → ② Preview에서 보정 결과 확인 →
            ③ Model Comparison에서 모델 선택 → Camera Setup의 Export 버튼으로 저장.
        </div>

        <h3>Camera Setup 패널</h3>
        <p>
            화면 위쪽의 Camera Setup은 왼쪽(Camera) / 중앙(Pattern) / 오른쪽(Actions)
            3열로 구성됩니다.
        </p>
        <table>
            <tr><th>열</th><th>구성</th></tr>
            <tr>
                <td><b>왼쪽 (Camera)</b></td>
                <td>
                    Vehicle(홈 화면에서 고른 차량) → Camera Name(차량의 카메라 목록에서 고르거나 새 이름) →
                    <b>INPUT</b>(실시간 / rosbag / 이미지 3버튼 - 실시간은 ROS1/ROS2 이미지
                    토픽을 실시간 구독해 직접 캡처(ROS 설치 없이도 됨: ROS 1 직접 접속, ROS 2 는 rclpy 또는 DDS), rosbag은 ROS1 .bag·ROS2 .db3/.mcap
                    로그에서 이미지 토픽을 뽑아 자동 추출, 이미지는 jpg/png/bmp 파일을 직접
                    선택) → <b>해상도 확인</b>(JPEG/PNG 이미지 한 장을 실제로 디코딩해 크기를
                    확인하고 Width/Height에 자동 반영) → Width/Height.
                </td>
            </tr>
            <tr>
                <td><b>중앙 (Pattern)</b></td>
                <td>Pattern type → Target-specific fields such as
                    Columns/Rows, Square Size, Center Spacing, Tag Pitch, Marker Size,
                    Dictionary/Tag Family, Grid type(Circle Grid), AprilGrid variant.</td>
            </tr>
            <tr>
                <td><b>오른쪽 (Actions)</b></td>
                <td>
                    <b>캘리브레이션 실행</b> → <b>Export</b>(계산된 모델을 골라
                    OpenCV YAML로 저장) → <b>취소</b>(코너 검출/모델 계산이 진행
                    중일 때 즉시 중단하고, 원하는 데이터/설정으로 다시 실행할 수
                    있는 상태로 되돌립니다). Rational on/off 체크박스는 없습니다 -
                    Standard 계산은 항상 Ideal Pinhole/Brown-Conrady/Rational
                    (8계수 고정)/Fisheye 네 모델을 함께 계산합니다.
                </td>
            </tr>
        </table>

        <h3>탭별 기능</h3>
        <table>
            <tr><th>탭</th><th>무엇을 보는 곳인가</th><th>판단 기준</th></tr>
            <tr>
                <td><b>① Dataset</b></td>
                <td>캘리브레이션에 사용할 이미지의 검출 상태를 확인합니다.
                    위에서부터 Coverage Map(보드가 이미지 안에서 얼마나 다양한 위치/크기/각도로 찍혔는지),
                    Dataset Diversity(다양성 점수 + Overall Dataset Score), Batch(이미지별 상태/코너 수/
                    재투영 오차/품질 점수)를 한 화면에서 확인합니다.</td>
                <td>검출 성공 프레임 수가 충분한지, 이미지가 흔들리거나 흐리지 않은지, 같은 포즈만 반복되지 않았는지 봅니다.
                    중앙만 찍힌 데이터보다 모서리/가장자리까지 넓게 분포한 데이터가 좋습니다. 실시간 캡처는 최소 50장 이상을 권장합니다.</td>
            </tr>
            <tr>
                <td><b>② Preview</b></td>
                <td>이미지 하나를 골라 원본과 (왜곡 보정 후 + Straightness 오버레이)를 나란히 보여줍니다.
                    오른쪽 패널의 초록~빨강 선은 보드의 행/열이 보정 후 얼마나 곧아졌는지를 나타내고,
                    이미지 아래에는 보정 전/후 Line Straightness 수치가 함께 표시됩니다.</td>
                <td>중앙보다 가장자리에서 선이 더 휘는지(빨강에 가까운지), 보정 전/후 수치가
                    얼마나 개선됐는지 봅니다. 0.5px 이하면 방사 왜곡이 사실상 제거된 것으로 봅니다.</td>
            </tr>
            <tr>
                <td><b>③ Model Comparison</b></td>
                <td>Ideal Pinhole, Brown-Conrady, Rational, Fisheye 모델을 같은 데이터 기준으로 비교하고 추천 모델을 보여줍니다.</td>
                <td>RMSE만 보지 말고 Validation, 가장자리 오차, 안정성, 모델 복잡도 패널티를 함께 봅니다. 복잡한 모델은 오차가 조금 낮아도 안정성이 낮으면 추천에서 밀릴 수 있습니다.</td>
            </tr>
        </table>

        <h3>비교표에서 쓰는 주요 기준</h3>
        <table>
            <tr><th>지표</th><th>의미</th><th>해석</th></tr>
            <tr>
                <td><b>Reprojection Error</b></td>
                <td>실제 검출된 코너 위치와 계산된 카메라 모델이 다시 투영한 코너 위치 사이의 픽셀 거리입니다.</td>
                <td><span class="good">낮을수록 좋습니다.</span> 다만 학습 이미지에서만 낮은 값은 과적합일 수 있습니다.</td>
            </tr>
            <tr>
                <td><b>RMSE</b></td>
                <td>오차 제곱 평균의 제곱근입니다. 큰 오차 프레임의 영향을 비교적 크게 받습니다.</td>
                <td>전체 대표 오차로 보기 좋지만, Max/P95와 같이 봐야 합니다.</td>
            </tr>
            <tr>
                <td><b>Mean / Median</b></td>
                <td>Mean은 평균, Median은 중앙값입니다.</td>
                <td>Mean이 Median보다 많이 크면 일부 나쁜 프레임이나 영역이 평균을 끌어올리고 있을 가능성이 있습니다.</td>
            </tr>
            <tr>
                <td><b>P95 / P99 / Max</b></td>
                <td>오차가 큰 쪽의 꼬리를 보는 지표입니다. P95는 95% 지점, P99는 99% 지점, Max는 최악값입니다.</td>
                <td>실사용 안정성을 볼 때 중요합니다. 평균은 낮아도 P99/Max가 크면 특정 영역에서 보정이 깨질 수 있습니다.</td>
            </tr>
            <tr>
                <td><b>Center / Middle / Edge</b></td>
                <td>이미지 중심, 중간 반경, 가장자리 영역을 나누어 오차를 계산합니다.</td>
                <td>광각/왜곡 렌즈는 Edge 오차가 특히 중요합니다. 중앙만 좋은 모델은 실제 보정에서 부족할 수 있습니다.</td>
            </tr>
            <tr>
                <td><b>AIC / BIC</b></td>
                <td>오차와 모델 복잡도를 함께 보는 정보 기준입니다.</td>
                <td><span class="good">낮을수록 좋습니다.</span> 계수가 많은 모델이 단순히 파라미터 수로 이기는 것을 막는 보정값입니다.</td>
            </tr>
            <tr>
                <td><b>Hold-out</b></td>
                <td>데이터 일부를 검증용으로 떼어 모델의 일반화 성능을 확인합니다.</td>
                <td>여기서도 안정적으로 낮은 오차를 내는 모델이 실제 사용에 더 안전합니다.</td>
            </tr>
            <tr>
                <td><b>Bootstrap / Stability</b></td>
                <td>데이터를 여러 번 다시 뽑아 계산했을 때 결과가 얼마나 흔들리는지 봅니다.</td>
                <td>오차가 낮아도 파라미터가 크게 흔들리면 데이터가 부족하거나 모델이 과하게 복잡할 수 있습니다.</td>
            </tr>
        </table>
        """

_ROOM = """
        <h2>Camera–LiDAR</h2>
        <p>
            보드 여러 장이 놓인 공간을 <b>한 번 캡처</b>해서 LiDAR와 <b>여러 카메라</b>의 extrinsic을 한꺼번에 구합니다.
            캡처하는 동안 차량과 보드는 모두 정지해 있어야 합니다.
            왼쪽 단계 목록의 ✓ 표시는 끝난 단계이고, 단계 이름을 클릭하면 그 단계로 이동합니다.
        </p>
        <div class="note">
            <b>차량 선택</b> — Camera–LiDAR를 열면 차량부터 고릅니다(새 이름이면 새 차량). 결과는
            <code>extrinsic/calib_result/&lt;차량&gt;/&lt;캡처 시각&gt;/&lt;카메라&gt;/</code>에 저장됩니다.
            같은 차량을 다시 고르면 LiDAR 토픽, 카메라 이름, 내부파라미터 프리셋, ROI를 그대로 불러오고,
            지난 결과를 <b>기존 extrinsic</b>으로 표시해 새 결과와 비교할 수 있게 합니다.
            차량은 홈 화면에서 고르고 바꿉니다 (창 위쪽 🚚 에 지금 차량이 표시됩니다).
        </div>
        <table>
            <tr><th>단계</th><th>하는 일</th><th>확인할 것</th></tr>
            <tr><td><b>1. 캡처</b></td>
                <td><b>실시간 탭</b>: <b>탐색</b> → 토픽 표에서 LiDAR(여러 개면 합쳐서 씀)와 카메라를 체크 → <b>구독 시작</b>.
                    썸네일에 표시되는 마커 검출 수를 보고 보드가 보이는 카메라를 <b>포함</b>하고 이름(FRONT_LEFT 등)을 붙인 뒤 <b>캡처</b>합니다.
                    <b>마커 보이는 카메라만 포함</b> 버튼으로 한 번에 고를 수도 있습니다.<br>
                    <b>bag 탭</b>: bag(ROS1) 또는 rosbag2(db3/mcap)를 엽니다.</td>
                <td>최근 스윕끼리 겹치는 비율이 낮거나(차량·보드가 움직임) 카메라와 LiDAR의 시각 차이가 200 ms를 넘으면 경고가 뜹니다.
                    캡처 원본은 세션의 <code>capture/</code>에 저장되어 나중에 bag 탭으로 같은 결과를 재현할 수 있습니다.</td></tr>
            <tr><td><b>2. LiDAR 보드</b></td>
                <td>위에서 본 화면(BEV)에서 <b>드래그로 ROI 그리기</b>로 보드가 있는 영역을 그리면 그 안에서만 보드를 찾습니다.
                    핸들로 크기와 회전을 바꾸고, 높이는 숫자로 입력합니다. 찾은 보드는 모든 카메라가 함께 씁니다.</td>
                <td>ROI가 없으면 전체를 탐색해서 오래 걸립니다. ROI는 차량 프로필에 저장되어 다음 세션에는 자동 탐색됩니다.</td></tr>
            <tr><td><b>3. 카메라</b></td>
                <td><b>모든 카메라 자동 순위</b>를 누르면 카메라마다 내부파라미터 프리셋을 보드 재투영 오차로 골라 줍니다.
                    <b>파일에서 렌즈 불러오기…</b>로 Camera Intrinsic 결과 YAML 을 바로 쓸 수 있고, Intrinsic 의 Export 창
                    <b>CAMERA–LiDAR 렌즈로 등록</b>으로도 이 목록에 넣을 수 있습니다.</td>
                <td>왜곡 보정 보기에서 직선이 곧게 펴지는지, 이미지 해상도에 맞는 프리셋인지 확인합니다.</td></tr>
            <tr><td><b>4. 카메라 보드</b></td><td><b>모든 카메라 보드 검출</b>을 누르면 카메라마다 마커를 보드별로 묶고 PnP로 자세를 구합니다.</td>
                <td>잘못 묶인 보드는 표에서 사용을 해제합니다.</td></tr>
            <tr><td><b>5. 대응·풀이</b></td><td><b>모든 카메라 풀기</b>. 위 표는 카메라별 요약, 아래 표는 선택한 카메라의 보드별 수치입니다.</td>
                <td>“확인 필요”로 표시된 카메라(재투영 오차가 크거나 보드가 3개 미만)를 살펴봅니다.</td></tr>
            <tr><td><b>6. 오버레이 검증</b></td><td>상단 <b>카메라</b> 선택으로 카메라를 바꿔 가며 투영 결과를 봅니다. 휠로 확대, 드래그로 이동, 더블클릭으로 맞춤.</td>
                <td>보드 테두리·구멍·벽 모서리에서 점과 영상이 어긋나지 않는지 봅니다.</td></tr>
            <tr><td><b>7. 저장</b></td><td>체크한 카메라를 모두 세션 폴더에 저장하고 <code>session.yaml</code>과 차량 프로필을 갱신합니다.</td>
                <td>카메라마다 <code>extrinsic.txt</code>(FAST-LIVO2 형식), <code>result.yaml</code>, <code>boards.yaml</code>, <code>overlay.png</code>, <code>report.txt</code></td></tr>
        </table>
        <p>3~6단계는 카메라마다 하는 단계라 창 위쪽에 <b>카메라 선택</b> 칸이 나옵니다.</p>
        <div class="note">
            <b>프리셋 파일</b> — 카메라 내부파라미터: <code>extrinsic/gui/config/camera_intrinsics.yaml</code>,
            보드 규격: <code>extrinsic/gui/config/target_specs.yaml</code>.<br>
            Camera Intrinsic에서 구한 값을 Camera–LiDAR에서 쓰려면 Export한 fx, fy, cx, cy, k1~k3, p1, p2
            (Rational이면 k4~k6, Fisheye면 <code>model: fisheye</code>와 k1~k4)를
            <code>camera_intrinsics.yaml</code>에 새 항목으로 추가하세요.<br>
            <b>다른 PC의 ROS 2 토픽</b>을 받을 때 큰 포인트클라우드가 빠지면
            <code>sudo sysctl -w net.core.rmem_max=8388608</code>을 실행하세요.
        </div>
"""
