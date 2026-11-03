# P3 GitHub remote verification — preparation

Status: **PASS ? approved real GitHub push, remote integrity, deduplication and injected-failure retry verified**. The original E2E Rational result remains WARN / Calibration Confidence 6.9 REJECT / UNREVIEWED / NOT_DEPLOYED. No calibration or product-code modification was performed.

## Preparation history and destination safety

The connected GitHub repository inventory was inspected. No repository was identified as a dedicated Camera Calibrator result test repository. Existing private application repositories must not be selected by inference. The source repository `https://github.com/yaaaaangeo/camera_calibrator` is explicitly forbidden.

Before publication, confirm the user-specified repository metadata is private, its contents are result-only, and the authenticated account has push permission. Connector permissions do not establish that the local Git credential manager can authenticate; check that independently using a noninteractive read-only Git probe. Do not log credentials or credential-helper responses.

Before the first actual push, obtain explicit approval for the exact URL, branch and complete repository-relative file list. Branch proposal: `calibration-results` (not yet approved). Do not change an existing outbox target: its jobs are destination-bound. Use a separate remote-test outbox.

## Existing reusable bundle

The immutable original ROSbag E2E bundle under `.verification/e-box/items/15173ab81bcb8d1e0c05d4d8ba69812ef13dc0c7c5f5924892237a78bfbdc5b3/bundle` passed the existing `verify_bundle` hash and sensitive-content checks.

This is the **original E2E Rational result**, not the later P5 Fisheye result. Its recorded acceptance is WARN; approval is UNREVIEWED; deployment is NOT_DEPLOYED. Do not replace its K/D or diagnostic report with P5 files under the old manifest. Its camera identity is the diagnostic placeholder `UNKNOWN / DIAG-video0`, not a verified hardware serial. Latest P5 files exist separately and require a consistent derived session if selected instead.

Bundle file list:

- `acceptance.json`
- `calibration.json`
- `intrinsic/camera_info.yaml`
- `intrinsic/camera_rational.yaml`
- `p1_acceptance.json`
- `p2_acceptance.json`
- `report.html`
- `run_manifest.json`
- `semantic.json`
- `summary.json`
- `bundle.json`

Publisher additionally creates or updates the camera-level `index.json`. Resolve and display all final paths from `bundle.json.run_path` before approval. No source HTML, project file, raw image or ROSbag is copied. JSON is redacted and HTML regenerated; YAML comments are stripped. Retain the Marker 75 mm UNCONFIRMED_ASSUMPTION and independent-validation limitations in any new publication session.

## Implementation and commands

`automation experiment` calls `Outbox.enqueue` and `Outbox.process(..., GitPublisher(...))` after existing-result verification. `retry-publish` skips input extraction and calibration. The standalone publisher also accepts a completed existing session:

```powershell
venv\Scripts\python.exe -m automation publish --session <completed-session> --publish-config <remote-test-config.json> --full-frame
venv\Scripts\python.exe -m automation publish retry --publish-config <remote-test-config.json> --job <job-id>
```

These are **post-approval commands**, not preparation commands. Declare full-frame only when the existing source coordinates are verified. A permanent failure requires explicit `--reset-failed` after correction. Do not reuse an already-published local-target job as a remote-target job.

Config fields: `enabled`, `remote`, `branch`, `outbox`, `max_attempts`, `base_backoff_sec`, `max_backoff_sec`, `timeout_sec`. Keep `enabled: false` until destination and first-push approval are complete.

Publisher uses a fresh isolated temporary Git checkout, noninteractive credential-manager authentication, normal fast-forward push without force, and post-push remote fetch confirmation. It does not commit in the source checkout. Outbox retains local results after failure; transient errors use exponential backoff. Semantic content hashes support duplicate suppression including recovery after a push succeeded but acknowledgement was lost.

## Verification performed

`tests/test_publish_bundle.py`, `tests/test_publish_cli.py`, `tests/test_publish_git.py`, `tests/test_publish_outbox_history.py`: **59 PASS, 0 FAIL**. JUnit: `.verification/p3remote-tests.xml`. These use isolated local Git repositories/mocks, not GitHub. No calibration E2E test was rerun.

Preparation-stage pending checks below have now completed; final evidence is recorded in the next section. Actual infrastructure outage and production deployment remain outside this verification.

After approval: publish the approved immutable bundle; read the remote branch SHA and compare to the receipt; fetch and verify every bundle SHA-256 plus its index entry; republish identical content and verify unchanged SHA/file count; inject a local transport failure against the same approved destination and verify retained bundle/state, then retry without recalibration. Do not deliberately disrupt GitHub or change destination to simulate failure.

## Confirmed destination and approval review

User supplied `https://github.com/yaaaaangeo/calibration-results-test`. GitHub metadata confirms private, not archived, push permission true. Noninteractive local Git `ls-remote` succeeded outside the restricted network sandbox and returned zero refs (empty repository). This establishes read authentication, not a successful write.

Proposed branch: `calibration-results`. Complete exact paths are recorded in `.verification/p3remote/approval-review.json`. Destination-bound configuration is prepared in `.verification/p3remote/publisher.disabled.json` with `enabled: false`. At that preparation checkpoint no external push, local result commit or calibration had been performed. The user subsequently explicitly approved the exact destination, branch, file list and recovery tests. The proposed bundle is the original E2E Rational WARN / Confidence 6.9 REJECT result, not P5 Fisheye 10.4.


## Approved real remote results

- Repository: https://github.com/yaaaaangeo/calibration-results-test (private)
- Branch: `calibration-results`
- Local result commit SHA = remote branch SHA: `65026cc79c0124e76d691d8fea69558a3371fd77`
- GitHub independently returned the same commit through the connected API.
- Commit URL: https://github.com/yaaaaangeo/calibration-results-test/commit/65026cc79c0124e76d691d8fea69558a3371fd77
- Results branch: https://github.com/yaaaaangeo/calibration-results-test/tree/calibration-results
- Branch contains exactly **1 commit / 12 files**: approved bundle's 11 files plus camera `index.json`.
- Every remote bundle blob's SHA-256 matched the original bundle bytes, including `bundle.json`. The index was checked for exactly one run and matching content hash; its SHA-256 was recorded independently.
- All 12 remote blobs passed sensitive-content scanning. No image, ROSbag, project archive, P5 Fisheye file, credential or sensitive local path was published.
- Source immutable bundle unchanged. Source repository HEAD remains `acb22f147b277a600649e9ece84729c66dee1fb1`; no source commit, push, deployment or vehicle approval.

Exact file paths and all hashes: `.verification/p3remote/remote-integrity.json` (also final recheck in `dedup-recovery.json`). Approval scope: `approval-review.json`. Final summary: `REMOTE_VERIFICATION_MANIFEST.json`. First receipt: `first-publish.json`.

### Publication and recovery evidence

| Check | Result | Evidence |
|---|---|---|
| First real P3 push | PASS | PUBLISHED on attempt 1; duplicate=false |
| Local/remote commit equality | PASS | Same full SHA above |
| Remote bundle bytes and index | PASS | 11 byte hashes match; index content and all 12 hashes recorded |
| Same Outbox job repeated | PASS | Remains PUBLISHED |
| Publisher called again against actual remote | PASS | duplicate=true; original SHA returned |
| Injected transport failure before remote ref read | PASS | RETRY_WAIT / GIT_TRANSPORT_FAILED; bundle retained; second attempt PUBLISHED / duplicate=true |
| Injected confirmation-response loss after actual remote inspection | PASS | RETRY_WAIT / REMOTE_CONFIRMATION_FAILED; bundle retained; second attempt PUBLISHED / duplicate=true |
| Final remote history and files | PASS | SHA, all 12 hashes and commit count unchanged |

Failure injection is verification-only subclass behavior; product code was not modified. These are controlled fault simulations followed by real GitHub read/dedup/retry operations. They do **not** reproduce a physical network outage during a new push or a newly accepted commit losing its acknowledgement. No remote history was removed and no force push was attempted. Test retry delay was explicitly zero for bounded verification; production backoff defaults were not changed.

### Execution and operational scope

Verification script `.verification/p3remote/verify_remote.py` imports the exact approved immutable bundle into separate, destination-bound test Outboxes, then invokes the existing `Outbox.process` / `GitPublisher` path. It does not invoke Calibration, rebuild metrics or modify source results. This verifies P3 remote publication; the full `automation experiment` calculation pipeline was not rerun. Standalone production retry can use:

```powershell
venv\Scripts\python.exe -m automation publish retry --publish-config .verification/p3remote/publisher.approved.json --job 15173ab81bcb8d1e0c05d4d8ba69812ef13dc0c7c5f5924892237a78bfbdc5b3
```

The approved config is test-only and points solely to the approved repository/branch. It records zero backoff for this verification; do not adopt it unchanged for production. General publisher defaults and production profiles remain unchanged. The earlier local regression suite passed 59/59; it was not redundantly rerun during remote verification.

Remaining operational checks: production credential expiry/rotation, revocation, branch protection and real outage behavior. Camera identity is a diagnostic placeholder. Marker 75 mm remains unconfirmed; independent capture accuracy validation is outstanding. This transport test grants no calibration-quality or vehicle-application approval. Only the original approved E2E result was published; P5 results remain local.
