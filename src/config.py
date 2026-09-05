"""Configuration for the pipeline. Every research knob is a flag here so the key
ablations are one-line changes:

  * use_endpoint_anchoring:  the STIR-specific signal on/off
  * use_cycle_consistency:   forward/backward signal on/off
  * use_agreement:           multi-teacher consensus on/off
  * teachers:                which trackers are in the ensemble
  * mode:                    per-frame "select" a teacher vs "aggregate" teachers
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class VerifierConfig:
    mode: str = "select"                    # {"select", "aggregate"}

    # signal switches (ablation)
    use_endpoint_anchoring: bool = True
    use_cycle_consistency: bool = True
    use_agreement: bool = True
    use_native_confidence: bool = True

    # softness of each signal, in PIXELS (tie to STIR delta thresholds 2/4/8/16/32)
    endpoint_tau: float = 12.0
    cycle_tau: float = 8.0
    agree_tau: float = 8.0

    # signal exponents (relative influence)
    w_endpoint: float = 1.0
    w_cycle: float = 1.0
    w_agreement: float = 1.0
    w_native: float = 0.5


    # turning confidence into a supervision mask
    weight_floor: float = 0.2               # frames below this confidence -> masked out
    visibility_floor: float = 0.5           # don't supervise position on occluded frames


@dataclass
class TeacherConfig:
    name: str
    checkpoint: str = ""                    # weights path (cotracker3/locotrack) or unused (mft)
    repo_root: str = ""                     # for teachers wired via an external repo (e.g. cloned MFT)
    config_path: str = ""                   # e.g. "configs/MFT_cfg.py" relative to repo_root
    enabled: bool = True
    run_cycle: bool = True                  # run the backward pass for this teacher


@dataclass
class DataConfig:
    # clips WITH IR endpoints (public STIR) — highest quality pseudo-labels
    labeled_root: str = "data/stir_labeled"
    # extra in-domain laparoscopic clips WITHOUT endpoints (SCARED, Hamlyn, ...)
    unlabeled_root: str = "data/laparoscopic_unlabeled"
    query_grid: int = 256                   # points to seed per clip when self-labeling
    max_frames: int = 0                     # 0 = full clip


@dataclass
class StudentConfig:
    arch: str = "cotracker3_online"         # the fast student to distill into
    checkpoint: str = ""                    # synthetic-pretrained init
    lr: float = 2e-5
    epochs: int = 10
    batch_clips: int = 1
    huber_delta: float = 4.0                # pixels
    vis_loss_weight: float = 0.5
    out_dir: str = "runs/student_ft"


@dataclass
class PipelineConfig:
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    data: DataConfig = field(default_factory=DataConfig)
    student: StudentConfig = field(default_factory=StudentConfig)
    teachers: List[TeacherConfig] = field(default_factory=lambda: [
        TeacherConfig("mft"),               # pretrained checkpoint ships in the repo; accurate, slow -> teacher only
        TeacherConfig("cotracker3"),        # strong general; ALSO the architecture the student fine-tunes from
        TeacherConfig("locotrack"),         # diversity
        TeacherConfig("mftiq"),             # off-the-shelf did NOT beat MFT in published STIR 2024 results --
                                             # kept for diversity, not because it's presumed stronger than MFT.
                                             # The verifier should discount it wherever it's actually wrong.
        TeacherConfig("bootstapir"),        # matching-based, architecturally distinct from the flow-chaining /
                                             # CoTracker lineage above. Track-On-R's own real-world-ft pipeline
                                             # uses BootsTAPIR specifically, not plain TAPIR -- see teachers.py.
        TeacherConfig("alltracker"),        # another architecturally distinct tracker (dense, not sparse-query)
        # NOTE: LiteTracker is intentionally excluded from the ensemble. It is
        # CoTracker3's own weights run through a different inference procedure
        # (memory buffer + EMA flow init), so it adds no independent signal as
        # a teacher. Its benefit is applied post-hoc, after training
        # -- see student_lt_wrapper.py.
        # Plain TAPIR / TAPNext / BootsTAPNext / LocoTrack's Anthro variant are
        # available via the same _REGISTRY pattern in teachers.py but skipped
        # by default -- same architecture families as what's already included
        # above, added label-generation compute for little extra diversity.
    ])
    pseudo_label_out: str = "data/pseudo_labels"
