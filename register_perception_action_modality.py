"""GR00T modality config for the WRIST-ONLY perception-CONDITIONED action model.

Used by gr00t/experiment/launch_finetune_perception_action.py: the short-range
fine-approach policy that predicts actions conditioned on its own target
perception (heatmap -> conditioning tokens -> DiT).

It sits between the two existing configs:
  - like register_perception_modality.py:  video is ONLY the wrist camera, single
    CURRENT frame (delta_indices=[0]). The heatmap label (target_uv) is defined in
    this camera's pixel frame and the perception processor pushes it through the
    same crop/resize as the image, so the head must see exactly that one image.
  - like register_new_modality.py:  a full 30-step RELATIVE action chunk of
    drill-tip poses, and the proprioceptive state the action head needs.

Why no ego_view here: the perception head requires exactly one image per sample
(it reads the VLM image-token grid as a single spatial map), and this model is the
close-range specialist — the coarse approach stays with the full VLA. Adding the
ego view means generalizing the head to slice one image's tokens out of several;
see PERCEPTION_ACTION_FINETUNE.md.

Auxiliary per-frame labels (heatmap_labels/ sidecar: u, v, visible, label_ok,
depth_z, wall_normal_cam_*) are NOT modalities — they arrive through
VLAStepData.metadata; see LeRobotEpisodeLoader._load_heatmap_labels.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


# Number of future steps the model predicts. MUST match --action-horizon of the
# perception-action launcher and the chunk_size used at deployment.
G1_FINE_APPROACH_CHUNK_SIZE: int = 30

perception_action_unitree_g1 = {
    # Wrist camera only, single current frame — the frame the heatmap label lives in.
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["wrist_view"],
    ),
    # Proprioception: drill-tip pose + both arms, same keys as the full VLA so a
    # VLA checkpoint's state encoder transfers.
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "right_drill_tip_eef_9d",
            "left_arm",
            "right_arm",
        ],
    ),
    # Action: a chunk of RELATIVE drill-tip EEF targets (deltas w.r.t. the current
    # pose, matched by key against `state`).
    "action": ModalityConfig(
        delta_indices=list(range(G1_FINE_APPROACH_CHUNK_SIZE)),
        modality_keys=[
            "right_drill_tip_eef_9d",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
            )
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

register_modality_config(perception_action_unitree_g1, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
