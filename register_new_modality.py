"""GR00T modality config for the G1 wrist-EEF embodiment.

Usage (fine-tuning):
    uv run python gr00t/experiment/launch_finetune.py \\
        --base-model-path nvidia/GR00T-N1.7-3B \\
        --dataset-path /path/to/g1_dataset \\
        --embodiment-tag NEW_EMBODIMENT \\
        --modality-config-path /path/to/this/file/gr00t_modality_config.py \\
        --num-gpus 1 \\
        --output-dir /tmp/g1_wrist_eef \\
        --max-steps 5000 --save-steps 1000 \\
        --global-batch-size 32

The recorded dataset has:
  state  (35-dim): left_wrist_eef_9d(9) + right_wrist_eef_9d(9)
                   + left_arm(7) + right_arm(7) + waist(3)
  action (21-dim): left_wrist_eef_9d(9) + right_wrist_eef_9d(9) + vel_cmd(3)
  video:           observation.images.ego_view  →  "ego_view" key
  language:        annotation.human.action.task_description

After fine-tuning, deploy with chunk_size=G1_WRIST_EEF_CHUNK_SIZE (see below).
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

# Number of future steps the model predicts. Must match --action-horizon at
# eval time and the chunk_size passed to VLAActionManager at deployment.
G1_WRIST_EEF_CHUNK_SIZE: int = 16

mrl_unitree_g1 = {
    # Video: single ego-view camera, current frame only.
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["ego_view"],
    ),
    # State: current wrist EEF poses (FK from Pinocchio) + arm + waist joints.
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=[
            "left_wrist_eef_9d",
            "right_wrist_eef_9d",
            "left_arm",
            "right_arm",
            "waist",
        ],
    ),
    # Action: predict a 16-step chunk of wrist EEF targets + base velocity.
    # RELATIVE means the pipeline computes deltas w.r.t. the current pose,
    # which improves generalisation across different start configurations.
    # XYZ_ROT6D matches the 9D layout used by the G1 FK / IK pipeline.
    # vel_cmd (lin/lat/ang) is NON_EEF with ABSOLUTE representation.
    "action": ModalityConfig(
        delta_indices=list(range(G1_WRIST_EEF_CHUNK_SIZE)),
        modality_keys=[
            "left_wrist_eef_9d",
            "right_wrist_eef_9d",
            "vel_cmd",
        ],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
            ),
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    # Language: task instruction stored as task index in the parquet.
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.action.task_description"],
    ),
}

# Always register custom embodiments under NEW_EMBODIMENT.
register_modality_config(mrl_unitree_g1, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
