"""Same perception architecture and supervision, using only the current head image."""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig


head_perception_unitree_g1 = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["ego_view"]),
    "state": ModalityConfig(delta_indices=[0], modality_keys=["right_drill_tip_eef_9d"]),
    "action": ModalityConfig(delta_indices=[0], modality_keys=["right_drill_tip_eef_9d"]),
    "language": ModalityConfig(
        delta_indices=[0], modality_keys=["annotation.human.action.task_description"]
    ),
}
register_modality_config(head_perception_unitree_g1, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
