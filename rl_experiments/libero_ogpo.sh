# M3 validation on LIBERO (cf. piRL: GR00T N1.5 few-shot SFT 52.5% -> 89.9% after RL).
# MODEL_PATH: a LIBERO_PANDA few-shot SFT run from submit_euler.sh (examples/LIBERO).
# This is the place to tune the PPO likelihood knobs (see gr00t/rl/README.md).
MODE=train
RUN_NAME=libero_ogpo_seed0
CHAIN=3
MODEL_PATH=/groot/runs/REPLACE_WITH_LIBERO_SFT_RUN
ARGS=(
    --model-path "$MODEL_PATH"
    --embodiment-tag LIBERO_PANDA
    --algo ogpo --seed 0
    --wandb-group libero_validation
    --collect.tasks libero_spatial/0:10
    --collect.execute-steps 8 --critic.horizon 16
    --env.backend libero --env.num-envs 20 --env.max-episode-steps 300
    --num-workers 6
)
