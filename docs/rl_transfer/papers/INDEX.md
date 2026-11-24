# Literature index: does RL on some task families make a VLA cheaper to adapt to new ones?

**Research question.** Take GR00T, fine-tune it with PPO on 5 task families in simulation. Then supervised-fine-tune it on 3
**held-out** families with N demos. Does it reach the same success as the non-RL model with substantially fewer demos?

**Why it matters.** If yes: a result labs care about, and a sales demo for an RL-environment library. If the gains stay on the
trained families: the product is "make task X work" (Enact-style / eval), not an environment library.

**Success criterion (to pre-register, see `../QUESTIONS.md` Q8).** On the held-out families, the RL'd model (arm C) reaches the
SFT-only control's (arm B) success at N demos using ≤ ½·N demos, with non-overlapping 95% CIs. Arm B must have seen the same
5 families via SFT. Otherwise we only measure sim/embodiment adaptation.

**Selection rule.** Well-recognised papers only, one per question we actually have to answer. Each entry says which decision it informs.

**State of the evidence (2026-11-23).** RL on VLAs reliably helps the **trained** tasks and their OOD variants (RL4VLA, SimpleVLA-RL, πRL).
**Korniak et al. 2026 is the closest prior work.** Multi-task RL on π0.5 *does* improve unseen tasks zero-shot, but only when (i) the method
learns a **critic** (filtered SFT on successes does not generalize), (ii) training covers **many tasks with diverse manipulation strategies**
(gains appear at ~16–144 tasks), and (iii) held-out tasks **resemble** training ones. **On LIBERO-90 it does not emerge at ≤64 tasks.**
There's also weak evidence in SimpleVLA-RL (Fig. 4). The LLM priors split: Chu 2025 (RL generalizes) vs Yue 2025 (RL only sharpens).
**No paper in this set measures demo-efficiency of subsequent SFT on held-out task families after RL.** That gap is still our experiment.
Korniak's results make the design choices sharper: which simulator, how many families, how similar the held-out families are, and critic vs plain PPO.

---

## 0. User-provided

### Deshpande, Guru, Hendrix, Jauhri et al. (Ai2) 2026 — *MolmoB0T: Large-Scale Simulation Enables Zero-Shot Manipulation* (CoRL 2026) · arXiv 2603.16861
`MolmoBot2026_LargeScaleSimZeroShot.pdf`
MolmoBot-Engine procedurally generates expert demos with scripted planners plus retries, in 200k+ MolmoSpaces scenes with heavy domain randomization.
MolmoBot-Data has 1.7M trajectories (Franka FR3 in DROID config, plus the RB-Y1 mobile manipulator). Three policies are trained **only on sim, SFT only, no RL**:
MolmoBot (Molmo2-4B VLM + DiT flow-matching head, chunk 16 / execute 8, frozen SigLIP), MolmoBot-Pi0 (exact π0 architecture, to isolate data),
and MolmoBot-SPOC (108M, discrete actions, explicitly "amenable to RL fine-tuning", RL left to future work).
Real-world zero-shot tabletop pick-and-place: **79.2% vs 39.2% for π0.5-DROID**. Same-architecture MolmoBot-Pi0 gets 46.5%, so the gain is the data.
Data ablations (Fig. 5): success scales predictably with the number of demos (10k→50k). Object-class diversity helps in sim but saturates in the real world.
Environment count barely matters at fixed 50k demos. App. C.9: LIBERO and SIMPLER are **structurally poor at evaluating generalists**,
because they were built for in-distribution policies trained on their own demos. They introduce LIBERO-DROID / SIMPLER-DROID.
**Use:**
(a) The strongest counter-thesis to "environments are valuable as RL gyms": here, environments are valuable as **demo generators** for plain SFT, with no RL.
That gives a third answer to the disagreement with your friend, and arguably a stronger product story. It's worth an arm (or a follow-up) comparing
"PPO on 5 families" against "SFT on N procedurally generated demos from the same 5 families" at matched compute.
(b) A warning for Q1: claims about generalization on LIBERO need care, because its tasks share scenes and objects.
(c) Its architecture choices (frozen vision encoder, 16/8 chunking, flow head) match GR00T's closely, so its recipes transfer.

### Korniak, Bagatella, Römer et al. (ETH LAS / Krause, Berkeley, CMU, PI) 2026 — *Reinforcement Learning Enables Scaling Generalist Robot Policy Improvement* (preprint, under review) · arXiv 2611.06905
`Korniak2026_RLGeneralistImprovement.pdf` · code: github.com/lasgroup/generalist-improvement
**The closest prior work to our hypothesis.** Asks whether RL on *many* tasks at once improves a VLA on tasks **not trained on**.
Base: **π0.5** (droid / libero checkpoints), chosen so the base has ≈30% zero-shot success on training tasks. Batch-online protocol: M=10 rounds × K=20 episodes/task.
Compares three mechanisms:
- **FSFT**: filtered SFT on successful rollouts.
- **BoN**: frozen policy, a learned Q-critic re-ranks N=32 sampled action chunks.
- **OGPO**: an off-policy critic plus a PPO-like update on the denoising process (critic-guided policy extraction). AWR is an ablation.

Envs: MolmoSpaces pick suite (18 objects × ~55 scenes, splits of 1→144 tasks = k objects × k scenes) and LIBERO-90 (splits of 1→64 tasks, 25 unseen tasks).
Findings:
(1) All methods improve training tasks. OGPO/BoN beat FSFT by +43–50 / +15–21 pp multi-task. Single-task experts are still slightly better per task,
but one 16-task run needs **2.9× fewer gradient steps** than 16 single-task runs.
(2) **Unseen tasks:** single-task RL *forgets* on unseen objects. As the number of training tasks grows, critic-based methods flip from forgetting to
generalization: **up to +58 pp on unseen scenes and +39 pp on unseen objects** over base. FSFT generalizes much less.
(3) Diversity of **objects** (that is, of grasp strategies) drives transfer to both new objects and new scenes. Diversity of scenes alone barely transfers to new objects.
Transfer is strongest to objects that **resemble** training objects (spatula→knife, box→bowl).
(4) **LIBERO (App. C.1): "at the scale we train on (≤64 tasks), generalization does not emerge"**, because LIBERO tasks differ too much. FSFT forgets held-out tasks, and BoN shows minimal forgetting.
(5) Recipe: critic = BroNet residual MLP on mean-pooled VLM prefix + proprio + action chunk, Q/V ensemble with min, **TD + regression** (distributional critics scale badly multi-task).
Full-VLA FSFT beats action-expert-only in-distribution (+12 pp) but costs −10.6 pp on unseen objects, against −2.2 pp for expert-only.
Compute: 4×GH200 nodes. 2–5 h per single-task run, up to ~200 h for the largest multi-task split. 3 seeds, rliable bootstrap CIs, 32 eval episodes/task.
**Use:**
(a) Raises the bar for "novel": zero-shot transfer from multi-task RL is now shown (on π0.5, MolmoSpaces). Our contribution has to be the **demo-efficiency
of subsequent SFT on held-out families**, a GR00T replication, and/or the product framing.
(b) **Q1:** strong evidence that a LIBERO family split will likely come out negative for reasons unrelated to our hypothesis. Favour MolmoSpaces, or ManiSkill families with a shared skill structure.
(c) **Q2:** 5 families may be too few. Families must vary the *manipulation strategy*, and held-out families should be "near" the trained ones (pre-register a similarity notion).
(d) **Q5:** a learned **Q-critic** matters more than the policy-update rule. Consider OGPO/BoN-style critic-guided updates instead of, or alongside, on-policy PPO.
BoN with a frozen policy is a cheap, strong control arm.
(e) Q4: their FSFT is a natural "RL without a critic" control. (f) Q9: compute calibration for Euler.
(g) Same institution (ETH, Krause group). Worth contacting the authors before building: they have the pipeline, and possibly views on the demo-efficiency question.

## 1. Base model and architecture family

### Bjorck et al. (NVIDIA) 2025 — *GR00T N1: An Open Foundation Model for Generalist Humanoid Robots* · arXiv 2503.14734
`Bjorck2025_GR00T_N1.pdf`
Dual-system VLA: a VLM (System 2, Eagle in N1/N1.5; Cosmos-Reason2 / Qwen3-VL in the N1.7 fork) feeds a **DiT action head trained with flow matching**
(System 1) that outputs continuous action chunks. Embodiment-specific state/action encoders allow cross-embodiment training. Pretrained on a "data pyramid":
real robot data, simulation/synthetic data, and human video with latent actions. Reports data-efficient post-training on sim and real humanoid benchmarks.
**Use:** what we are fine-tuning. The flow head has no tractable log-likelihood, which drives Q5. Embodiment tags matter for LIBERO (`LIBERO_PANDA`).

### NVIDIA GEAR 2026 — *GR00T 2: Scaling Robot Foundation Models for Generalization, Dexterity, and Adaptation* (whitepaper, Nov 2026, 135 pp.) · no arXiv ID yet
`NVIDIA2026_GR00T2_Whitepaper.pdf` · research.nvidia.com/labs/gear/gr00t-2 (project page says code in December 2026)
A **world-action model**, not a VLA. A single 2B transformer initialised from Cosmos-Predict2.5-2B (video diffusion) jointly flow-matches **future video latents and the action chunk**.
Wan 2.1 VAE for video, frozen Cosmos-Reason1 for language, embodiment-specific state/action MLPs. Inputs: 0.8 s of history at 5 fps from up to 4 cameras.
Inference: 1–2 denoising steps for video, 2–4 for actions, ~153 ms per chunk on an RTX 5090.
Pretraining: ~400k h egocentric human video, ~10k h UMI, ~23.5k h teleop. Main embodiments: YAM, Dexter, Tianji+Sharpa (no DROID/Franka in the main results).
**RL (§3.3, §4.1, App. F):** exploration via Coefficients-Preserving Sampling noise, **only in action denoising** (video stays deterministic). A GRPO-style group-relative update
with a baseline that remembers past returns, so all-success or all-fail groups still give signal. The recipe was found by an "AutoRL" agent over ~35 days in sim.
Real-world precision tasks (cable/charger insertion, pipe fitting) go from 40–76% after SFT to **88–98% with GRPO + self-imitation**. Residual RL also helps.
App. H.3 claims WAMs **consume simulation data better than VLAs**.
**Use:** Q6. A possible base model once weights are public. It comes with a native RL recipe (GRPO + CPS, not PPO). Costs: rollouts also denoise video,
and Franka support is unclear.

### Black et al. (Physical Intelligence) 2024 — *π0: A Vision-Language-Action Flow Model for General Robot Control* · arXiv 2410.24164
`Black2024_Pi0.pdf`
PaliGemma-3B VLM plus a separate flow-matching "action expert" that outputs 50-step action chunks. Pretrain on a large cross-embodiment mixture, then post-train
on curated task data. The defining flow-VLA paper, and the architecture used by πRL and MolmoBot-Pi0.
**Use:** most RL-for-flow-VLA results are reported on π0/π0.5. Treat π0 numbers as the closest proxy for what GR00T will do.

### Kim et al. 2024 — *OpenVLA: An Open-Source Vision-Language-Action Model* (CoRL 2024) · arXiv 2406.09246
`Kim2024_OpenVLA.pdf`
7B Llama-2 VLA with DINOv2+SigLIP vision and **discretized action tokens**, trained on 970k Open-X trajectories. Shows LoRA fine-tuning to new setups.
**Use:** the backbone of most RL-on-VLA studies (RL4VLA, SimpleVLA-RL via OpenVLA-OFT). Their findings come from an autoregressive token policy,
so check whether each finding should carry over to a flow head before relying on it.

## 2. Simulators and task-split protocol

### Liu et al. 2023 — *LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning* (NeurIPS D&B) · arXiv 2306.03310
`Liu2023_LIBERO.pdf`
130 language-conditioned Franka tasks in 4 suites. Spatial, Object and Goal each vary one kind of knowledge. LIBERO-100 = LIBERO-90 + LIBERO-10 (long horizon).
**50 human teleop demos per task** and fixed initial-state sets. It's explicitly framed around **transfer of declarative vs procedural knowledge** between tasks.
**Use:** the default environment (Q1). LIBERO-90's 90 tasks give enough material to define skill families (Q2), and the 50 demos/task bound our demo sweep (Q8).
The GR00T fork already has the SFT recipe and eval harness. Caveat: see MolmoBot App. C.9 and the fixed init states (Q7).

### Tao et al. 2024 — *ManiSkill3: GPU Parallelized Robot Simulation and Rendering for Generalizable Embodied AI* (RSS 2025) · arXiv 2410.00425
`Tao2024_ManiSkill3.pdf`
GPU-parallel physics **and rendering** (SAPIEN/PhysX), giving orders of magnitude more env-steps per second than CPU sims, which makes visual RL cheap.
Many manipulation tasks, motion-planning demo generation, and PPO baselines included.
**Use:** the fallback for Q1 if LIBERO rollout throughput on Euler is the bottleneck. RL4VLA, RLinf and πRL all run PPO here.

### Kim et al. (Ai2) 2026 — *MolmoSpaces* (RSS 2026) · arXiv 2602.11337
`Kim2026_MolmoSpaces.pdf`
Open simulation ecosystem: 230k+ procedurally generated indoor scenes, 130k+ object models, ~42M annotated grasps (Objaverse + THOR), with articulation-tested assets.
MolmoSpaces-Bench has **eight base tasks: navigate-to, pick, pick-and-place, pick-and-place-next-to, pick-and-place-color, open, close, open-door**.
Non-articulated tasks and open/close run on a **Franka FR3 in DROID configuration**. Open-door runs on the RB-Y1. Controlled variation axes (objects, scenes, prompts),
and strong sim-to-real correlation for zero-shot evaluation (R = 0.96).
**Use:** our simulator (Q1, decided). It constrains the families (Q2): there's **no insertion, rotation or peg task**. The verbs are pick/place variants plus open/close,
so families have to come from object/grasp classes and articulation types. The Franka-DROID setup matches GR00T N1.7's `OXE_DROID` embodiment tag.

### Yu et al. 2019 — *Meta-World: A Benchmark and Evaluation for Multi-Task and Meta RL* (CoRL 2019) · arXiv 1910.10897
`Yu2019_MetaWorld.pdf`
50 Sawyer tasks with the canonical **train-task / held-out-task splits** (ML10: 10 train, 5 test; ML45: 45 train, 5 test) and multi-task suites (MT10/MT50).
The standard reference for "learn on some task families, measure adaptation on unseen ones". Also shows that held-out-task transfer is hard and split-dependent.
**Use:** the template for how we define and justify the 5/3 family split (Q2), and the precedent reviewers will expect us to cite.

## 3. RL algorithm

### Schulman et al. 2017 — *Proximal Policy Optimization Algorithms* · arXiv 1707.06347
`Schulman2017_PPO.pdf`
Clipped surrogate objective, multiple epochs of minibatch updates on on-policy data.
**Use:** the algorithm. RL4VLA finds PPO beats GRPO and DPO for VLAs.

### Schulman et al. 2015 — *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (ICLR 2016) · arXiv 1506.02438
`Schulman2015_GAE.pdf`
GAE(λ): the bias–variance knob for advantages from a learned value function.
**Use:** the advantage estimator inside PPO. Relevant to the sparse-reward, long-horizon setting (Q7) and to critic warm-up (πRL App. H).

## 4. Making diffusion / flow policies RL-trainable (Q5)

### Ren et al. 2024 — *Diffusion Policy Policy Optimization (DPPO)* (ICLR 2025) · arXiv 2409.00588
`Ren2024_DPPO.pdf`
Treats the denoising chain as an inner MDP nested inside the environment MDP (a "two-layer MDP"), so each denoising step is a Gaussian with a tractable log-prob, and runs PPO over it.
Fine-tunes pretrained diffusion policies on robomimic and FurnITURE-Bench, with sim-to-real. Shows that RL fine-tuning of a demo-pretrained diffusion policy is stable and
explores in a structured way, staying near the data manifold.
**Use:** the foundational recipe for PPO on a generative action head. πRL's Flow-SDE is its flow-matching descendant.

### Zhang et al. 2025 — *ReinFlow: Fine-tuning Flow Matching Policy with Online RL* (NeurIPS 2025) · arXiv 2505.22094
`Zhang2025_ReinFlow.pdf`
Injects **learnable noise** into the deterministic flow path, turning it into a discrete-time Markov process with exact likelihoods. Works with very few, even one,
denoising step. Gets +40% net success on manipulation with Shortcut-model policies, at a fraction of DPPO's wall time.
**Use:** the theory behind option (a) in Q5. This is "Flow-Noise" in πRL.

### Patil, …, Levine, Simchowitz 2026 — *OGPO: Sample Efficient Full-Finetuning of Generative Control Policies* (ICML 2026) · arXiv 2605.03065
`Patil2026_OGPO.pdf`
**Our chosen RL method.** Decouples DPPO's two-layer MDP:
- **Environment level:** an off-policy Q-critic trained with TD on a replay buffer, so all data is reused, failures included.
- **Denoising level:** for a replay state, sample a group of G action chunks through SDE denoising, score each with the frozen target critic,
  and use the group-relative advantage Â_i = Q(s,a_i) − mean_j Q(s,a_j). Then apply a PPO-clipped update to the denoising-chain likelihood ratio against a slow EMA policy.

Stabilizers: a success buffer with a BC loss, two-sided conservative advantages (early critics are underfit), and Q-variance reduction (TD target averaged over N actor samples).
OGPO+ adds Q-based Best-of-N at test time. Claims SOTA on multi-task, high-precision insertion and dexterous tasks, and that it is the only method to fine-tune
**poorly initialised BC policies** to near-full success with no expert data in the buffer.
Korniak 2026 adapted it to π0.5 (BroNet critic ensemble of 10, min over a random 2, BoN with N=32 during collection).
**Use:** Q5, decided. The "poorly initialised" claim is what makes the base-GR00T + RL comparison (Q4) plausible even at low zero-shot success.

### Chen, Liu, Zhang et al. 2025 — *πRL: Online RL Fine-tuning for Flow-based VLA Models* · arXiv 2510.25889
`Chen2025_PiRL.pdf`
Two ways to get log-likelihoods from a flow VLA: **Flow-Noise** (learnable noise network, exact likelihood) and **Flow-SDE** (ODE→SDE with the same marginals, two-layer MDP).
Then PPO. On LIBERO, few-shot SFT (40 demos per suite) plus RL takes π0 from 57.6% to 97.6% and π0.5 from 77.1% to 98.3% (Flow-Noise; Flow-SDE is within ~1.5 pts). On ManiSkill it shows gains in-distribution and OOD.
**App. H runs PPO on GR00T N1.5 on LIBERO: few-shot SFT 52.5% → 89.9% average with Flow-SDE.** The VLM is frozen, only the action expert is trained,
the critic is attached to the action head, **dropout is replaced by identity** (dropout breaks the PPO ratio), and the critic needs warm-up (performance dips early).
**Use:** the most directly reusable paper. It answers most of Q5 and de-risks Q6. It's built on RLinf.

## 5. RL fine-tuning of VLAs: what it buys

### Liu, Gao et al. 2025 — *What Can RL Bring to VLA Generalization? An Empirical Study (RL4VLA)* (NeurIPS 2025) · arXiv 2505.19789
`Liu2025_RL4VLA_WhatCanRLBring.pdf`
Controlled SFT-vs-RL comparison on OpenVLA in ManiSkill, along visual, semantic and execution OOD axes. RL, **especially PPO**, beats SFT on semantic and execution
generalization, and matches it on visual robustness. PPO beats DPO and GRPO. Recipe: **shared actor-critic backbone, warm-up with SFT, minimal PPO epochs**.
**Use:** the evidence for using PPO (not GRPO) and for the warm start in Q3. Its OOD axes are within-task variations, not new task families,
which is exactly the step our experiment adds.

### Li et al. 2025 — *SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning* · arXiv 2509.09674
`Li2025_SimpleVLA-RL.pdf`
veRL-based RL (GRPO-style, sparse binary reward) for OpenVLA-OFT. On LIBERO, **one demo per task + RL reaches 96.9%, vs 17.3% after one-demo SFT and 99.1% for full SFT + RL**.
Discovers novel behaviours ("pushcut"). **Fig. 4 (closest prior to our hypothesis):** train on 9 tasks of a suite, then track the 10th, unseen task. RL training
steadily improves unseen-task success, while SFT on the same 9 tasks overfits and often drops unseen tasks to 0%.
**Use:** (a) evidence that RL can substitute for demos *on the trained tasks*. (b) The only direct prior for cross-task transfer, but zero-shot,
within one suite, and on unseen tasks that share scenes and objects. Our demo-efficiency curve on held-out *families* is the stronger, untested version.

### Zang et al. 2025 — *RLinf-VLA: A Unified and Efficient Framework for RL of VLA Models* · arXiv 2510.06710
`Zang2025_RLinf-VLA.pdf`
A framework for VLA RL: unified interface across VLAs, algorithms (PPO, GRPO) and simulators (ManiSkill, LIBERO, RoboTwin), and flexible GPU allocation
between rendering, inference and training (1.6–1.9× speedup on ManiSkill). Reports 98.1% on all 130 LIBERO tasks and 97.7% on 25 ManiSkill tasks, and distills best practices.
The paper itself evaluates OpenVLA(-OFT) and π0. **GR00T support comes from πRL (N1.5), built on RLinf. Check N1.7 compatibility.**
**Use:** the build-vs-adopt decision (Q6), and its system design for the Euler pipeline (Q9).

### Physical Intelligence 2025 — *π\*0.6: a VLA That Learns From Experience (RECAP)* · arXiv 2511.14759
`PI2025_Pi06star_RECAP.pdf`
Real-world RL at scale: **advantage-conditioned policies** trained on demos, on-policy rollouts and expert interventions. Offline-RL pretraining, then
on-robot specialization. Laundry, box assembly, espresso. On hard tasks it more than doubles throughput and roughly halves failures.
**Use:** shows the leading lab treats "RL after SFT" as core, which matters for the "labs would care" framing. Also an alternative to PPO
(no log-likelihood needed for the flow head) if PPO on GR00T is unstable.

### Kalashnikov et al. (Google) 2021 — *MT-Opt: Continuous Multi-Task Robotic RL at Scale* · arXiv 2104.08212
`Kalashnikov2021_MT-Opt.pdf`
Multi-task offline/online RL across 12 tasks on 7 real robots with shared data and a success-detector reward. Shows that sharing experience across tasks improves
rare tasks, and that a multi-task policy **fine-tunes to new tasks faster** than training from scratch.
**Use:** pre-VLA robotics precedent for "multi-task RL produces a better initialization for new tasks", and an honest contrast in scale and setting.

## 6. Does RL transfer beyond trained tasks? (the core scientific prior, from LLMs)

### Chu, Zhai et al. 2025 — *SFT Memorizes, RL Generalizes* (ICML 2025) · arXiv 2501.17161
`Chu2025_SFTMemorizesRLGeneralizes.pdf`
Controlled study on a card-arithmetic game and visual navigation (V-IRL) with rule and visual variants. RL generalizes to unseen variants while SFT memorizes the training rules.
**SFT is still needed first**, to stabilize output format before RL works.
**Use:** the main prior for a positive result, and support for the SFT warm start (Q3). Its variants are within-task. Ours are new task families.

### Yue et al. 2025 — *Does RL Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?* (NeurIPS 2025) · arXiv 2504.13837
`Yue2025_DoesRLIncentivizeReasoning.pdf`
With pass@k at large k, base models match or beat their RLVR-trained versions. RL **sharpens sampling toward solutions the base model already contains**
and narrows coverage. It does not add new capabilities.
**Use:** the main prior for a *negative* result (gains only on trained families) and the sharpest form of your friend's position.
It also suggests a diagnostic: compare best-of-k success of base vs RL'd on held-out families. If RL only sharpens, it shouldn't reduce demos needed for new families.

### Shenfeld, Pari & Agrawal 2025 — *RL's Razor: Why Online RL Forgets Less* · arXiv 2509.04259
`Shenfeld2025_RLsRazor.pdf`
Across LLM and robotics fine-tuning, forgetting is predicted by the **KL between fine-tuned and base policy on the new task**. On-policy RL implicitly finds
low-KL solutions, so it preserves prior capabilities better than SFT at equal new-task performance.
**Use:** explains *why* RL might help held-out families: it doesn't erode the pretrained prior. Suggests logging KL to base during PPO and
including "how much did held-out zero-shot success drop" as a secondary metric.

## 7. Demo efficiency and evaluation methodology (Q8)

### Mandlekar et al. 2021 — *What Matters in Learning from Offline Human Demonstrations for Robot Manipulation (robomimic)* (CoRL 2021) · arXiv 2108.03298
`Mandlekar2021_robomimic.pdf`
A systematic study of learning from demos: dataset size, demonstrator quality (proficient vs multi-human), observation choices, and **checkpoint selection**
(the best vs last checkpoint differ widely, and offline validation loss is a poor proxy for success).
**Use:** how to run the SFT demo sweep fairly: fixed training budget per point, evaluate by rollouts not validation loss, and the same selection rule for every arm.

### Lin, Hu et al. 2024 — *Data Scaling Laws in Imitation Learning for Robotic Manipulation* (ICLR 2025) · arXiv 2410.18647
`Lin2024_DataScalingLawsIL.pdf`
Large real-world study. Generalization to new environments and objects follows **power laws in environment/object diversity**, while demos per environment saturate quickly.
Gives a practical data-collection recipe.
**Use:** the functional form for our success-vs-demos curves (fit a power law, compare exponents and offsets across arms). It also reminds us that
demo *count* is not the only axis. Hold diversity fixed across arms.

### Agarwal et al. 2021 — *Deep RL at the Edge of the Statistical Precipice (rliable)* (NeurIPS 2021, outstanding paper) · arXiv 2108.13264
`Agarwal2021_StatisticalPrecipice.pdf`
With few seeds, point estimates mislead. Use stratified-bootstrap CIs, IQM and performance profiles.
**Use:** how we report the demo curves and the "≤ ½ demos" claim with uncertainty across 3 seeds × tasks (Q8).

### Kress-Gazit et al. (TRI) 2024 — *Robot Learning as an Empirical Science: Best Practices for Policy Evaluation* · arXiv 2409.09491
`KressGazit2024_RobotLearningEmpiricalScience.pdf`
Practical protocol for evaluating robot policies: define success precisely, enough trials, blind/A-B comparisons, report CIs and failure modes,
document the initial-condition distribution.
**Use:** a checklist for the eval protocol, and for the sales-demo version of the result, which has to survive a skeptical lab.

---

## Decisions → papers

| Question (`../QUESTIONS.md`) | Read first |
|---|---|
| Q1 simulator | **Korniak 2026 App. C.1**, LIBERO, ManiSkill3, MolmoBot App. C.9, RLinf-VLA |
| Q2 family split | **Korniak 2026 §4.3**, Meta-World, LIBERO, SimpleVLA-RL Fig. 4 |
| Q3 warm start | RL4VLA, Chu 2025, πRL, Korniak 2026 (base needs ≈30% zero-shot) |
| Q4 control arms | Korniak 2026 (FSFT / BoN controls), SimpleVLA-RL Fig. 4, MolmoBot (sim-SFT alternative), Yue 2025 |
| Q5 flow head → PPO | πRL (App. H = GR00T), **Korniak 2026 (critic-guided > on-policy SFT)**, ReinFlow, DPPO |
| Q6 build vs adopt | RLinf-VLA, πRL |
| Q7 reward / episodes | SimpleVLA-RL (sparse binary works), GAE, πRL (critic warm-up) |
| Q8 protocol | robomimic, Lin 2024, rliable, Kress-Gazit 2024 |
| Q9 Euler | Korniak 2026 App. D.5 (4×GH200, ≤200 h), RLinf-VLA (resource allocation), ManiSkill3 (throughput) |
| Q10 kill switch | RL's Razor (KL), Yue 2025 (pass@k diagnostic) |

## Considered and left out
VLA-RL (Lu 2025, OpenVLA + process reward model; RL4VLA and SimpleVLA-RL cover it) · Flow-GRPO (Liu 2025, image generation; its ODE→SDE trick is in πRL) ·
RT-1/RT-2/Open-X (background, not decision-relevant) · OpenVLA-OFT (SFT recipe; GR00T has its own) · DeepSeek-R1 / GRPO (RL4VLA shows PPO > GRPO for VLAs) ·
MAML / meta-RL methods (we do plain SFT adaptation, not meta-learning).

## Referenced, no PDF
- **Sutton & Barto, *Reinforcement Learning: An Introduction*** (2nd ed.). Background only.

## Open questions for later
- Should procedurally generated sim demos (MolmoBot-style) be a fourth arm? It tests "environments as data generators" vs "environments as RL gyms" directly.
- Is there post-2025 work measuring post-RL adaptation efficiency on held-out tasks for VLAs? Korniak 2026 covers zero-shot transfer only. Re-search before writing anything up.
- OGPO (Patil et al. 2026) is the policy-extraction method Korniak uses. Add the paper once we decide on critic-guided updates (Q5).
