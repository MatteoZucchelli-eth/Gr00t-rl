#!/usr/bin/env bash
# Download every paper listed in INDEX.md into this folder (PDFs are not meant for git).
set -euo pipefail
cd "$(dirname "$0")"

get() { [ -s "$1" ] && pdfinfo "$1" >/dev/null 2>&1 && { echo "skip  $1"; return; }; echo "fetch $1"; curl -sSfL -o "$1" "$2"; }

# User-provided
get MolmoBot2026_LargeScaleSimZeroShot.pdf                 https://arxiv.org/pdf/2603.16861v3
get Korniak2026_RLGeneralistImprovement.pdf      https://arxiv.org/pdf/2611.06905v1
# Base model and VLA architectures
get NVIDIA2026_GR00T2_Whitepaper.pdf               https://research.nvidia.com/labs/gear/gr00t-2/assets/gr00t2_whitepaper.pdf
get Bjorck2025_GR00T_N1.pdf                      https://arxiv.org/pdf/2503.14734
get Black2024_Pi0.pdf                            https://arxiv.org/pdf/2410.24164
get Kim2024_OpenVLA.pdf                          https://arxiv.org/pdf/2406.09246
# Benchmarks
get Liu2023_LIBERO.pdf                           https://arxiv.org/pdf/2306.03310
get Tao2024_ManiSkill3.pdf                       https://arxiv.org/pdf/2410.00425
get Kim2026_MolmoSpaces.pdf                       https://arxiv.org/pdf/2602.11337
get Yu2019_MetaWorld.pdf                         https://arxiv.org/pdf/1910.10897
# RL algorithm
get Schulman2017_PPO.pdf                         https://arxiv.org/pdf/1707.06347
get Schulman2015_GAE.pdf                         https://arxiv.org/pdf/1506.02438
# RL for diffusion / flow policies
get Ren2024_DPPO.pdf                             https://arxiv.org/pdf/2409.00588
get Zhang2025_ReinFlow.pdf                       https://arxiv.org/pdf/2505.22094
get Patil2026_OGPO.pdf                            https://arxiv.org/pdf/2605.03065
get Chen2025_PiRL.pdf                            https://arxiv.org/pdf/2510.25889
# RL fine-tuning of VLAs
get Liu2025_RL4VLA_WhatCanRLBring.pdf            https://arxiv.org/pdf/2505.19789
get Li2025_SimpleVLA-RL.pdf                      https://arxiv.org/pdf/2509.09674
get Zang2025_RLinf-VLA.pdf                       https://arxiv.org/pdf/2510.06710
get PI2025_Pi06star_RECAP.pdf                    https://arxiv.org/pdf/2511.14759
get Kalashnikov2021_MT-Opt.pdf                   https://arxiv.org/pdf/2104.08212
# Does RL transfer? (LLM evidence)
get Chu2025_SFTMemorizesRLGeneralizes.pdf        https://arxiv.org/pdf/2501.17161
get Yue2025_DoesRLIncentivizeReasoning.pdf       https://arxiv.org/pdf/2504.13837
get Shenfeld2025_RLsRazor.pdf                    https://arxiv.org/pdf/2509.04259
# Demo efficiency and evaluation methodology
get Mandlekar2021_robomimic.pdf                  https://arxiv.org/pdf/2108.03298
get Lin2024_DataScalingLawsIL.pdf                https://arxiv.org/pdf/2410.18647
get Agarwal2021_StatisticalPrecipice.pdf         https://arxiv.org/pdf/2108.13264
get KressGazit2024_RobotLearningEmpiricalScience.pdf https://arxiv.org/pdf/2409.09491
