
# -----------------------------
# PROJECT CONFIGURATION
# Medusa PDC Optimization Study
# -----------------------------

PROJECT_NAME = "Medusa-PDC-Optimization"

DESCRIPTION = """
PDC-based optimization of Medusa candidate verification using:
- OpenMP-style multiprocessing
- SIMD vectorization (NumPy)
- KV-cache cost modeling (FastDecode + NEO inspired)
"""


# -----------------------------
# MEDUSA SETTINGS
# -----------------------------
MEDUSA_NUM_HEADS = 5
MEDUSA_NUM_LAYERS = 1


# -----------------------------
# EXPERIMENT SETTINGS
# -----------------------------
NUM_RUNS = 5

CANDIDATE_SIZES = [10, 20, 40, 80, 160]


# -----------------------------
# PDC OPTIMIZATION FLAGS
# -----------------------------
USE_OPENMP_PARALLELISM = True   # multiprocessing
USE_SIMD_VECTORIZATION = True    # NumPy batch scoring
USE_KV_CACHE_MODEL = True        # memory cost simulation


# OpenMP-style workers
NUM_WORKERS = 4


# -----------------------------
# BASELINE SETTINGS
# -----------------------------
BASELINE_TYPE = "sequential_medusa_like_scoring"
BASELINE_GPU_COUPLED = True


# -----------------------------
# OPTIMIZED SETTINGS
# -----------------------------
OPTIMIZED_TYPE = "cpu_parallel_scoring_pipeline"
DECOPLED_FROM_GPU = True


# -----------------------------
# PDC MAPPING (FOR REPORT + VIVA)
# -----------------------------
PDC_MAPPING = {
    "parallelism": "OpenMP-style multiprocessing over candidate scoring",
    "simd": "NumPy vectorized batch scoring",
    "scheduling": "NEO-inspired batch execution of candidates",
    "memory": "KV-cache cost modeling inspired by FastDecode and NEO"
}
