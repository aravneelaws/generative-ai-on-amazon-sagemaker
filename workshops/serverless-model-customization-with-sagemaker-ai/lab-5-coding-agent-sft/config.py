# Base model for this lab.
#
# Must be a JumpStart model that supports serverless SFT customization and can be
# served by the SageMaker LMI/DJL container.
#
# Verified end-to-end in Lab 1: huggingface-reasoning-qwen3-4b
BASE_MODEL_ID = "huggingface-reasoning-qwen3-4b"

# Fixed dataset / resource names used across the notebooks.
#
# The training data is a subset of bigcode/self-oss-instruct-sc2-exec-filter-50k.
# The evaluation data is the HumanEval benchmark (openai_humaneval), which ships a
# clean per-problem unit-test suite, so pass@1 is a real execution measurement.
DATASET_PREFIX = "self-oss-code-sft"

# Training-subset size. 10K keeps the single-epoch job inside a short workshop window.
SUBSET_SIZE = 10_000

# Held-out validation size, carved from the BigCode subset (HumanEval is the test set).
VAL_SIZE = 200
