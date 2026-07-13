"""The training layer is intentionally just thin ms-swift commands."""

from pathlib import Path


ROOT = Path(__file__).parents[1]


def _script(name: str) -> str:
    return (ROOT / "scripts" / name).read_text(encoding="utf-8")


def test_training_scripts_allow_an_explicitly_empty_validation_dataset() -> None:
    for name in (
        "train_qwen3_vl_sft.sh",
        "train_qwen3_vl_grpo.sh",
        "train_qwen3_vl_opsd.sh",
    ):
        script = _script(name)
        assert "VAL_DATA=${VAL_DATA-" in script
        assert 'if [[ -n "$VAL_DATA" ]]' in script


def test_sft_script_is_a_direct_eight_gpu_qwen3_vl_lora_command() -> None:
    script = _script("train_qwen3_vl_sft.sh")

    assert "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" in script
    assert "NPROC_PER_NODE=${NPROC_PER_NODE:-8}" in script
    assert "swift sft" in script
    assert "Qwen/Qwen3-VL-8B-Instruct" in script
    for option in (
        "--dataset",
        "--val_dataset",
        "--tuner_type lora",
        "--freeze_vit true",
        "--deepspeed zero2",
        "--packing true",
        "--padding_free true",
        "--attn_impl flash_attn",
    ):
        assert option in script


def test_rollout_script_uses_the_last_four_gpus() -> None:
    script = _script("start_qwen3_vl_rollout.sh")

    assert "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5,6,7}" in script
    assert "swift rollout" in script
    assert "--vllm_tensor_parallel_size 4" in script
    assert "--vllm_enable_lora true" in script
    assert "--vllm_max_lora_rank" in script


def test_grpo_script_uses_official_server_mode_and_one_adapter() -> None:
    script = _script("train_qwen3_vl_grpo.sh")

    assert "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}" in script
    assert "NPROC_PER_NODE=${NPROC_PER_NODE:-4}" in script
    assert "swift rlhf" in script
    assert "--rlhf_type grpo" in script
    assert "--vllm_mode server" in script
    assert "--vllm_server_host 127.0.0.1" in script
    assert "--vllm_server_port 8000" in script
    assert "--external_plugins plugins/ms_swift_risk_rewards.py" in script
    assert '--adapters "$SFT_ADAPTER"' in script
    assert '--ref_adapters "$SFT_ADAPTER"' in script
    assert "--remove_unused_columns false" in script


def test_opsd_script_is_a_direct_gkd_command() -> None:
    script = _script("train_qwen3_vl_opsd.sh")

    assert "swift rlhf" in script
    assert "--rlhf_type gkd" in script
    assert "--vllm_mode server" in script
    assert "--remove_unused_columns false" in script
