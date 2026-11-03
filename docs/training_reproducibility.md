# 训练脚本可复现性指南

本指南说明如何编写符合本项目可复现性与工程规范的训练脚本。

## 必需步骤

### 1. 种子固定
在训练开始前调用 `set_all_seeds(seed)`：
```python
from adasquash.repro import set_all_seeds
set_all_seeds(args.seed)
```

### 2. 生成运行元数据
调用 `collect_run_metadata()` 并保存到 `run_manifest.json`：
```python
from adasquash.repro import (
    collect_run_metadata, save_run_manifest, file_fingerprint,
    execution_identity_mismatches, set_all_seeds
)

# 初始化种子
set_all_seeds(args.seed)

# 收集元数据
run_manifest = collect_run_metadata(seed=args.seed)
run_manifest["hyperparameters"] = vars(args)
run_manifest["data"] = {
    "dataset": args.dataset,
    "dataset_revision": args.dataset_revision,
    # ... 其他数据配置
}
save_run_manifest(Path(args.output_dir) / "run_manifest.json", run_manifest)
```

### 3. 训练流指纹
使用 `TrainStreamFingerprint` 记录训练数据流：
```python
from adasquash.repro import TrainStreamFingerprint

stream_fp = TrainStreamFingerprint(start_step=0)
# 在每次训练 batch 后调用（训练前哈希）
stream_fp.update(x, y)
# 完成后记录到 manifest
run_manifest["data"]["train_stream"] = stream_fp.record()
```

## 双臂实验要求

### 比较前检查执行身份
在结果分析前，必须调用 `execution_identity_mismatches()` 验证：
```python
from adasquash.repro import execution_identity_mismatches

# 加载两个 run 的 manifest
ref_manifest = json.loads(Path("arm_a/run_manifest.json").read_text())
other_manifest = json.loads(Path("arm_b/run_manifest.json").read_text())

identity = execution_identity_mismatches(ref_manifest, other_manifest)
if not identity["comparable"]:
    raise ValueError(f"Runs not comparable: {identity['reason']}")
if not identity["identical"]:
    # 不相同但可比较，检查 stream 指纹
    ...
```

### 比较训练流
验证两臂消费相同的数据窗口：
```python
from adasquash.repro import compare_train_streams

stream_comparison = compare_train_streams(
    ref_manifest["train_stream_fingerprint"],
    other_manifest["train_stream_fingerprint"]
)
if stream_comparison["identical"]:
    print("Training streams identical - paired design valid")
else:
    print(f"Stream mismatch: {stream_comparison['problems']}")
```

## 设计锁定 D1-D6

脚本必须遵守的固定设计：

- **D1**：使用 `F.scaled_dot_product_attention` 作为注意力实现
- **D2**：`_MIN_P = 1e-3`、`_MIN_ALPHA = 1e-4` 小数位固定
- **D3**：所有 fused kernel 必须保持正向闭式 + 反向闭式重算
- **D4**：`QKSquash` 必须使用 `tanh` 压缩
- **D5**：`NORM_REGISTRY` 必须包含 `ln/rmsnorm/dyt/dytc/dytd/dytcd`
- **D6**：缓存键使用 `data_ptr()` 而非 `id()`

## 结果分析

### 必须输出的文件

1. `run_manifest.json` - 完整运行元数据
2. `ckpt_best.pt` - 验证最佳 checkpoint
3. `ckpt_final.pt` - 最终 checkpoint
4. `train.jsonl` - 训练日志

### 可选但推荐的输出

- `sanity_overfit.json` - 过拟合检查报告
- `test_results.json` - 最终测试结果（若 `--final-test-eval`）

## 常见错误

### 错误 1：种子不固定
问题：训练流不确定，结果无法复现。
解决：确保 `set_all_seeds()` 在所有随机操作前调用。

### 错误 2：忽略运行元数据
问题：无法追溯训练上下文。
解决：始终调用 `collect_run_metadata()` 并保存到 `run_manifest.json`。

### 错误 3：双臂比较未验证执行身份
问题：代码/环境变化导致的 PPL 差异被误认为是归一化层效果。
解决：在比较前调用 `execution_identity_mismatches()`。

## 调试技巧

### 使用 `--require-pinned-data-revision`
强制要求数据集和分词器使用具体的 commit 哈希：
```bash
python scripts/train_gpt2.py \
    --dataset "HuggingFaceFW/fineweb-edu" \
    --dataset-revision "abc123..." \
    --tokenizer-revision "def456..." \
    --require-pinned-data-revision
```

### 使用 `--resume` 验证续跑
确保续跑时数据流正确继续：
```bash
# 首次训练
python scripts/train_gpt2.py --seed 42 --output-dir runs/exp1

# 续跑（从最佳 checkpoint 恢复）
python scripts/train_gpt2.py --seed 42 --resume runs/exp1/ckpt_best.pt --output-dir runs/exp1
```