# 可复现性与工程规范改进总结

## 改进内容

### 1. CI Gate 增强 (ci_gate.py)

**新增设计锁定 D7-D8：**

- **D7 (设计文档完整性)**：新增对 `docs/design_notes.md` 中 D1-D6 章节的检查，防止设计决策意外丢失
- **D8 (run_manifest 合同)**：新增对 `repro.py` 中 `save_run_manifest`, `collect_run_metadata`, `execution_identity_mismatches` 函数的检查，确保训练脚本正确集成运行元数据

**新增 Layer 4：覆盖率门禁**

- 添加 `check_coverage()` 函数
- 运行 `coverage run -m pytest` 再 `coverage report --fail-under=80`
- 确保代码覆盖率 >= 80%

### 2. 设计文档更新 (docs/design_notes.md)

**新增 D7-D8 设计决策：**

- D7: 设计文档完整性 - 文档必须包含 D1-D6 章节
- D8: run_manifest.json 写入合同 - 训练脚本必须生成 run_manifest

**更新目录：**

- 添加 `adasquash/repro.py` - D8（run_manifest 合同）
- 添加 `tests/test_reproducibility.py` - D8（self-test via CLI）

### 3. 新增指南文档 (docs/training_reproducibility.md)

创建完整的训练脚本可复现性指南，包含：

- 必需步骤：种子固定、运行元数据、训练流指纹
- 双臂实验要求：执行身份检查、训练流比较
- 设计锁定 D1-D6 说明
- 必需输出文件列表
- 常见错误及解决方法
- 调试技巧

## 可复现性合同

### 运行元数据 (run_manifest.json)

训练脚本必须包含：

```python
from adasquash.repro import (
    collect_run_metadata, save_run_manifest, set_all_seeds
)

set_all_seeds(args.seed)
run_manifest = collect_run_metadata(seed=args.seed)
run_manifest["hyperparameters"] = vars(args)
save_run_manifest(Path(args.output_dir) / "run_manifest.json", run_manifest)
```

### 双臂比较

跨臂分析必须在比较前执行：

```python
from adasquash.repro import (
    execution_identity_mismatches, compare_train_streams
)

identity = execution_identity_mismatches(manifest_a, manifest_b)
if not identity["comparable"]:
    raise ValueError(f"Not comparable: {identity['reason']}")

streams_match = compare_train_streams(stream_a, stream_b)
```

## CI 流程

```
Layer 1: 语法检查 (所有 .py 文件解析)
Layer 2: 设计锁定 (D1-D8，源码检查)
Layer 3: Pytest 套件 (540 项测试)
Layer 4: 覆盖率 (>= 80%)
```

## 使用方法

运行 CI 门禁：

```bash
python3 ci_gate.py                    # 完整检查
python3 ci_gate.py --syntax-only    # 仅语法检查
```

运行可复现性自测：

```bash
python3 -m adasquash.repro --self-test
```