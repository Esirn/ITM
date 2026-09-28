# 02 V 增量软件验收命令

工作目录 `/home/a200/0proj/ITM`；读取方法提交 `0b3642252dbb1c9295d9ee5e16a8eec726ec6fb8`。仅 CPU 验收，不执行 V、GPU 预检或创建实际运行授权。测试的微型模型前向与临时假授权不属于真实模型作业。

独立执行与01相同的135项相关依赖选择，结果保存到02目录；不覆盖01日志：

```bash
bash docs/itm_rework/protocol/b0_state_diagnostic_acceptance_20260928_v1/regression.sh
```

独立临时目录重建 implementation，并从原始NPZ/NumPy插值/SO(3)参考核对所有48套输入、398帧统计及48保存状态：

```bash
OMP_NUM_THREADS=2 PYTHONPATH=src /home/a200/miniconda3/envs/itm/bin/python scripts/accept_b0_state_diagnostic.py --output docs/itm_rework/protocol/b0_state_diagnostic_acceptance_20260928_v1/acceptance.json --prepare-dir /tmp/itm02_state_diagnostic_acceptance_20260928_v1 > docs/itm_rework/protocol/b0_state_diagnostic_acceptance_20260928_v1/logs/acceptance.txt 2>&1
```

该次已完成，重放时须使用新的 `--output` 与 `--prepare-dir`；现有目录/验收结果拒绝覆盖。临时准备目录仅有 implementation.json，字节/hash与01一致。审计过程禁止 Module前向、CUDA设备API、预检、监督器和子进程；禁止操作触发0次。允许读取既有R checkpoint张量到CPU检查final200及来源，不构造真实模型或新参数、不保存新checkpoint。

来源检查核对1656个绑定文件、完整包含1640个冻结来源。全部48套分支输入 normalized/reference 最大差0；保存的x_t/f10/f11/mask精确一致，target从原motion.npy与已绑定均值/标准差独立重建一致。无需为本次V重做旧完整动作评价；所有旧负例和冻结解释边界保持不变。
