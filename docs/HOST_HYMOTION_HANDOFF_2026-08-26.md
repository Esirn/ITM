# ITM / HY-Motion 大显存宿主机交接

> 更新日期：2026-09-08
>
> 分支：`master`；远端：`https://github.com/Esirn/ITM.git`

## 使用前由用户手动填写

下面信息无法从当前机器可靠推断。将本文交给宿主机Codex前，请只修改本节表格的
“宿主机实际值”列；也可以不改文件，直接把实际值附在提示词后面。宿主机Codex必须先
验证路径，不能根据示例路径盲目修改全项目。

| 项目 | 当前机器参考值 | 宿主机实际值（手填） |
|---|---|---|
| ITM仓库目录 | `/home/a200/0proj/ITM` | `/home/4T-2/syx0011/0proj/ITM` |
| HY-Motion源码/资产目录 | `/home/a200/mount/a40/relatedworks/HY-Motion-1.0` | `/home/4T-2/group_motion/relatedworks/HY-Motion-1.0/` |
| HumanML3D处理后数据根目录 | `/home/a200/0proj/datasets/all` | `/home/4T-2/group_motion/relatedworks/HumanML3D/HumanML3D/` |
| MDM兼容Mean/Std目录 | `/home/a200/mount/a40/relatedworks/mdm/HumanML3D/HumanML3D` | `/home/4T-2/group_motion/relatedworks/mdm/HumanML3D/HumanML3D/` |
| AMASS根目录 | `/home/a200/mount/a40/datasets/AMASS` | `/home/4T-2/group_motion/datasets/AMASS` |
| SMPL资产目录 | `/home/a200/mount/a40/datasets/SMPL` | `/home/4T-2/group_motion/datasets/SMPL` |
| OpenAI CLIP源码包 | `/home/a200/0proj/datasets/mdm-need/CLIP-main.zip` | `/home/4T-2/group_motion/datasets/mdm-need/CLIP-main.zip` |
| 首选训练GPU | 本机曾用`cuda:1` | `根据实时情况使用` |
| 非Git资产传输方式 | 本机直接复制/重新生成 | `大文件尽量用软链接/挂载` |
| Conda安装位置 | `/home/a200/miniconda3` | `/home/syx0011/miniconda3/` |

以下内容通常**不需要手动改**：模型结构、joint IDs、checkpoint SHA-256、
环境版本、训练门槛和测试协议。若宿主机HY-Motion版本或checkpoint不同，应停止并报告，
不要直接改这些核验值来“让检查通过”。

## 目标与当前边界

宿主机下一阶段目标是实现 HY-Motion 1.0 Lite 的 Text+IMU 控制预筛。当前仅完成官方
Text-only复现、统一backbone接口、22关节全局旋转和joint-mounted虚拟IMU导出；尚未实现
HY-Motion IMU residual adapter，也没有任何HY-Motion Text+IMU实验结果。

MDM Stage-2b仍是论文主模型。HY-Motion只有在paired IMU稳定优于shuffled IMU、文本质量
基本保留且动作质量不退化时，才升级为竞争主模型。

## Git同步前置条件

宿主机clone/pull后先确认分支、同步状态和工作树：

```bash
git status --short
git pull --ff-only
```

工作树应为空；若宿主机已有修改，不要覆盖或reset，先检查来源。

Git只同步代码和文档。`.gitignore`排除了整个`outputs/`，因此checkpoint、manifest、IMU
cache、采样NPZ/GIF和干净runtime快照不会随Git出现。

## 宿主机目录约定

以下是当前机器路径，不是对宿主机的强制要求。优先使用上方手填的实际路径；宿主机不同
时先验证，再修改`configs/paths.toml`，然后重建含绝对路径的manifest：

```text
/home/a200/0proj/ITM                         项目仓库
/home/a200/0proj/datasets/all                HumanML3D texts/joints/joint_vecs
/home/a200/0proj/datasets/AMASS              AMASS（或修改配置）
/home/a200/mount/a40/relatedworks/HY-Motion-1.0
                                               HY-Motion源码、权重和资产
```

不要原样照抄下列路径。将shell变量改成上方手填值后执行：

```bash
ITM_ROOT=/宿主机/ITM路径
HYMOTION_ROOT=/宿主机/HY-Motion-1.0路径
HUMANML_ROOT=/宿主机/HumanML3D处理后数据路径
MDM_STATS_ROOT=/宿主机/MDM兼容MeanStd路径
AMASS_ROOT=/宿主机/AMASS路径

test -d "$ITM_ROOT/.git"
test -d "$HYMOTION_ROOT/.git"
test -f "$HYMOTION_ROOT/ckpts/tencent/HY-Motion-1.0-Lite/latest.ckpt"
test -d "$HUMANML_ROOT/new_joint_vecs"
test -f "$MDM_STATS_ROOT/Mean.npy"
test -f "$MDM_STATS_ROOT/Std.npy"
test -d "$AMASS_ROOT"
```

任一检查失败时先定位宿主机真实目录，再改`configs/paths.toml`；不要修改sshfs源目录。

### HumanML3D目录实测差异（2026-09-08）

已通过sshfs挂载视图实测两个宿主机候选目录：

| 宿主机路径 | motion/text完整性 | 统计量 | 用途结论 |
|---|---|---|---|
| `/home/4T-2/group_motion/relatedworks/HumanML3D/HumanML3D/` | 29,228条标准split样本完整 | `Mean/Std`与当前ITM略有差异；无`mean_motion/std_motion` | 推荐作为HumanML数据根目录 |
| `/home/4T-2/group_motion/relatedworks/mdm/HumanML3D/HumanML3D/` | 当前视图中`new_joint_vecs/new_joints`各只有`012314.npy` | `Mean/Std`与当前ITM逐字节一致 | 只作为MDM统计量来源，不作为完整数据根目录 |

本机`/home/a200/0proj/datasets/all/`有35,958个motion文件，其中额外6,730个文件名以`04`
开头；它们均不在`train.txt`、`val.txt`或`test.txt`中。三个split文件在完整宿主机目录与
本机逐字节一致，总计29,228条，所以缺少`04****`不影响标准split训练、采样或评价。

完整宿主机目录的`Mean.npy/Std.npy`与当前ITM版本shape均为`(263,)`，但Mean最大绝对差约
`9.71e-4`，Std最大绝对差约`1.03e-4`。为严格复现现有MDM Stage-2b，MDM训练和采样的
`--mean/--std`应显式指向第二个目录中的兼容文件。HY-Motion使用自身
`HY-Motion-1.0/stats/`，不依赖这两份HumanML统计量。

`mean_motion.npy/std_motion.npy`是66维关节位置统计量，当前主要由MotionLab脚本使用。
缺失它们不阻塞HY-Motion Text-only或HY-Motion adapter预筛；若还要复跑MotionLab，应从
本机同步这两个小文件，或按原MotionLab协议重新计算并核验。

## 必需HY-Motion资产

Text-only和冻结主干IMU训练至少需要：

```text
HY-Motion-1.0/.git/
HY-Motion-1.0/hymotion/                         仅用于审计，不直接导入修改工作树
HY-Motion-1.0/stats/
HY-Motion-1.0/scripts/gradio/static/assets/dump_wooden/
HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0-Lite/config.yml
HY-Motion-1.0/ckpts/tencent/HY-Motion-1.0-Lite/latest.ckpt
HY-Motion-1.0/ckpts/Qwen3-8B/
HY-Motion-1.0/ckpts/clip-vit-large-patch14/
```

Lite checkpoint SHA-256应为：

```text
d83f118f8d74db76249db86dcf9982a8229f43ef4e9fa11f683019d6230dd486
```

挂载工作树的`hymotion/network/text_encoders/text_encoder.py`被修改为跳过Qwen并返回
零context，不能直接用于正式实验。必须在宿主机运行clean workspace提取命令。

关闭prompt rewrite和duration estimation时不需要约57 GB的`Text2MotionPrompter/`；本轮
不需要HY-Motion Full的4.17 GB checkpoint。

## Conda环境

项目使用两个隔离环境：

- `itm`：数据、评价、渲染、MDM和通用测试，Transformers 4.49.0、NumPy 2.2.5。
- `hymotion`：HY-Motion推理和后续训练，Transformers 4.53.3、NumPy 1.26.4。

不要在`itm`里升级Transformers到4.53，也不要在`hymotion`中升级NumPy到2.x。

### 创建itm

```bash
cd "$ITM_ROOT"
conda env create -f environment.yml
conda install -n itm pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -n itm "mkl<2025" "intel-openmp<2025"
conda run -n itm pip install --no-build-isolation chumpy==0.70
conda run -n itm pip install --no-build-isolation \
  /path/to/CLIP-main.zip
conda run -n itm pip install -e .
```

本机的CLIP本地包原路径为`/home/a200/0proj/datasets/mdm-need/CLIP-main.zip`。宿主机没有
该文件时，可使用官方OpenAI CLIP仓库构建的同一源码包，但应记录来源和版本。

### 创建hymotion

推荐直接使用本仓库新增的环境文件：

```bash
cd "$ITM_ROOT"
conda env create -f environment-hymotion.yml
```

若`bitsandbytes`在宿主机CUDA环境安装失败，可以暂时跳过；当前Lite fp32/bf16 smoke未调用
它。其余版本不要随意升级，尤其是`transformers==4.53.3`与`numpy==1.26.4`。

## 非Git实验资产

若宿主机能访问本机文件，可复制以下最小集合：

```text
outputs/manifests_full/                         约9.8 MB，但内含绝对路径
outputs/standard_imu/                           约2.4 GB，正式train/val/test虚拟IMU缓存
outputs/mdm_control/stage1_full_pilot_v2.pt     约97 MB
outputs/mdm_control/stage2b_balanced_b.pt       约97 MB
```

HY-Motion开发本身不依赖MDM checkpoint，但公平复测和论文对比需要Stage-2b。若宿主机路径
不同，不要直接使用复制来的manifest；重建manifest和standard IMU cache。复制后至少抽查：

```bash
head -1 outputs/manifests_full/train.jsonl
head -1 outputs/manifests_full/train_standard_imu.jsonl
```

确认其中`text_path`、`joint_vec_path`、`imu_path`和`source_path`均在宿主机存在。

## 首次启动验证

先设置实际路径和空闲GPU；后续命令均使用这些变量：

```bash
export ITM_ROOT=/宿主机/ITM路径
export HYMOTION_ROOT=/宿主机/HY-Motion-1.0路径
export DEVICE=cuda:0
cd "$ITM_ROOT"

conda run -n itm python -c \
  "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
conda run -n hymotion python -c \
  "import torch, transformers, numpy; print(torch.__version__, transformers.__version__, numpy.__version__)"

conda run -n itm pytest -q

conda run -n itm python scripts/prepare_hymotion_workspace.py \
  --source "$HYMOTION_ROOT" \
  --output outputs/hymotion/runtime_root

conda run --no-capture-output -n hymotion python scripts/sample_hymotion_text.py \
  --hymotion-root "$HYMOTION_ROOT" \
  --runtime-root outputs/hymotion/runtime_root --model lite \
  --text "a person walks forward while swinging both arms" \
  --frames 60 --seed 1234 --device "$DEVICE" \
  --output outputs/hymotion/text_only_smoke/walk_swing_60.npz

conda run -n itm python scripts/export_hymotion_virtual_imu.py \
  --results outputs/hymotion/text_only_smoke/walk_swing_60.npz \
  --sensor-config head \
  --output outputs/hymotion/text_only_smoke/walk_swing_head_virtual_imu.npz
```

预期Lite smoke核心采样约2秒；本机RTX 4090峰值PyTorch分配显存约19.6 GB。宿主机数值
不要求完全相同，但shape应为motion `[1,60,201]`、rot6d `[1,60,22,6]`、global rotation
`[1,60,22,3,3]`，且输出不得含NaN/Inf。

## 宿主机实现顺序

1. 冻结HY-Motion MotionFlow和Qwen/CLIP文本编码器。
2. 实现每帧六槽12D IMU投影、sensor embedding、帧内sensor attention和temporal encoder。
3. 输出1024维control tokens，并为18个MMDiT block分别设置零初始化residual projection。
4. 6个double-stream block只修改`motion_feat`；12个single-stream block只修改拼接序列的
   motion prefix，绝不能修改text suffix。
5. 支持head joint 15和wrists joints 20/21，保留sensor mask与独立条件dropout。
6. 先跑16条backward smoke，再跑512 train/128 val、2 epochs预筛。
7. 必须比较paired、shuffled、zero和Text-only；同seed、文本、长度和初始噪声。
8. 只有paired相对shuffled改善率达到70%、文本质量没有明显崩坏、jerk不恶化时，才启动
   7009条全量训练。

不要第一步就解冻HY-Motion，不要引入Text/IMU以外模态，不要修改官方挂载仓库，也不要
把joint acceleration proxy写成真实设备acceleration error。

## 当前可复核结果

- `docs/HYMOTION_MIGRATION.md`：当前复现、表示、定性结果与虚拟IMU边界。
- `outputs/hymotion/text_only_fixed_prompts/`：走/跑/坐/转/跳的NPZ、JSON和GIF，不随Git。
- `outputs/hymotion/text_only_smoke/`：60帧smoke及head/wrists虚拟IMU，不随Git。
- 当前全量测试：96 passed（2026-08-26，9条PyTorch Transformer性能warning）。

坐下样例语义清楚，快跑和转弯有可见变化，当前jump seed较弱。成功与失败样例都应保留。
