# Forge Neo Remi 使用指南

[English](usage.md) | [简体中文](usage.zh-CN.md)

面向用户的参考：界面的每个页面、Anima 功能、安装、启动参数、数据存储位置、键盘快捷键、从 Gradio 界面迁移、安全与故障排查。[项目 README](../../README.zh-CN.md) 是简短的概览；其他文档见[文档索引](README.md)。

**目录：** [Remi WebUI 界面](#remi-webui-界面) · [Anima 集成](#anima-集成) · [快速开始](#快速开始) · [启动参数](#启动参数) · [数据存储位置](#数据存储位置) · [键盘快捷键](#键盘快捷键) · [从 Gradio 界面迁移](#从-gradio-界面迁移) · [内置扩展](#内置扩展) · [安全与远程访问](#安全与远程访问) · [故障排查](#故障排查) · [兼容性说明](#兼容性说明)

## Remi WebUI 界面

浏览器应用（`webui/`）只通过有文档的 API 与服务器通信（[`/api/v2`](api.md)，外加一个用于进度和实时预览的 WebSocket）；服务器从不渲染 HTML。下面逐页说明。

**整体框架。** 左侧是窄导航栏（手机上为底部栏和“更多”面板），通往各个页面；扩展页面排在内置页面与“设置”之间。顶栏显示页面标题、**模型栏**（界面预设、带架构/大小与自动识别的格式及能力徽章如 `Diffusers`、`GGUF`、`INT8-ConvRot`、`FP8`、`NVFP4`、`Turbo`、`RGBA`、`Control` 的 checkpoint、VAE / 文本编码器、“低精度扩散”）、**Worker 状态胶囊**（显示进程内 / 独立子进程 / 多 GPU 池 / MLX Sidecar 状态）、你固定的 ⚡ **快捷设置**、**模型网络**按钮、**命令面板**（Ctrl/⌘+K：可搜索所有页面、命令和快捷键）、连接状态指示和**外观**菜单（浅色 / 深色 / 跟随系统、宽松或紧凑密度、强调色、语言）。生成任务运行时，顶栏下方有一条进度线。

**生成（文生图与图生图）。** 同一个视图，顶栏可在文生图 / 图生图之间切换，两者各有独立的表单。宽屏下分为三栏，可调整宽度、可折叠：

- *参数*：顶部是运行控件（生成、加入队列、中止、跳过、跨模型参数 **Recipes（配方）** 保存与应用菜单，以及包含无限生成、连续生成多次和完成通知的菜单）；表单上方在检测到参数不匹配时会显示事前 **工作流建议** 芯片，点击即可一键修正（如 Turbo 切 8 步 / CFG 1.0、Kromoss 切 8 步 DMD2、RGBA 透明图切 PNG 输出）；下面是正向和反向提示词，使用带语法高亮的编辑器（注意力权重、LoRA 标签、提示词调度、通配符、括号检查、标签补全、token 计数、样式，以及支持附带参考图/BBox 草案与 Diff 预览的 **提示词扩写（PUP）**）；再往下按你配置的顺序排列参数卡片：采样器、尺寸（宽高比预设、粘贴“W x H”）、引导、高分辨率修复、种子、批次，以及每个扩展面板（可折叠的分区）。扩展通过 `requires_capabilities` 声明适用模型能力，不适用于当前模型的面板（如加载 FLUX.3 或 Qwen-Image-2.1 时的 Anima 专用面板）会自动收纳到“当前模型不适用”抽屉并给出原因；可搜索的脚本列表也支持“仅显示已修改”筛选和逐面板重置。值旁边的小圆点表示已修改，点击即可恢复默认值。你输入的值会按浏览器保存。
- *舞台*：任务运行时显示实时预览（步数、批次、剩余时间、队列位置），之后显示所选结果及操作：下载、保存、复制参数、复用种子、使用全部参数、发送到图生图 / 局部重绘 / **参考图（`ImageStitch`）** / 后期处理。参考图列表按当前模型自动标注槽位标签（Qwen-Image-2.1 为 `<image_1>`、FLUX.3 为 `ref_image_0`、Fibo 2.0 为 `ref_0`），支持点击插入提示词和拖拽排序。在图生图中，舞台就是源图的**图像编辑器**：画笔、橡皮擦、填充、矩形和套索选区、取色器、涂鸦与重绘蒙版图层、撤销/重做、缩放与平移、压感、**画布原地 Staging**（在局部重绘选区原地生成 1–4 张候选图并左右切换审阅、一键应用或丢弃），以及显示缩放框和“仅蒙版区域”的叠加层。
- *历史*：本次会话的结果，以虚拟化缩略图条显示；全屏查看器支持缩放、平移、平铺预览、**RGBA 四色底板**切换（棋盘格 / 纯黑 / 纯白 / 绿幕，快捷键 `B`）和手柄操作。

宽度小于 72 rem 时历史栏变为缩略图条；在手机上舞台排在最前，运行控件固定在屏幕底部。把图片拖到提示词上即可读取其生成参数。

**后期处理。** 对单张图片、一批图片、文件夹或视频进行放大、面部修复和其他后处理操作，并提供前后对比滑块。

**图片信息。** 拖入、粘贴或选择图片，查看其生成参数和所有元数据（文本块、EXIF、PNG 块），并把参数或图片发送到其他页面。

**图库。** 为输出文件夹中的所有图片建立索引：搜索提示词、文件名或模型；按表单、文件夹、模型、**项目看板（Boards）**、日期范围和星标筛选；两端对齐或瀑布流布局；多选后可加星标、加入看板、将整个看板批量送入参考图、打包下载、**两张对比**（像素滑块/并排视图 + 结构化 **参数 Diff** 抽屉，高亮提示词、Seed、采样器、CFG、步数、LoRA 与扩展差异，支持一键复制差异到表单）或删除。查看器与详情面板显示**资产血缘**（来源任务、重试、img2img/inpaint/放大衍生链）、生成参数与 RGBA 底板切换，并可在文件管理器中定位文件。

**模型网络。** 一个浏览器统一管理 LoRA / LyCORIS、嵌入、checkpoint、VAE 和文本编码器，可作为整页或提示词旁的抽屉打开（Alt+N）：文件夹树、架构筛选（“与当前预设兼容”）、排序、卡片和列表视图、预览（图片或视频）、元数据与训练标签、条目用户元数据编辑器（描述、激活文本、推荐权重、checkpoint 的首选 VAE / 文本编码器——点击其卡片时一并选中等）、从生成结果设置预览图，以及重新扫描文件夹的刷新按钮（同时更新模型栏）。数千个条目也能流畅滚动。

**队列。** 正在运行的任务及其实时预览，按运行顺序排列的待运行任务（拖动或用键盘调整顺序，高 / 普通 / 低优先级，复制、编辑、取消），暂停 / 继续，以及所有任务的可搜索历史（含输出、星标和“再次排队”）。同时内置**非节点线性多阶段管线**（`提示词扩写 → 生成 → 精修 → 后处理`：支持保存模板，修改后段参数重跑时自动命中并复用前段未变更阶段的缓存产物）与 **Recipes（配方）** 管理（增删改查、JSON 导入导出与一键应用）。可导入旧 Agent Scheduler 和 Prompt Queue 的任务。

**设置。** 所有设置集中在一个可搜索的虚拟化列表中，带分类、“已修改” / “未保存” / “已固定”筛选、逐项重置、导出与导入（也支持 A1111 的 `config.json`），以及服务器工具：重新加载界面、重启、卸载或重新加载模型、查看内存中的权重、为所有 checkpoint 计算哈希、系统信息和许可证（附网页界面构建产出的 `THIRD-PARTY-NOTICES.txt`：其包含的每个库及其许可证全文）。若某项设置保存了类型不对的值（手动编辑的 `config.json`、扩展的赋值；`/sdapi/v1/options` 会像设置页一样检查修改的值，见 api.md）、固定选项中没有的值（可输入任意值的列表没有固定选项）或 null，会按默认值读取，服务器启动时把它重置为默认值并在日志中警告；服务器运行期间（由扩展）写入的这类值，上游代码和扩展在下次启动前按原样读取。超出滑块范围的数字会保留：Remi 按范围内最近的一端读取，上游代码和扩展读取保存的数字。

**扩展。** 已安装扩展（启用、禁用、检查更新、更新、卸载）、**健康诊断（Diagnostics）**（展示各扩展的 `remi-extension.json` 清单兼容性、加载耗时、最近 hook 警告/错误、隔离执行的 action 错误/超时及 Worker 状态）、扩展索引（默认是第三方维护的 [Forge Neo 索引](https://github.com/eduardoabreu81/forge-neo-extensions)，安装前请自行确认扩展可靠；可搜索、按标签筛选，并可只看近期有更新的扩展“2025+”）、从 git 网址安装，以及可恢复的配置备份。

**扩展页面。** 扩展可以添加自己的页面：Remi 自带 **JoyCaption**（图片描述，单张或整个文件夹，带进度与取消）、**Prompt Workshop** 和 **JSON Builder**。

JSON Builder 在画布上用方框排布 **MOSS-Image Layout**、**FLUX.3 Image Klein BBox** 以及 **Bria Fibo 2.0 VGL 2.0**（切换至 VGL 模式时会显示 CC BY-NC 4.0 非商业许可提示）结构化提示词：拖动方框移动位置、拖动右下角调整大小（方向键微调选中方框，Shift 每次 16 像素），滚轮缩放、中键拖动平移。每个项目有自己的颜色，方框边框和列表序号共用。修改画布尺寸时，方框保持原有的相对位置和大小。布局较长时可以展开项目列表占满侧栏，画布上的 ? 按钮有简短说明。

描述或文字字段可在**字段编辑器**中编辑——双击字段或点铅笔按钮：与生成区相同的提示词编辑器，带补全，另有风格选择器，"应用风格"只把加了风格的文本写回该字段。画布左上角按钮可在方框下垫图作绘制参考（本地文件或选中的会话结果，不会保存或发送）。

"发送到文生图/图生图"把编译后的提示词（JSON 一行，"Extra"单独一行）和尺寸写入对应标签页；"载入提示词…"粘贴读回，"从文生图 / 图生图载入"直接读取该标签页当前的提示词和尺寸。

**提示词工具。** 每个提示词下方都有：标签视图（可重新排序、调整权重、禁用或删除的标签块）、快捷标签组、提示词历史、按收藏夹分组的收藏、通过你配置的服务进行的可选翻译，以及在使用 OpenAI 兼容服务时的“AI 撰写提示词”（由大语言模型根据任意语言的描述撰写或修改提示词）。在你选择翻译服务之前，不会向任何地方发送内容。

**API。** 同一个服务器提供 [`/api/v2`](api.md)（界面所用的 API：命名参数、校验、任务、WebSocket 事件），加上 `--api` 时还提供兼容 A1111 的 `/sdapi/v1`。交互式文档位于 `/docs`。

## Anima 集成

| 领域 | 内置能力 |
| --- | --- |
| 采样 | Beta57、Flow Euler、UniPC2、PC3、SPEED（实验：前段半分辨率；切换 sigma 可在设置中调整）、ER SDE、Dynamic Shift |
| 引导 | NAG、Momentum Guidance、Skimmed CFG、SMC、CFG-Zero*、APG、TCFG、FDG、DCW、CNS、CFG 生效区间（只在模型时间窗口内执行 CFG：起止值是模型去噪时间的比例，经 `percent_to_sigma` 换算为 sigma，不是步数比例；窗口外若无其他 hook 读取，则跳过负向推理）、CLIP Modulation guidance（默认首次使用时下载固定版本的 adapter 与 CLIP-L，也可使用本地文件） |
| 空间控制 | Regional Conditioning、FreeFuse 多 LoRA 路由、Artist Scheduled Mixer |
| 多样性 | DAVE（前段 DC 衰减，提升同一提示词不同 seed 的构图多样性） |
| 提示词遵循 | Cross-Attention Boost（高噪声前段放大正向提示词的 cross-attention 并做范数匹配，改善弱标签与多主体绑定） |
| LoRA 工作流 | 分层权重、阶段调度、LoRA 在 2B/2.9B/3.8B（28/40/52 层）之间通用、提示词缩放 |
| 稳定性 | 功能冲突解析、Hires Guard、生成元数据、安全回退 |
| 加速 | Spectrum 预测，以及 Anima per-block `torch.compile`（首次重新编译后，改变的分辨率或 batch size 复用同一个动态形状图；batch 1（CFG 1，或 CFG 区间外的步）和参考帧仍各自编译独立的图） |
| 低比特加载 | **低精度扩散**（Diffusion in Low Bits）中的 `float8-e4m3fn (Anima scaled)`（按张量缩放的 FP8，显存与普通 FP8 相同、精度更高）和 `int8 (Anima ConvRot)`（INT8 矩阵乘；需要 CUDA 13 或 `--enable-triton-backend`，SM 8.0+） |

组合实验性方法前，请先阅读 [Remi Anima 生成与参数指南](../remi-anima-generation-guide.zh-CN.md)。

### Anima 2.9B 与 3.8B checkpoint

**28/40/52 层架构兼容：** 自动识别原始 Anima（28 层）、2.9B（40 层）和 3.8B（52 层）的 DiT 结构。在更深的模型上，分层权重、阶段调度器的 block 权重和 FreeFuse 的采集层仍按 28 层布局解释。

| 模型 | 文件 | 预设 |
|---|---|---|
| Anima 2B（preview、base、aesthetic、turbo） | checkpoint 放在 `models/Stable-diffusion`；`qwen_image_vae` 和 `qwen_3_06b_base` 作为 VAE / 文本编码器模块 | Anima 预设 |
| [Anima-2.9B](https://huggingface.co/Gazingstars123/Anima-2.9B) preview-v1 | 同上 | Anima-2.9B（Euler、SGM Uniform、30 步、CFG 4） |
| [Anima-3.8B](https://huggingface.co/lylogummy/Anima-3.8B) v1.1 | 同上，**另加 `qwen35_4b.safetensors`**（放在 `models/text_encoder`，作为第三个模块选中） | Anima-3.8B（Res Multistep、Beta、32 步、CFG 5） |

**Anima-3.8B。** v1.1 checkpoint 内置 Semantic Connector（v2）：它包在模型自带的 LLM adapter 外面，每个去噪步都根据 Qwen3.5 4B 编码器第 7、15、23、31 层的隐状态重写文本条件。Remi 原生运行它：connector 留在扩散模型里，Qwen3.5 编码器（FP8 权重，按文件原样保留）对提示词编码一次，context 每一步重新计算。没有选中 `qwen35_4b` 时，3.8B 的生成会停止并提示所需文件名；在其他模型上选中它只会记一条警告。与模型的参考工作流一致，负面提示词只使用 Anima 自身的编码器；**设置 → Stable Diffusion → [Anima 3.8B] 负面提示词也使用语义连接器** 可以让负面提示词也走 connector。NegPiP 和 MultiDiffusion 作用于 connector 的输出；Regional 与 Artist Mixer 的附加提示词只由 Anima 自身的编码器编码（主提示词仍经过 connector）。不支持 v1 预览版单独的 progressive adapter、MXFP8 和 GGUF 版 3.8B。

**LoRA 跨深度使用。** 在一种深度上训练的 LoRA 会被重映射到另一种深度，并写入生成信息（`Lora remap: name 28→40`）。LoRA 的训练深度先看文件元数据，再看其中的块编号；猜错时（例如只训练了前段块的 40 层 LoRA 会被读成 28 层），可以在标签里写明：`<lora:name:1:src=40>`。在更深的模型上，**设置 → 附加网络 → 更深的 Anima 模型上的 LoRA：插入块** 决定插入块是使用其复制来源块的 LoRA（`copy`，默认），还是不使用（`skip`）。用于更浅的模型时，插入块上的权重会被丢弃。完全无法套用的 LoRA 会列在任务警告里，不会被悄悄忽略。作用在 LLM adapter 上的 LoRA 权重也会同步到 3.8B connector 使用的那份 adapter。

**控制。** 对 Anima，ControlNet 模型列表只列出能在其 DiT 上运行的模型：ControlNet-LLLite（与 LoRA 一样跨深度重映射）和 control-LoRA。

- **Control-LoRA**（[Anima-Control-Pose](https://huggingface.co/Claquasse/Anima-Control-Pose)，把 `anima_pose_preview2.safetensors` 放进 `models/ControlNet`）：给单元一张骨架图（黑底 OpenPose 渲染图，或使用 openpose 预处理器），再选中该模型。控制图经 VAE 编码后加到 DiT 的 patch embedding 上，强度由单元的**权重**决定（0 为无控制，1 跟随骨架，最高 2），并受起止步数限制。与它一起训练的 LoRA 以 1.0 在整个采样过程中作用于 DiT，并在提示词编码前作用于 LLM adapter，因此起止步数和权重不影响这部分。在 1024 px 左右、权重 1 到 1.5 时跟随效果最好，腿部比举起的手臂更容易跟上；它是用自家的细骨架训练的（COCO 身体关键点、白色关节，加上脸部和手部的点，黑底），没有脸部点的骨架可能被画成背影。模型卡建议使用详细的提示词；其权重仅限非商业用途。

### DiT 模型家族、ControlNet-Union 与推理加速

除 Anima 外，Remi 的声明式 `ModelCapability` 层还完整集成：

- **`Fun-Controlnet-Union`（`DiTControlPatch`）：** 单文件零门控 `control_blocks` 控制支持 **Qwen-Image-2.1-Fun-Controlnet-Union**（16 层偶数层、64/129 通道结构控制与重绘）和 **Z-Image-Turbo-Fun-Controlnet-Union**（16/33 通道），ControlNet 模型列表会按当前加载的 DiT 家族自动过滤。
- **Z-Image（Base / Turbo / Omni / Edit）** 与 **ERNIE-Image：** 支持 Diffusers/分片目录加载、敏感层量化保护、跨步文本/RoPE 缓存、per-block `torch.compile` 与 Spectrum 接缝。
- **Bria Fibo 2.0：** 支持 VGL 2.0 结构化 JSON 编辑（与 JSON Builder 及 PUP 联动）和 4 步 Edit Turbo 工作流建议（注意：Fibo 2.0 权重为 CC BY-NC 4.0 非商业许可）。
- **MOSS-Image / Kromoss：** 内置 8 步 DMD2 LoRA 工作流建议（`steps=8, cfg=1.0`）与 Layout JSON 编辑。
- **Spectrum 预测模式：** 在默认 **Chebyshev** 预测器之外，Spectrum 提供可选的 **TaylorSeer**（高阶有限差分外推）与 **MagCache**（幅度衰减自适应跳步缓存）模式。
- **Apple Silicon MPS 与推理基准：** 内置有界 MPS 反量化权重缓存（`backend/operations_mixed_precision.py`）和统一内存预算感知（`backend/memory_management.py`）；可运行 `python tools/perf/bench_inference.py --synthetic`（或 `--real`）对 CUDA、MPS 或 CPU 合成场景进行吞吐、显存和 100 任务队列稳定性基准测试（详见 [inference-performance.md](../research/inference-performance.md)）。

## 快速开始

### 环境要求

- [Git](https://git-scm.com/downloads)
- [Python 3.13](https://www.python.org/downloads/) 或 [uv](https://docs.astral.sh/uv/)
- 受支持的 PyTorch 设备；NVIDIA GPU 是测试最充分的选择（Apple Silicon 可通过 MPS 运行）
- 网页界面需要：[Node.js](https://nodejs.org/) 22.18+ 或 24.11+（每次更新后构建一次，见[网页界面的构建](#网页界面的构建)）。已安装 pnpm 时使用 pnpm；否则启动器通过 Node.js 24 及更早版本自带的 Corepack 或 `npx` 运行它。
- 可选：用于视频导出的 [FFmpeg](https://ffmpeg.org/)

### 安装与运行

```bash
git clone --branch remi https://github.com/beautifulrem/forge-neo-remi.git forge-neo-remi
cd forge-neo-remi
```

**Windows：** 运行 `webui-user.bat`。推荐使用 `uv`：先创建环境，再在 `webui-user.bat` 的 `COMMANDLINE_ARGS` 中添加 `--uv`：

```bat
uv venv venv --python 3.13 --seed
```

**Linux 和 macOS：** 运行 `./webui.sh`。新安装会优先使用 `python3.13`；若不可用，则由 `uv` 查找或安装 Python 3.13 并创建虚拟环境。已有环境会直接复用，不会删除或重建。在 macOS 上，启动器从 PyPI 安装原生 `torch` 和 `torchvision`，不使用 CUDA wheel。个人设置写在仓库根目录的 `webui.settings.sh` 中（Git 会忽略该文件）：

```bash
export PYTHON="/path/to/python3.13"
export COMMANDLINE_ARGS="--uv"
# 可选：覆盖自动选择的 PyTorch 安装命令
# export TORCH_COMMAND="pip install torch torchvision"
```

首次启动会安装 Python 依赖、构建网页界面，并输出 `Running on local URL:  http://127.0.0.1:7860/`；在浏览器中打开即可（或使用 `--autolaunch` 自动打开）。把 checkpoint 和配套模块放进 `models/` 中对应的子目录（`Stable-diffusion`、`VAE`、`text_encoder`、`Lora` 等），然后在模型栏中选择。WebUI 运行期间新增的文件，在模型网络浏览器（Alt+N）对应标签页中点击刷新按钮后即可出现。

平台依赖请参考上游 [Unix 安装说明](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Unix)。Apple Silicon 可以使用 PyTorch MPS，但 CUDA 专用 attention 和编译后端不可用，生成速度通常也低于独立显卡。

### 网页界面的构建

界面是从 `webui/` 构建到 `webui/dist` 的静态应用。启动器（`launch.py`，由 `webui.sh` / `webui-user.bat` 调用）会在 `webui/dist` 缺失或比源码旧时自动构建：即首次克隆后，以及每次改动了 `webui/` 的更新之后。它使用找到的 pnpm（已安装的 pnpm，其次 Corepack，最后 `npx pnpm@12`；无论哪一种，都会运行 `webui/package.json` 固定的那个 pnpm 版本）执行 `pnpm install --frozen-lockfile` 和 `pnpm build`，并为构建结果打上标记，源码未变时不会重复构建。

机器上没有 Node.js 时：

- 已有的构建会照常使用，并提示它可能已过期。
- 完全没有构建时服务器仍会启动（API 可用），浏览器会显示一个说明如何构建界面的页面。
- 要在没有 Node.js 的机器上运行 Remi，可在任意一台装有 Node.js 的机器上构建一次（产物是纯静态文件，与操作系统无关），再把目录复制过去：

  ```bash
  cd webui
  pnpm install --frozen-lockfile   # 或：npx pnpm@12 install --frozen-lockfile
  pnpm build                       # 生成 webui/dist
  ```

  把 `webui/dist` 复制到目标安装的相同位置。每次更新改动了 `webui/` 后，都需要重新构建并复制。

设置环境变量 `REMI_SKIP_WEBUI_BUILD=1` 可关闭自动构建（`0`、`false`、`no`、`off` 或空值不关闭）；`--skip-install` 和 `--skip-prepare-environment` 也会跳过构建。

### Docker（NVIDIA 或 AMD GPU）

`docker/Dockerfile` 用当前检出的代码构建镜像（构建上下文是仓库根目录）：网页界面在 Node.js 阶段构建，服务器镜像包含 Python 3.13、启动器使用的 PyTorch 版本和 `requirements.txt`，不含 Gradio。全部用户数据放在同一个卷 `/home/remi/data` 中：镜像把 `REMI_DATA_DIR` 设为这个路径，入口脚本再把它作为 `--data-dir` 传给启动器（`launch.py` 本身不读取 `REMI_DATA_DIR`）。详见 [docker/README.md](../../docker/README.md)。

```bash
docker build --build-arg REMI_REVISION="$(git rev-parse HEAD)" -f docker/Dockerfile -t remi-webui .
docker run --gpus all -p 127.0.0.1:7860:7860 -v remi-data:/home/remi/data remi-webui
```

AMD GPU 使用 `docker/Dockerfile.rocm`，流程相同，只是换成 ROCm 版 PyTorch：

```bash
docker build --build-arg REMI_REVISION="$(git rev-parse HEAD)" -f docker/Dockerfile.rocm -t remi-webui-rocm .
docker run --device /dev/kfd --device /dev/dri --group-add video -p 127.0.0.1:7860:7860 -v remi-data:/home/remi/data remi-webui-rocm
```

容器内的服务监听所有网卡，默认没有登录，所以上面的命令只把端口发布到本机。要让其他机器访问，先用 `--auth-file` 设置登录（见 [docker/README.md](../../docker/README.md#run)），并阅读[安全与远程访问](#安全与远程访问)。由于服务监听所有网卡，扩展页面无法在容器里安装或更新扩展：把扩展放进卷中的 `extensions/` 目录后重启容器，或在设置登录的同时加上 `--enable-insecure-extension-access`。

### 首次 Anima 生成建议

1. 在模型栏中选择受支持的 Anima checkpoint 及其文本编码器和 VAE。
2. 在生成页中打开 **Anima Remi 预设** 面板。
3. 应用 **Official Base/Aesthetic — Safe (Recommended)**。
4. 先关闭可选引导、空间控制、高分辨率修复和编译，生成一次。
5. 每次只增加一种高级功能，并使用相同 seed 对比。

安全预设从 `ER SDE`、`36` 步、`CFG 4.5` 和模型标准 shift 行为开始。它会以可复现的方式配置未启用模块，但不会静默开启实验性组合。Anima-Turbo checkpoint 请选择 **Official Turbo**（`Euler`、`10` 步、`CFG 1`），或选择 **Turbo + NAG**，在 CFG 1 下通过 Normalized Attention Guidance 保留反向提示词，无需额外的无条件推理。Turbo checkpoint 使用 Base 设置（或反之）时，会在日志中给出警告。

## 启动参数

把参数写在 `COMMANDLINE_ARGS`（`webui-user.bat`、`webui.settings.sh`）中，或直接传给 `python launch.py`。`python launch.py --help` 会列出全部参数，包括内置和已安装扩展添加的参数（例如 `--agent-scheduler-sqlite-file`、`--controlnet-loglevel`、`--lora-dir`）。

**服务器与访问**

| 参数 | 用途 |
| --- | --- |
| `--port N` | 监听端口。默认：从 7860 起第一个空闲端口（`--nowebui` 时为 7861） |
| `--listen` | 监听所有网卡（`0.0.0.0`），允许其他机器连接 |
| `--server-name HOST` | 监听指定地址或主机名 |
| `--subpath /sd` | 在路径前缀下提供服务，用于反向代理（代理是否去掉前缀均可） |
| `--forwarded-allow-ips 127.0.0.1` | 信任来自这些反向代理（地址或网段，逗号分隔）的 `X-Forwarded-For` / `X-Forwarded-Proto`，从而按真实客户端统计登录失败次数。不设置时忽略这些请求头，代理后的所有客户端共用一个计数 |
| `--tls-keyfile KEY --tls-certfile CERT` | 提供 HTTPS（会话 cookie 变为 `Secure`） |
| `--auth user:pass[,user2:pass2]` | 网页界面、`/api/v2`、文件和 WebSocket 需要登录（原 `--gradio-auth`） |
| `--auth-file PATH` | 从文件读取 `--auth` 格式的登录信息，每行一个或多个（原 `--gradio-auth-path`） |
| `--source-url URL` | 用户获取本服务器源代码的地址：登录界面、设置 → 许可证和设置 → 服务器与模型会链接它（GNU AGPL v3 第 13 条）。默认是 Remi 的公开仓库：仓库中有正在运行的提交时指向该提交，否则指向 `remi` 分支。**运行修改过的副本时请设置它**，指向你所运行版本已公开的源代码（[licensing.md](licensing.md)）。未给出此参数时可用 `REMI_SOURCE_URL` 设置 |
| `--api` | 同时提供兼容 A1111 的 `/sdapi/v1` API |
| `--api-auth user:pass[,…]` | `/sdapi/v1` 的 HTTP Basic 登录（API 客户端访问扩展路由时也使用） |
| `--api-no-auth-on-localhost` | 设置了 `--auth` 而没有 `--api-auth` 时：本机的 API 客户端无需登录即可使用 `/sdapi/v1` 和扩展 API 路由（旧 WebUI 的行为） |
| `--nowebui` | 仅 API（`/api/v2`、`/sdapi/v1`，无网页界面），默认端口 7861 |
| `--cors-allow-origins URLS`、`--cors-allow-origins-regex RE` | 允许其他来源的浏览器前端，以及仅监听本机时额外允许的主机名 |
| `--enable-insecure-extension-access` | 在网络监听时仍允许从界面安装和更新扩展 |
| `--hide-ui-dir-config` | 隐藏文件夹设置、指向服务器路径的输入（图生图和后期处理的从目录批量处理、后期处理的视频片段、提示词扩写（PUP）的本地模型路径，仅限 API）以及“打开文件夹” / “在文件夹中显示”操作；空的参数粘贴不再读取上次生成的 `params.txt`，队列导入也不显示文件路径 |
| `--freeze-settings`、`--freeze-settings-in-sections`、`--freeze-specific-settings` | 设置只读 |
| `--autolaunch` | 启动时打开浏览器 |
| `--ngrok TOKEN` | 通过 ngrok 隧道发布 |
| `--timeout-keep-alive N` | HTTP keep-alive 超时（秒） |
| `--api-server-stop` | 启用 `/sdapi/v1/server-kill|restart|stop` |

**路径与安装**

| 参数 | 用途 |
| --- | --- |
| `--data-dir PATH` | 用户数据的根目录：设置、样式、输出、扩展、`models/` 和 `remi-data/`（默认：安装目录） |
| `--model-ref PATH` | 用集中的模型目录替代本地 `models` 目录 |
| `--ckpt-dirs PATH`（可重复） | 额外的 checkpoint 目录 |
| `--forge-ref-a1111-home PATH`、`--forge-ref-comfy-home PATH`、`--forge-ref-comfy-yaml PATH` | 复用 AUTOMATIC1111 或 ComfyUI 安装的模型目录，或读取 ComfyUI 的 `extra_model_paths.yaml` |
| `--uv`、`--uv-symlink`、`--uv-local-cache` | 使用 `uv` 安装依赖 |
| `--skip-install`、`--skip-prepare-environment` | 跳过依赖安装（以及网页界面构建） |
| `--sage` / `--flash` | 安装可选 attention 依赖；安装成功不代表一定会被选用 |
| `--disable-extra-extensions`、`--disable-all-extensions` | 不加载用户扩展，或不加载任何扩展 |

**模型、预设与推理 Worker**

| 参数 | 作用 |
| --- | --- |
| `--disable-preset sd xl …` | 从模型栏和预设设置中移除这些界面预设 |
| `--sort-preset` | 按字母顺序列出界面预设 |
| `--force-w4a8` | 以 `w4a8` 格式加载 `convrot_w4a4` checkpoint（画质更好） |
| `--use-ck-attention`、`--enable-triton-backend` | 使用 Comfy-Kitchen attention（覆盖其他 attention 包）；启用 `comfy-kitchen` 的 Triton 内核 |
| `--pynvml` | 通过 NVIDIA Management Library 读取实际可用显存 |
| `--worker-process` | 为 PyTorch 推理启动一个独立的受监管子进程（`SubprocessWorker`）。预览：队列仍在进程内执行所有任务 |
| `--worker-gpus 0,1` | 在 `WorkerPool` 中按指定的 GPU 索引为每张卡启动一个独立推理 Worker。预览：队列还不会把任务派给它们 |
| `--worker-mlx` | 启动可选的 Apple Silicon MLX / `mflux` Sidecar Worker。预览：队列还不会把任务派给它 |

这三个 `--worker-*` 参数目前是预览：Worker 会出现在 Workers 监控里（健康状态、重启），但队列不会把任务派给它们，仍在进程内执行所有任务。没有给出参数时，`REMI_WORKER_PROCESS=1`、`REMI_WORKER_GPUS=0,1` 和 `REMI_WORKER_MLX=1` 起同样的作用（开关只认 `1`）。

在 Apple Silicon 上，`REMI_MPS_DEQUANT_CACHE_MB` 设置在步骤之间缓存多少 MiB 反量化后的权重（`0` 关闭缓存）。不设置时，缓存最多占推理所需之外剩余统一内存的四分之一，上限 2 GiB。手动设置的值按原样使用，不受这个上限约束；不是整数的值会被忽略，负数会关闭缓存，两者都会在控制台给出警告。

**已变更的参数。** 仅用于 Gradio 的参数已移除：请把 `--gradio-auth` / `--gradio-auth-path` 改为 `--auth` / `--auth-file`，并删除 `--gradio-debug`、`--gradio-allowed-path` 和 `--no-gradio-queue`。遇到未知参数时 WebUI 会拒绝启动，因此旧的登录参数不会被悄悄忽略。`--share` 仍被接受，但只会输出警告（请改用 `--listen`、反向代理或 `--ngrok`）；`--use-textbox-seed`、`--ui-config-file`（不再读取 `ui-config.json`）、`--api-log`、`--ngrok-region` 和 `--add-stop-route` 不再起作用；`--theme light|dark|system` 是尚未在外观菜单中选择主题的浏览器所用的主题（与以前一样，URL 中的 `?__theme=light` 会选定主题）；`--agent-scheduler-sqlite-file` 告诉队列导入器旧 Agent Scheduler 数据库的位置。不要盲目安装所有 attention 后端；原生 PyTorch attention 往往兼容性最好，详见上游[额外安装说明](https://github.com/Haoming02/sd-webui-forge-classic/wiki/Extra-Installations)。

## 数据存储位置

以下路径相对于数据目录（`--data-dir`，默认为安装目录）。

| 内容 | 位置 |
| --- | --- |
| 设置 | `config.json`（`--ui-settings-file`） |
| 提示词样式 | `styles.csv` |
| 生成的图片 | `output/txt2img-images`、`output/img2img-images`、`output/extras-images`，网格图和保存的图片在相邻目录（设置 → 图片保存路径） |
| 模型 | `models/`（或 `--model-ref`） |
| 用户扩展 | `extensions/` |
| 扩展备份 | `config_states/`（扩展 → 备份与恢复）。早期版本不论 `--data-dir` 如何设置都把它们放在安装目录中：这些备份仍会列出并可以恢复，新备份写入数据目录，旧备份移到这里后才能在页面上删除 |
| 内置扩展的数据 | `extension-data/`：Spectrum 预设、LoRA 分段与画师混合模板、提示词黑名单、后期处理预设、提示词工坊的设置 |
| **Remi 数据** | `remi-data/`：`jobs.db`（队列、历史、多阶段管线链与模板）、`files.db`（文件索引、图库星标、项目看板与资产血缘）、`recipes.db`（跨模型配方）、`uploads/`（你上传的图片，超过 `remi_upload_retention_days` 后清理，但排队中的任务、未完成的管线或已保存的管线模板仍在使用的会保留）、`outputs/`（未被保存设置写入输出目录的结果，按同样规则清理，但在图库中加星、加入看板或所属任务已加星的结果会保留）、`thumbs/`（缩略图缓存）、`networks/`（提取的预览图和占位图）、`prompt-tools.db`（提示词历史、收藏、补全使用频率）、`secret.key` 和 `revoked-sessions.json`（登录会话） |
| 自定义 CSS | `user.css`（存在时应用到网页界面） |
| 完成提示音 | `notification.mp3`（数据目录中没有时也会读取安装目录中的） |
| 浏览器本地 | 表单值（草稿）、外观、语言、面板尺寸：保存在浏览器的本地存储中。生成菜单中的“重置所有参数”可将表单恢复为默认值 |

迁移安装或 Docker 卷时请保留 `remi-data/`；删除它会清空队列、历史、图库星标和提示词工具数据，但不会删除 `output/` 中的图片。

## 键盘快捷键

命令面板（Ctrl+K，Mac 上为 ⌘K）列出所有命令及其快捷键；不在文本框内时按 `?` 可查看整个应用的快捷键（图像编辑器中按 `?` 显示编辑器自己的列表）。

| 位置 | 按键 |
| --- | --- |
| 全局 | Ctrl+K / ⌘K 命令面板 · ? 键盘快捷键 · Alt+N 模型网络抽屉 |
| 生成 | Ctrl/⌘+Enter 生成（运行中：中止并立即重新生成；开启“让 [Ctrl + Enter] 恢复为仅中断生成”设置后只中止）· Ctrl/⌘+Shift+Enter 加入队列 · Alt+Enter 跳过 · Esc 中止（在提示词编辑器中也可以；其他文本框和对话框内不行）· 结果：← / → 选择、Enter 打开、Delete 移除 |
| 提示词编辑器 | Ctrl/⌘+↑ / ↓ 调整选中内容的权重 · Alt+← / → 移动标签 · Tab 或 Enter 接受补全 · Ctrl/⌘+Z 撤销 · Ctrl+Y / ⌘⇧Z 重做 |
| 图像编辑器 | B 画笔 · E 橡皮擦 · G 填充 · M 矩形 · L 套索 · I 取色器 · H 抓手 · [ / ] 画笔大小 · W / A / S + 滚轮 调整画笔大小 / 不透明度 / 柔和度 · 按住 Shift 擦除 · Ctrl/⌘+Z / Y 撤销 / 重做 · Ctrl/⌘+X 或 Delete 清空图层 · Ctrl/⌘+I 反转蒙版 · R 或 0 适应窗口 · 1 实际大小 · + / − 缩放 · F 最大化 · 空格+拖动或右键拖动 平移 · 暂存的候选图：← / → 切换、Enter 采纳、Esc 丢弃 · ? 帮助 |
| 查看器 | ← / → 上一张 / 下一张 · Home / End · + / − / 0 / 1 缩放 · S 下载 · T 平铺预览 · B 切换 RGBA 底板 · Esc 关闭 |
| 图库 | 方向键移动 · Enter 打开 · 空格选择（Shift 扩展选择）· F 星标 · Ctrl/⌘+A 全选 · Delete 移除 · Esc 清除选择 |
| 模型网络 | / 搜索 · 方向键移动 · Enter 应用 · I 元数据 · E 编辑 · Shift+F10 条目菜单 |
| 队列 | 空格拿起任务，↑ / ↓ 移动，空格放下，Esc 取消 |
| 后期处理 | Ctrl/⌘+Enter 处理 · Ctrl/⌘+Shift+Enter 加入队列 · Esc 中止 |
| 设置 | / 搜索 · Ctrl/⌘+S 应用 |

## 从 Gradio 界面迁移

Remi 的界面是全新的；生成引擎、模型、设置文件、样式、输出目录和 infotext 格式都与之前相同，你的文件和图片可以原样沿用。

| 旧界面 | Remi |
| --- | --- |
| 文生图和图生图标签页 | **生成**页，顶栏中切换文生图 / 图生图 |
| 图生图子标签（图生图、涂鸦、局部重绘、涂鸦重绘、上传蒙版重绘、批量处理） | **源图**卡片中的模式选择器；图像编辑器在舞台上 |
| 顶部的 checkpoint、VAE / 文本编码器和预设下拉框 | 顶栏中的**模型栏** |
| 快捷设置 | 顶栏的 ⚡ 快捷设置；在设置页中逐项固定 |
| 提示词下方的 Extra Networks 标签页 | **模型网络**抽屉（Alt+N）和**模型网络**页面 |
| “无限生成”右键菜单、Repeat Generate | 生成按钮旁的菜单：无限生成、连续生成多次 |
| Agent Scheduler 标签页和“Enqueue” | **队列**页面，以及生成按钮旁的**加入队列**（Ctrl+Shift+Enter）；队列设置保留，布局类设置见[其 README](../../extensions-builtin/agent-scheduler/README.md) |
| Prompt Queue 标签页 | **队列**页面 |
| Prompt All-in-One | 每个提示词下方的提示词工具栏（首次使用时自动导入原有历史和收藏）；各项旧功能的去向见[其 README](../../extensions-builtin/sd-webui-prompt-all-in-one-neo/README.md) |
| TagComplete | 内置自动补全（原有设置保留，未保留的设置及原因见[其 README](../../extensions-builtin/sd-webui-tagcomplete-neo/README.md)） |
| 括号检查、移动端布局 | 已内置 |
| PNG Info 标签页 | **图片信息**页面 |
| 设置标签页及其“默认值”和“操作” | **设置**页面；服务器操作位于“服务器与模型”。不再读取 `ui-config.json`：表单从默认值、界面预设和浏览器保存的值开始 |
| 扩展标签页 | **扩展**页面 |
| JoyCaption、Prompt Workshop 和 JSON Builder 标签页 | 同名页面 |
| 模型合并（Checkpoint Merger）标签页 | **模型融合** 页面（控件相同；合并与保存在生成之后排队执行，显示进度并可取消） |
| `/file=` 链接 | 文件通过 id 提供（`/api/v2/files/{id}`） |
| `/agent-scheduler/v1/*`、`/prompt-queue/*` API | `POST /api/v2/jobs` 和 `/api/v2/queue/*`（[API 指南](api.md)）；旧路由返回 `410 Gone` |
| `/tacapi/v1/*`、`/physton_prompt/*` API | `/api/v2/prompt-tools`（旧路由返回 `410 Gone`） |

**迁移旧队列。** 打开 **队列 → ⋯ → 从 Agent Scheduler 或 Prompt Queue 导入…**。导入器只读取旧数据、不做修改，可以安全地重复运行（已导入的任务会跳过）。Agent Scheduler 中待运行的任务会进入队列（队列会先暂停，便于你检查），已完成的任务连同图片进入历史，书签变为星标。按位置保存的脚本参数会根据当前面板命名；无法全部命名的任务会以“不可重放”状态保留并列在报告中：在对应表单中打开即可检查并重新排队。Agent Scheduler 数据库默认位于 `extension-data/agent-scheduler/task_scheduler.sqlite3`，或 `--agent-scheduler-sqlite-file` 指定的位置；Prompt Queue 的 `queue.json` 位于 `extension-data/sd-webui-prompt-queue/`。

**第三方扩展。** 只处理图像、没有界面的扩展通常可以继续使用。带 Gradio 界面（`ui()`、`on_ui_tabs`、JavaScript）的扩展需要移植：仍声明 `ui()` 的脚本会被禁用，导入 Gradio 的文件会被跳过，原因会显示在控制台和 `GET /api/v2/app`（`not_ported`）中；WebUI 的其余部分照常工作。移植方法见[扩展开发指南](extension-guide.md)。为 `/sdapi/v1` 编写的 API 客户端在加上 `--api` 后可继续使用。

## 内置扩展

Remi 内置一组经过筛选、已全部移植到新界面的扩展：

- ADetailer Neo、NegPiP、Detail Daemon、Prompt Anchor、Prompt Blacklist、Post Processing Suite、Lock img2img Steps
- JoyCaption、Prompt Workshop 和 JSON Builder（页面）、提示词工具（Prompt All-in-One 的后继）与标签补全、语言包
- Forge ControlNet 及其预处理器、ControlLLLite 和 IP-Adapter、MultiDiffusion、Image Stitch、Soft Inpainting、Never OOM、Sparse / Radial attention、编译和 PiD
- Remi 的 Anima 引导、Regional、FreeFuse、Artist Mixer、DAVE、Cross-Attention Boost、LoRA 分层权重与阶段调度、提示词缩放、分辨率、Dynamic Shift、Hires Guard、Spectrum 和预设

Agent Scheduler、Prompt Queue 和 Repeat Generate 已成为核心的队列功能。“内置”表示这些扩展与 Remi 一起版本化并受其测试覆盖，并不代表所有外部扩展都兼容。

## 安全与远程访问

默认情况下服务器只监听 `127.0.0.1`。

- **登录。** `--auth` / `--auth-file` 保护所有内容：网页界面、`/api/v2`、文件、WebSocket、`/internal` 和 API 文档。浏览器会看到登录页；会话有效期 30 天，重启后仍然有效，退出登录会将其吊销。脚本可以改用 HTTP Basic 凭据。同一客户端一分钟内登录失败十次后会被锁定一分钟。`/sdapi/v1` 在设置了 `--api-auth` 时使用它；否则使用 `--auth` 登录（脚本可用 HTTP Basic），仅监听本机时也是如此，因为同一台机器上的反向代理或隧道同样从回环地址连接。无法发送凭据的本地工具可以用 `--api-no-auth-on-localhost` 放行（来自本机且未经过代理的请求）。
- **对外开放服务器。** 请同时使用 `--listen`（或 `--server-name`）和 `--auth`，最好放在 TLS 反向代理之后，或使用 `--tls-keyfile` / `--tls-certfile`。不支持 `--share`。网络监听时，除非指定 `--enable-insecure-extension-access`，否则不能从界面安装和更新扩展；图库中的“在文件夹中显示”只对同一台机器上的浏览器有效，`--hide-ui-dir-config` 会移除所有在服务器上打开或定位文件夹的操作。
- **拒绝跨站请求。** 来自其他网站页面的写操作请求和 WebSocket 连接会被拒绝，恶意网页无法操控你的 WebUI。脚本和 API 工具不发送 `Origin` 头，不受影响；其他来源的浏览器前端需要用 `--cors-allow-origins` 按确切来源放行：`*`（或匹配所有来源的正则）只让所有网站读取无需登录的内容，绝不携带你的登录，也不能做任何修改。
- **拒绝 DNS 重绑定。** 仅监听本机的服务器只响应 `localhost` / 回环主机名（或 `--cors-allow-origins` 放行的主机）。本地反向代理必须以这些名称作为 `Host` 转发，或放行其公开来源。
- **文件**只能通过不透明 id 访问，上传内容会检查类型和大小（`remi_upload_max_mb`），其他请求体的大小上限是它的两倍，扩展 HTML、infotext 和元数据都以纯文本显示或经过净化。
- **扩展路由**（扩展添加到服务器的路由）同样经过跨站和主机检查；启用 `--auth` 时，它们还需要登录或 `--api-auth` 凭据（仅在使用 `--api-no-auth-on-localhost` 时本机 API 客户端免登录）。
- **启动警告**会指出有风险的组合：无登录即可被访问、`--cors-allow-origins=*`、不安全的扩展访问、未使用 TLS 的登录。威胁模型和对外开放的检查清单见 [docs/remi-webui/security.md](security.md)。
- **对外网络请求**只发生在你使用或配置的功能中：扩展页面中的扩展索引、你配置的翻译服务（若为 OpenAI 兼容服务，“AI 撰写提示词”时你输入的描述也会发送给它），以及部分功能首次使用时的模型下载（CLIP Modulation、JoyCaption、ControlNet 预处理器、ADetailer）。

## 故障排查

**浏览器显示“The web interface is not built”。** 启动器运行时未找到 Node.js、构建失败，或构建被跳过（`--skip-install`、`--skip-prepare-environment`、`REMI_SKIP_WEBUI_BUILD`），控制台会说明是哪一种。Node.js 版本不受构建工具支持时会提示“Node.js vX cannot build the web interface”，构建通常会失败。安装 Node.js 22.18+ 或 24.11+ 后重启，或在其他机器上构建 `webui/dist` 再复制过来（[网页界面的构建](#网页界面的构建)）。

**构建失败**（`pnpm install` 或 `pnpm build` 报错退出）。在 `webui/` 中运行 `pnpm install --frozen-lockfile && pnpm build` 查看完整信息。使用代理时，请为 npm/pnpm 配置代理（`HTTPS_PROXY`）。服务器仍会启动，期间 API 可用。

**更新后页面看起来还是旧的。** 刷新页面；应用发现新的构建时也会自动刷新一次。如果仍然如此，说明构建没有运行（见上文）。

**403 “Unknown host refused”。** 你通过其他主机名（局域网名称、反向代理、隧道）访问了仅监听本机的服务器。请使用 `--listen` / `--server-name`，或用 `--cors-allow-origins=https://your.host` 放行该名称，或让代理发送 `Host: localhost`。

**403 “Cross-site request refused”。** 其他来源的页面（独立前端、用户脚本站点）发送了写请求。用 `--cors-allow-origins` 放行该来源。在反向代理之后，请按浏览器发送的样子转发公开的主机名，放在 `Host`、`X-Forwarded-Host` 或 `Forwarded: host=` 中；端口不是 `http` 或 `https` 的默认端口时要带上端口（也可以用 `X-Forwarded-Port` 给出端口）。`X-Forwarded-Proto` 与这项检查无关（[反向代理](api.md#reverse-proxies)）。

**没有实时预览或进度，顶栏的连接指示为红色。** WebSocket（`/api/v2/ws`）被拦截。反向代理必须转发 WebSocket 升级（nginx：`proxy_http_version 1.1; proxy_set_header Upgrade $http_upgrade; proxy_set_header Connection "upgrade";`）。同时检查设置中是否启用了实时预览。

**在反向代理的子路径下页面空白。** 使用与代理路径一致的 `--subpath /your/prefix` 启动。

**端口被占用。** 未指定 `--port` 时，服务器会依次尝试 7860、7861 …… 直到 7879；指定 `--port` 时该端口被占用则启动失败。请在控制台中查找 `Running on local URL:`。

**某个扩展的面板或页面不见了。** 查看控制台：声明 Gradio `ui()` 而非 `params()`、或导入 Gradio 的脚本会被禁用并输出说明（也会列在 `GET /api/v2/app` 的 `not_ported` 中）。禁用的扩展在扩展页面中管理；更改在“重新加载界面”后生效。

**提示“Extension access is disabled”。** 服务器正在网络上监听；如果你信任所有能访问它的人，可以加上 `--enable-insecure-extension-access` 启动。

**表单保留了奇怪的值。** 表单值按浏览器保存。使用生成菜单中的“重置所有参数”，或点击单个值旁的重置圆点。

**登录状态无法保持。** 使用 `--tls-*` 或在 HTTPS 之后时，会话 cookie 带有 `Secure` 标记，请访问 HTTPS 地址。cookie 的作用路径受 `--subpath` 限定。

**显存不足或生成缓慢。** 这些属于引擎问题，与 Forge Neo 相同：尝试 Never OOM 面板、“低精度扩散”中的低比特 dtype、更小的批次，以及原生 PyTorch attention。

## 兼容性说明

- Forge Neo 已移除的训练功能、旧模型家族和旧版 WebUI 组件不会在 Remi 中恢复。
- 为 Gradio 界面编写的扩展需要移植；只处理图像的扩展和 `/sdapi/v1` 客户端通常无需修改。
- FlashAttention 等 CUDA 专用组件并非所有平台都可用。
- 实验性引导方法可能同时改变画面风格和质量；对比时请保留固定 seed 的基准图。
- Torch compile（整模型预设和 **Anima per-block**）在 NAG、NegPiP、Regional、FreeFuse、Artist Mixer、DAVE 或 Cross-Attention Boost（各自生效区间内）以及 Sparse Attention 启用时，相关步以未编译方式运行，确保这些功能不会被缓存的计算图静默忽略；这类运行不会获得编译加速。
- **Anima FlowMatch** 只对 flow-matching 模型有意义；在其他模型上（例如从 Anima Flow 采样器切换模型后仍保留该选项），会回退为 Simple 调度。
- 在 flow-matching 模型（Anima、Flux）上：Karras、Exponential、Polyexponential、KL Optimal 和 Align Your Steps 是按 SD 的噪声尺度设计的，会让画面发白，任务完成时会给出警告（请改用 Simple、Beta、SGM Uniform 或 Normal）；DDIM、PLMS 和 Kohaku LoNyu Yog 会直接报错拒绝。CFG++ 采样器在 CFG 1（它们的推荐用法）下会保留负向提示词。
- 真实 checkpoint 的生成结果才是最终兼容性检验。启动测试或单元测试不能证明每种 checkpoint 与量化格式的出图质量。
