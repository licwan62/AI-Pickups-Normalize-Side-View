# Pickup Side Profile Measurement Tool

## 单页尺寸工作台（推荐）

双击根目录 `start_workbench.cmd`，或运行：

```powershell
.venv\Scripts\python.exe -m pickup_measure.workbench
```

启动后自动打开 `http://127.0.0.1:8765`。可加 `--port 8766` 更换端口，或加
`--no-browser` 只启动服务。前端仅一个 `pickup_measure/web/index.html`，无需前端构建。
HTML 通过本地 Python 服务访问全量表、保存进度和执行图像处理，不能直接双击 HTML 使用。

1. 拖入单张或多张图片，自动获取全量表候选。文件名精确命中时直接匹配；否则通过
   Qwen 识别品牌、车型、版本、年份，再搜索全量表。AI 不提供尺寸。
2. 对照原图核对市场、年份、车身版本及 **长 / 宽 / 高（mm）**，选择具体记录并勾选
   核对框，点击“确认尺寸”。不同宽度的候选也会提示三维差异。
3. 点击“生成真实尺寸 SVG”，或“生成全部已确认”。页面展示处理阶段、预览、下载和失败信息。
   图像定位、方向归一、裁剪和毫米 SVG 渲染复用原有 Python 流程。

“导入现有 input”会读取 `config.yaml` 的 `images_dir`，原图保持原样；导入的现有素材需
在页面选择后点击搜索或识图匹配，避免扫描目录时自动发起大量 AI 请求。
可手动搜索修正车型，也可重新识图；重新匹配会撤销尺寸确认。重复内容图片合并为同一任务。

全量表优先使用 `master_table.source` 与现有缓存；源和缓存均不可用时使用
`data/dimensions/全量表_汇总.csv`，页面会标明实际来源。更新全量表后可以点击“刷新”。
确认与生成前都会重新检查源表以及原图，尺寸或图片改变后须重新匹配确认。
Qwen 凭据沿用 `config.yaml` / `DASHSCOPE_API_KEY`，只在 Python 服务中使用。

工作台的任务与日志保存在 `data/workbench/`，刷新页面或重启服务不会丢失进度；
上次中断的任务标为失败，可重新匹配或重试生成。新拖入的图片保存到
`<images_dir>/_workbench/<唯一目录>/`。工作台输出独立保存为：

```text
<output_dir>/workbench/<任务 ID>/
├── review.json                     # 原图哈希、识别结果、所选全量表行和确认时间
└── <Size>/
    ├── <车型 ID>.svg                # 可直接下载，图像已内嵌
    └── <车型 ID>/                   # 原图、裁剪、方向、边界等过程文件
```

侧视 SVG 的车辆本体为 **车长 × 车高**；车宽用于核对与追溯。根元素的 `width`、
`height` 使用 mm，`viewBox` 对应毫米坐标，标注留白使画板大于车辆本体。
SVG 包含栅格车辆图及矢量标注，不是自动描摹的车辆矢量轮廓。
工作台仅绑定本机 `127.0.0.1`，现有命令行和项目 CSV 流程继续可用。

批量读取皮卡侧视图，由 AI 识别主车辆、车头方向和透视校准并紧密裁剪，再按表格中的真实车长和车高生成可直接导入 Adobe Illustrator 的毫米尺寸 SVG，标注车型名称、整车 `LENGTH` 和 `HEIGHT`。也可以用 `--side-view` 处理单张图片：Qwen 车辆识别 → 全量表匹配 → 裁剪、校正、渲染。

> 局部尺寸测量（HOOD/CAB/NECK/BED）、比例质检、红框标注和 `measurements.tsv` 已移除；原 `--measure` 参数不再可用，`--no-measure` 和 `--approve-warning` 仍可传入但不起作用。

## 环境安装

需要 Python 3.11 或更高版本。在项目根目录运行：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如果 PowerShell 不允许执行激活脚本，也可以不激活环境，后续直接使用 `.venv\Scripts\python.exe` 代替 `python`。

Qwen 识别需要 DashScope API Key。可以在 `config.yaml` 的 `detection.api_key` 填写，留空时读取环境变量 `DASHSCOPE_API_KEY`。`--side-view` 的车辆识别和主体定位都会调用 Qwen。

## 1. 准备车辆表格

默认输入文件为 `input/vehicles.tsv`，使用 Tab 分隔，编码建议为 UTF-8：

```tsv
name	Size	image_path	length_mm	width_mm	height_mm
Ford F150 Raptor	PK-XL	input/images/F150_01.jpg	5908	2200	2027
```

字段说明：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `name` | 是 | 汇总表中的车型名称 |
| `Size` | 是 | 输出分类目录，例如 `PK-XL`、`YL`、`YM+` |
| `image_path` | 否 | 图片路径；填写后优先读取此路径 |
| `length_mm` | 是 | 车辆总长，单位 mm |
| `width_mm` | 是 | 车辆总宽，单位 mm |
| `height_mm` | 是 | 车辆总高，单位 mm |

也支持逗号分隔的 CSV 文件，通过 `--input` 指定即可。

程序会按表头顺序拼接除 `image_path` 外的所有字段来自动生成 ID，字段之间使用下划线连接；空白和文件名非法字符也会替换为下划线。例如上面的记录会生成
`Ford_F150_Raptor_PK-XL_5908_2200_2027`。除 `image_path` 外字段完全相同的两行会因 ID 重复而被拒绝。

## 2. 准备车辆图片

建议在表格中填写 `image_path`。如果留空，则把图片放入 `input/images/`，文件名应与自动生成的 ID 相同，例如：

```text
input/images/Ford_F150_Raptor_PK-XL_5908_2200_2027.jpg
```

支持 `.jpg`、`.jpeg`、`.png`、`.webp` 和 `.avif`。程序会自动处理 EXIF 旋转并转换为 RGB；AVIF 解码后会按现有流程输出为 PNG crop。若表格填写了 `image_path`，则优先使用该路径，不要求图片文件名与自动生成的 ID 相同。

为提高自动识别精度，建议使用接近正侧面的整车图，轮胎完整接触地面，背景尽量纯净，车辆四周不要被裁断。

## 3. 运行自动处理

在项目根目录运行：

```powershell
python main.py
```

程序将逐车执行 AI 主体定位、方向归一、透视校正和无留白裁剪，然后导出带整车长高标注的 SVG。某一辆车失败不会中断整个批次，详情记录在日志和运行汇总中。

常用命令：

```powershell
# 显示更详细的日志
python main.py --verbose

# 复用已经保存的裁剪边界，不重新检测
python main.py --reuse-points

# 指定其他 TSV/CSV、图片区和输出目录
python main.py --input data\vehicles.tsv --images data\images --output result

# 自动识别困难时，打开人工框选窗口
python main.py --manual
```

### 3.1 单图识别与全量表匹配

单张图片可以直接交给 Qwen 识别，再从 NAS 上的全量表（`*/全量/全量表.csv`）找到对应尺寸行并生成 SVG：

```powershell
python main.py --side-view "img/input/PK-XL/Ford F-150 Raptor Pickup 2021-2026 SuperCrew 5.5.png"

# 只识别和匹配，写 match.json，不裁剪、不渲染
python main.py --side-view photo.png --match-only

# 指定市场；未指定时依次尝试文件名中的市场后缀和识别结果中的 market
python main.py --side-view photo.png --market US

# 多个候选车长/车高不同而需要强行采用最高分时
python main.py --side-view photo.png --allow-ambiguous

# 强制重新构建全量表缓存
python main.py --side-view photo.png --refresh-master
```

匹配结果写入 `<output>/side_view/<图片名>/match.json`，`status` 为 `MATCHED`、`AMBIGUOUS`、`NO_MATCH`、`FAILED` 或 `EXPORTED`。`candidates` 保留前 12 个候选行及分数，`selected` 包含命中的 `DIMENSION-ID`、`Size` 和长宽高；`AMBIGUOUS` 表示最高分附近存在不同车长或车高组合，可用 `--allow-ambiguous` 接受最高分，或先用 `--match-only` 检查候选。宽度不同的行不会阻止匹配。最终 SVG 与批量流程一致，输出到 `Size/<id>.svg`。

全量表相关设置位于 `config.yaml`：

```yaml
master_table:
  source: '\\NAS8824B4\Public\PQData\pub_all_cars_data\data\基础数据'
  patterns:
    - "*/全量/全量表.csv"
  cache_path: data/cache/master_vehicles.csv
  refresh: auto        # auto | always | never
  fallback_size: UNCLASSIFIED
  preferred_markets: [US, EU, RU]
  min_score: 0.62
  identify_prompt_file: promting_identify.md
```

`auto` 会按文件大小和修改时间判断缓存是否仍有效；NAS 暂时不可用时会继续使用已有缓存。`refresh: never` 只读缓存，适合离线运行。

## 4. 输出文件

每辆车的最终标注 SVG 以 ID 命名，直接保存在 Size 目录；过程文件保存在对应的 ID 子目录：

```text
output/PK-XL/
├── F150_01.svg
└── F150_01/
    ├── source.jpg
    ├── crop_source.png
    ├── orientation.json
    ├── detection.json
    └── points.json
```

- `<id>.svg`：Size 目录下的最终交付文件，包含紧密裁剪车辆、车型名称及整车 `LENGTH`、`HEIGHT` 标注。
- `crop_source.png`：紧密裁剪的原始像素，仅作为 SVG 内嵌图源，不需要单独导入工程。
- `orientation.json`：车头方向识别结果、置信度、翻转状态及各项判断分数。车头在左时，`crop_source.png` 会水平翻转为车头朝右。
- `points.json`：自动或人工确定的裁剪边界。
- `detection.json`：AI 返回的主体边界、车头方向和透视校准结果；不包含红色车型结构点。

批次级文件直接保存在 `output/` 根目录：

```text
output/
├── run_summary.json
└── pickup_measure.log
```

## 5. Illustrator 导入说明

请导入 Size 目录下的最终 `<id>.svg`；不要把 `crop_source.png` 作为最终工程图导入。SVG 的 `width`、`height` 和 `viewBox` 使用毫米定义，因此 Illustrator 中的物理车长和车高以 TSV 数据为准，与栅格图的 PPI 无关。

Size 目录下的 `<id>.svg` 为了把总长、总高尺寸线放到车辆两侧，画板会比车辆本体更大；车辆本体的尺寸仍然严格等于 TSV 中的 `length_mm × height_mm`。

## 6. 调整标注样式

在 `config.yaml` 的 `annotation` 节点中调整显示效果：

```yaml
annotation:
  background_color: "#FFFFFF"
  image_opacity: 0.50
  outline_color: "#C8242A"
  outline_width_mm: 8
  dimension_color: "#202124"
  dimension_width_mm: 4
  font_size_mm: 82
  font_family: "Arial"
```

其中线宽和字号均使用毫米。输出目录通过 `config.yaml` 的 `output_dir` 设置，例如 `output_dir: img/output`；相对路径以配置文件所在目录为基准，未配置时使用该目录下的 `output/`。显式传入 `--output` 会覆盖配置，其相对路径以当前工作目录为基准。上文输出结构以 `output/` 为例，日志、汇总文件及续跑检查均使用实际输出目录。图片区通过 `--images` 设置。

## 7. 运行测试

```powershell
python -m pytest -q
```

当前版本以自动处理为主，并保留 `--manual` 人工接管入口。Streamlit 可视化关键点调整界面和 YOLO/SAM 自动识别尚未加入。
