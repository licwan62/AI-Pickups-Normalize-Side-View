# Pickup Side Profile Measurement Tool

批量读取皮卡侧视图，自动识别车辆边缘并紧密裁剪，再按 TSV 中的真实车长和车高生成可直接导入 Adobe Illustrator 的毫米尺寸 SVG。工具同时生成结构化红色线框、尺寸标注、单车过程文件和批次汇总表。

## 环境安装

需要 Python 3.11 或更高版本。在项目根目录运行：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如果 PowerShell 不允许执行激活脚本，也可以不激活环境，后续直接使用 `.venv\Scripts\python.exe` 代替 `python`。

## 1. 准备车辆表格

默认输入文件为 `input/vehicles.tsv`，使用 Tab 分隔，编码建议为 UTF-8：

```tsv
id	name	image_path	length_mm	width_mm	height_mm
F150_01	Ford F150 Raptor		5908	2200	2027
```

字段说明：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `id` | 是 | 车辆唯一 ID，也用于查找默认图片文件名 |
| `name` | 是 | 汇总表中的车型名称 |
| `image_path` | 否 | 图片路径；填写后优先读取此路径 |
| `length_mm` | 是 | 车辆总长，单位 mm |
| `width_mm` | 是 | 车辆总宽，单位 mm |
| `height_mm` | 是 | 车辆总高，单位 mm |

也支持逗号分隔的 CSV 文件，通过 `--input` 指定即可。

## 2. 准备车辆图片

把图片放入 `input/images/`，文件名应与表格的 `id` 相同，例如：

```text
input/images/F150_01.jpg
```

支持 `.jpg`、`.jpeg`、`.png` 和 `.webp`。程序会自动处理 EXIF 旋转并转换为 RGB。若表格填写了 `image_path`，则优先使用该路径，不要求图片文件名与 `id` 相同。

为提高自动识别精度，建议使用接近正侧面的整车图，轮胎完整接触地面，背景尽量纯净，车辆四周不要被裁断。

## 3. 运行自动处理

在项目根目录运行：

```powershell
python main.py
```

程序将逐车执行自动边缘识别、无留白裁剪、长高独立缩放、比例质检、线框绘制和尺寸计算。某一辆车失败不会中断整个批次，详情记录在日志和运行汇总中。

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

## 4. 比例质检与警告复核

程序比较图片裁剪区域的长高比和车辆真实长高比：

- 误差不超过 3%：自动导出。
- 误差大于 3% 且不超过 6%：状态为 `WARNING`，暂停 SVG 导出。
- 误差大于 6%：状态为 `BLOCKED`，禁止导出。

遇到 `WARNING` 时，先检查对应的 `output/<id>/points.json` 和 `qc_report.json`。确认边界正确后，可批准该警告并复用边界：

```powershell
python main.py --reuse-points --approve-warning
```

## 5. 输出文件

每辆车的过程文件保存在 `output/<id>/`：

```text
output/F150_01/
├── source.jpg
├── crop_source.png
├── vehicle.svg
├── annotated.svg
├── points.json
├── annotation_points.json
├── qc_report.json
└── measurements.tsv
```

- `vehicle.svg`：仅包含按真实车长、车高缩放的车辆图片。
- `annotated.svg`：白底、50% 透明车辆图、红色结构线框和外置尺寸标注。
- `crop_source.png`：紧密裁剪的原始像素，仅作为 SVG 内嵌图源，不需要单独导入工程。
- `points.json`：自动或人工确定的裁剪边界。
- `annotation_points.json`：轮廓、底盘线、车门线等自动测量坐标，内部单位为 mm。
- `measurements.tsv`：单车详细测量结果。

批次级文件直接保存在 `output/` 根目录：

```text
output/
├── measurements.tsv
├── run_summary.json
└── pickup_measure.log
```

汇总表 `output/measurements.tsv` 的列为：

```text
车型  车长  车宽  CAB高  车头高  车颈高  车尾高
```

表格实际使用 Tab 分隔，所有尺寸均为整数毫米，不附带单位文字。

## 6. Illustrator 导入说明

请导入 `vehicle.svg` 或 `annotated.svg`，不要把 `crop_source.png` 作为最终工程图导入。SVG 的 `width`、`height` 和 `viewBox` 使用毫米定义，因此 Illustrator 中的物理车长和车高以 TSV 数据为准，与栅格图的 PPI 无关。

`annotated.svg` 为了把总长、总高和局部尺寸线放到车辆两侧，画板会比车辆本体更大；车辆本体的尺寸仍然严格等于 TSV 中的 `length_mm × height_mm`。

## 7. 调整标注样式

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

其中线宽和字号均使用毫米。输入文件、图片区和输出目录建议通过命令行参数设置；`config.yaml` 当前主要用于 SVG 标注样式。

## 8. 运行测试

```powershell
python -m pytest -q
```

当前版本以自动处理为主，并保留 `--manual` 人工接管入口。Streamlit 可视化关键点调整界面和 YOLO/SAM 自动识别尚未加入。
