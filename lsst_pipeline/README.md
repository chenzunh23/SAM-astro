# Scarlet Deblend Demo 使用说明

这个文件夹用于在一组三波段 HSC coadd FITS cutout 上运行 LSST Science Pipelines 的
detection / merge / scarlet deblend 流程。当前版本也支持把 SAM 的 mask 转成 LSST
footprint，然后继续使用 LSST 的 merge 和 scarlet deblend。

## 目录结构

- `input/projection_cutout/`
  - 输入的 HSC-G、HSC-R、HSC-I coadd FITS。
  - `coadd_scene.json`、`summary.json` 记录输入数据来源和摘要。
- `input/repo/deepCoadd/skyMap.pickle`
  - 最小 Butler 风格 repo 元数据，主要给 merge/deblend 读取 tract/patch/skyMap。
- `scarlet_deblend_from_fits.py`
  - 主流程脚本。
  - `--detection-mode lsst` 使用 LSST 原生检测。
  - `--detection-mode sam` 使用 SAM 生成检测 footprint，再交给 LSST merge/deblend。
- `export_scarlet_visualizations.py`
  - 从 pipeline 输出中导出 PNG 可视化、逐源图像/光谱、science leaf 表和 merge 区域图。
- `convert_outputs_to_viewable_fits.py`
  - 把 LSST `SourceCatalog`/footprint 输出转换成简单 image FITS，方便 DS9/FIJI 查看。
- `run_demo_in_lsst_container.sh`
  - 旧版 Docker/container 运行包装脚本。
- `output/`
  - 保存不同实验运行的输出结果。

## 环境准备

不要直接用普通系统 Python 跑主流程。需要先加载 LSST Science Pipelines：

```bash
source loadLSST.sh
setup lsst_distrib
```

如果要使用 SAM，默认假定其中存在：

```text
scripts/amg_fits_core.py
```

## 运行 LSST 原生流程

```bash
python scarlet_deblend_from_fits.py \
  --detection-mode lsst \
  --repo input/repo \
  --tract 9813 \
  --patch 4,5 \
  --coadd HSC-G=input/projection_cutout/HSC-G/deepCoadd-HSC-G-9813-4,5.fits \
  --coadd HSC-R=input/projection_cutout/HSC-R/deepCoadd-HSC-R-9813-4,5.fits \
  --coadd HSC-I=input/projection_cutout/HSC-I/deepCoadd-HSC-I-9813-4,5.fits \
  --clip-sky-sources-to-exposure-bbox \
  --output-dir output/lsst_run
```

## 运行 SAM + LSST scarlet 流程

```bash
python scarlet_deblend_from_fits.py \
  --detection-mode sam \
  --sam-repo /path/to/segment-anything \
  --repo input/repo \
  --tract 9813 \
  --patch 4,5 \
  --coadd HSC-G=input/projection_cutout/HSC-G/deepCoadd-HSC-G-9813-4,5.fits \
  --coadd HSC-R=input/projection_cutout/HSC-R/deepCoadd-HSC-R-9813-4,5.fits \
  --coadd HSC-I=input/projection_cutout/HSC-I/deepCoadd-HSC-I-9813-4,5.fits \
  --clip-sky-sources-to-exposure-bbox \
  --output-dir output/sam_run
```

SAM 相关默认参数已经按当前实验设置写入脚本，主要包括：

- `--sam-points-per-side 32`
- `--sam-crop-n-layers 1`
- `--sam-pred-iou-thresh 0.8`
- `--sam-max-mask-area-ratio 0.5`
- `--sam-min-mask-region-area 15`
- `--sam-astro-preprocess-in-model`
- `--sam-astro-preprocess-sigma-iters -1`
- `--sam-astro-preprocess-z-clip -3 3`

当前 SAM 模式会对每个 SAM mask 在科学 coadd 图像上寻找多个 peak，然后把这些 peak
写入 LSST footprint。这样一个 SAM 区域可以在 scarlet deblend 中分出多个 child source。

常用调节参数：

- `--sam-detection-scope multiband`
  - 默认方式，把三波段一起用于 SAM。
- `--sam-detection-scope per-band`
  - 每个波段单独跑 SAM，再由 LSST merge 合并。
- `--sam-max-peaks-per-mask`
  - 每个 SAM mask 最多允许多少个 peak。
- `--sam-peak-threshold-sigma`
  - mask 内局部峰值的阈值。
- `--sam-peak-min-distance`
  - 多 peak 之间的最小距离。

## 主流程输出

每次运行的 `--output-dir` 下通常包含：

- `detect/HSC-*/deepCoadd_det.fits`
  - 每个波段的检测 catalog。
- `detect/HSC-*/deepCoadd_calexp.fits`
  - 拷贝/整理后的 coadd exposure。
- `merge/deepCoadd_mergeDet.fits`
  - 多波段 merge 后的 parent footprint 和 peak catalog。
- `merge/deepCoadd_peak_schema.fits`
  - peak schema。
- `deblend/deepCoadd_deblendedFlux.fits`
  - scarlet deblend 后的 source catalog。
- `deblend/deepCoadd_scarletModelData.pickle`
  - scarlet 模型数据，包含 blend、source、factorized component、morphology、spectrum。
- `manifest.json`
  - 本次运行的参数、输入、检测数量、merge/deblend 摘要。
- `sam/`
  - 仅 SAM 模式存在，保存 SAM label map 和相关中间结果。

注意：`deepCoadd_deblendedFlux.fits`、`deepCoadd_mergeDet.fits` 是 LSST
`SourceCatalog` FITS，不是普通二维图像。DS9/FIJI 不能直接把里面的 footprint 当图像查看。

## 导出可视化结果

运行：

```bash
python export_scarlet_visualizations.py \
  --input output/sam_run \
  --output output/sam_run_visuals
```

快速测试时可以限制逐源 panel 数量：

```bash
python export_scarlet_visualizations.py \
  --input output/sam_run \
  --output output/sam_run_visuals_quick \
  --max-source-panels 20
```

可视化输出包括：

- `source_panels/source_XXXXXXXX_panel.png`
  - 单个 source 的模型图像和光谱图。
  - 左侧图像带本地像素坐标标尺，并标出 peak 位置。
- `source_panels/source_spectra.csv`
  - 每个 source 的光谱、模型 bbox、peak 坐标。
- `visual_check/science_leaf_map.png`
  - 类似原始 visual check 的全局检查图，只显示 `deblend_nChild == 0` 的 science-ready leaf source。
  - 橙色 `+` 是 child leaf，青色圆是 isolated leaf，红色 `x` 是多子源 blend parent anchor。
- `visual_check/science_leaf_footprint_map.png`
  - 每个 science leaf footprint 的彩色叠加图。
- `visual_check/complex_family_leaf_gallery.png`
  - 多子源 parent 的局部 cutout 图，并标注 child source id。
- `visual_check/science_leaf_sources.csv`
  - science-ready leaf source 表。
- `zscale_mask_overlays/*.png`
  - 以 HSC-I 图像的 astropy zscale 显示为底图，逐个叠加输入输出 FITS 中可解析的 mask、label map 或 LSST footprint。
  - 如果某个 LSST catalog 没有可画的 footprint mask，会退回显示 footprint bbox；完全没有 mask/bbox 的 FITS 会输出带说明文字的 PNG。
- `zscale_mask_overlays/zscale_mask_overlays.csv`
  - 每个 FITS 对应的 overlay 类型、mask 像素数、bbox 数和输出 PNG 文件名。
- `merge_regions/merge_parent_XXXXXXXX.png`
  - merge 后每个 parent footprint 的可视化。
- `merge_regions/merge_parent_XXXXXXXX_mask.fits`
  - 每个 parent footprint 对应的简单二维 mask FITS。
- `merge_regions/merge_regions.csv`
  - 每个 merge parent 的 mask 面积和 peak 数量。

## 固定 ROI 对比 SAM merge 和 LSST merge

`roi_visualization.py` 会在固定大小的局部区域中对比两套 merge 后的 parent footprint：
SAM 检测输入经过 LSST merge 后的结果，以及 `no_docker_test` 中原始 LSST 检测经过
merge 后的结果。脚本默认直接读取两边的 `merge/deepCoadd_mergeDet.fits`，并用 astropy
还原 footprint mask；如果已经导出了 `merge_regions/merge_parent_*_mask.fits`，也可以
用 `--sam-merge-regions` 或 `--lsst-merge-regions` 指定。

```bash
python roi_visualization.py \
  --sam-input output/scarlet_sam_per_band_peaks \
  --lsst-input output/no_docker_test \
  --output output/scarlet_sam_per_band_peaks_roi \
  --roi-size 128 \
  --max-rois 6
```

也可以固定到指定 parent 或手动给局部像素中心：

```bash
python roi_visualization.py \
  --sam-input output/scarlet_sam_per_band_peaks \
  --lsst-input output/no_docker_test \
  --output output/scarlet_sam_per_band_peaks_roi_selected \
  --parents 33,70,156 \
  --centers 256,256
```

输出包括：

- `roi_XXX_*.png`
  - 同一个 ROI 下的 SAM merge 与 LSST merge mask 对比图；不同 parent 使用不同颜色。
- `roi_visualization.csv`
  - 每个 ROI 的中心、边界，以及相交的 SAM parent id 和 LSST parent id。

## 转换为 DS9/FIJI 更容易打开的 FITS

```bash
python convert_outputs_to_viewable_fits.py \
  --input output/sam_run \
  --output output/sam_run_viewable_fits
```

这个脚本会把 LSST catalog 中的 footprint 画成普通二维 image FITS。此类输出适合用
DS9/FIJI 检查 mask 区域。

## parent、child 和 science leaf

- merge 后的一个连通 footprint 是一个 parent blend 区域。
- deblend 会根据 parent 区域中的 peak 拆出 child source。
- 如果一个 parent 没有 child，它自己就是 isolated leaf source。
- 如果一个 parent 有 child，那么 child 是最终用于科学检查的 leaf source。
- 本项目的 visual check 默认只把 `deblend_nChild == 0` 的记录当作 science-ready leaf。

因此，判断“最终源”的数量时不要只看 parent 数，也不要只看 merge footprint 数，应该看
`science_leaf_sources.csv` 或 `deblend_nChild == 0` 的 catalog 行。

## 常见问题

### 为什么 `tract` 和 `patch` 仍然需要填写？

即使输入 FITS 文件本身没有显式标注 tract/patch，LSST merge/deblend 仍需要它们来构造
DataId、读取 skyMap、写入一致的 catalog 元数据。对当前 cutout，请使用：

```text
--tract 9813 --patch 4,5
```

### SAM 的单波段输入怎么处理？

SAM 原始接口默认接收三通道图像。单波段测试时通常把同一个 FITS 图像重复三遍，形成伪
RGB 输入。当前主脚本的 `per-band` 模式就是按这个思路处理每个波段。

### 哪些 FITS 可以直接用 DS9/FIJI 打开？

通常可以直接打开：

- `detect/HSC-*/deepCoadd_calexp.fits` 的图像 HDU。
- `sam/*_sam_labelmap.fits`。
- `merge_regions/*_mask.fits`。
- `convert_outputs_to_viewable_fits.py` 生成的二维 image FITS。

通常不适合直接当图像打开：

- `merge/deepCoadd_mergeDet.fits`
- `deblend/deepCoadd_deblendedFlux.fits`
- `deblend/deepCoadd_scarletModelData.pickle`

这些需要 LSST Python 或本项目脚本读取。
