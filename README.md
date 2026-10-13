# 真实质感 · 全图细化 · 高清修复 · 自动调色

这个仓库提供 [完整 ComfyUI 工作流](真实质感_全图细化_高清修复.json) 和其中自制的 **Anima 自动肤色调整 / Anima 画面通透** 节点。公开版保留现有节点、连线、模型及 LoRA 名称和权重；人物外貌、衣着、姿势、场景等正向提示词已移除，同时清空了提示词编辑器的缓存和 LoRA 的人物触发文字。导入后请先填写自己的正向提示词。

调色链在高清修复后：`高清修复 → 自动肤色 → 画面通透 → Olm Image Adjust → 保存图像`。自动肤色的 `skin_mask` 输出接到画面通透，人物检测器的 `SEGM_DETECTOR` 输出接到自动肤色。检测时会按尺寸补边，减轻人物贴住画面边缘时选区被截断的问题。

## 调色所需的四项资源

| 资源 | 用途 | 获取位置 |
| --- | --- | --- |
| 本仓库代码 | `AnimaAutoSkinTone`、`AnimaSceneTone`，含边缘选区修复 | 直接使用本仓库 |
| [ComfyUI-Impact-Subpack](https://github.com/ltdrdata/ComfyUI-Impact-Subpack) | `UltralyticsDetectorProvider` 人物检测节点 | 原作者仓库或 ComfyUI Manager |
| [person_yolov8m-seg.pt](https://huggingface.co/Bingsu/adetailer/blob/main/person_yolov8m-seg.pt) | 人物分割模型，约 55 MB | 模型作者页面 |
| [ComfyUI-Olm-ImageAdjust](https://github.com/o-l-l-i/ComfyUI-Olm-ImageAdjust) | 末端整体亮度、对比度、饱和度 | 原作者仓库 |

本仓库不复制后三项。尤其 [Olm 的许可](https://github.com/o-l-l-i/ComfyUI-Olm-ImageAdjust/blob/main/LICENSE.txt)禁止未经书面许可重新分发其扩展，安装时请使用原作者的发布渠道。

## 安装和导入

1. 把本仓库克隆或解压到 `ComfyUI/custom_nodes/ComfyUI-Anima-SkinTone/`。该目录下应直接有 `__init__.py`、`skin_tone.py`、`scene_tone.py`、`web/skin_tone.js`。
2. 在 ComfyUI Manager 安装 `ComfyUI Impact Subpack`。完整工作流里的 `DetailerForEach`、`MaskToSEGS` 等还需要单独安装 [ComfyUI-Impact-Pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack)。两者是不同扩展。
3. 下载 `person_yolov8m-seg.pt`，放到 `ComfyUI/models/ultralytics/segm/`；检测节点里选择 `segm/person_yolov8m-seg.pt`。
4. 按原作者说明安装 Olm 节点；重启 ComfyUI 并刷新浏览器。
5. 将 [真实质感_全图细化_高清修复.json](真实质感_全图细化_高清修复.json) 拖入 ComfyUI。使用 Manager 的「安装缺失节点」处理其余插件，并为工作流中的底模、CLIP、VAE、LoRA、放大模型选择本机实际拥有的文件。
6. 在正向提示词节点中填写自己的角色和场景描述，再运行。

完整工作流还使用 `WeiLinPromptUIOnlyLoraStack` / `WeiLinPromptUIWithoutLora`、`easy hiresFix`、`Seed (rgthree)`、`Image Comparer (rgthree)` 等其他扩展；这四项资源只涵盖调色链。JSON 仅记录模型文件名与 LoRA 权重，**不包含任何模型或 LoRA 权重文件**。若不具备相同文件，可在对应节点替换成自己的模型。

## 调色参数

公开工作流保留当前轻调数值：自动肤色的去粉紫 `1`、提亮 `3`、浓淡 `0`、宽容度 `50`、边缘柔化 `0.35%`；画面通透的中间调 `5`、暗部 `3`、去冷灰 `40`、暖意 `20`、强度 `100`。Olm 仅做非常轻微的整体亮度、对比度和饱和度调整。去粉紫是 0–100 的数值，`1` 的作用很轻；可按自己的成图逐步提高。

自动选区基于人物分割和肤色条件，可能误选颜色接近的衣服、反光，也可能漏掉极小或被严重遮挡的皮肤。更换图片后先查看绿色选区预览。滑块预览只用于试调；保存最终图片时需重新运行工作流。

## 许可与来源

本仓库自制节点代码按 [MIT 许可](LICENSE)分享。Impact Subpack、Olm 和检测模型各自保留原作者的许可与维护渠道。自制节点只用 ComfyUI 中已有的 PyTorch、NumPy、Pillow 和 aiohttp，不需要单独安装 Python 包。
