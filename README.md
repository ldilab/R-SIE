# R-SIE
This repo provides the official code of TACL paper:
Overcoming Source Object Grounding for Semantic Image Editing
```markdown
# R-SIE 🍰  — Region-wise Diffusion for Semantic Image Editing
> Official implementation (PyTorch + diffusers) of  
> **“Overcoming Source-Object Grounding for Semantic Image Editing”**  
> accepted to *TACL* (2025).  
> We introduce a **Region-wise Diffusion Process (RwDP)** that cleanly
> separates background reconstruction from object manipulation, eliminating
> the grounding errors that plague prior SIE pipelines.

<div align="center">
<img src="docs/teaser.png" width="640"/>
</div>

---

## 📦 Repository layout



r-sie/
├─ src/                # source codes
│   └─ RwDP_pipeline.py     # huggingface diffusers pipeline for region-wise diffusion process
└─ README.md

````

> **Planned additions**
> - **`code for data generation process** – automatic triplet generation (code & instructions)  🚧  
> - **`test_sets/`** – links to the cleaned evaluation splits  🚧  
> - **`requirements`** – required packages to run codes  🚧  
---

## 🔧 Prerequisites

| package | tested version |
|---------|----------------|
| Python  | 3.9            |
| PyTorch | 1.12           |
| diffusers | 0.22.0       |
| transformers | 4.36.2    |
| accelerate | 0.27.2      |


---

## 📄 License

Code is released under the **MIT License**; see `LICENSE`.
Model checkpoints follow the original **Stable Diffusion v1.5** CreativeML
license.

---

## 🙋 Questions & Contact

Open an issue or e-mail **[y970120@snu.ac.kr](mailto:y970120@snu.ac.kr)**.
We welcome pull requests for bug-fixes or documentation!

```
```
