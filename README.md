# ML Accelerated Phase-Field Simulation of 3D Microstructure Evolution

This repository contains code and data for the paper:

**"Accelerating Phase-Field Simulation of Three-Dimensional Microstructure Evolution in Laser Powder Bed Fusion with Composable Machine Learning Predictions"**

- **Authors**: Jin Young Choi, Tianju Xue, Shuheng Liao, Jian Cao  
- **Publication**: [ScienceDirect link](https://www.sciencedirect.com/science/article/pii/S2214860423005511)

---

### 📝 Overview

Phase-field (PF) modeling is a powerful method to simulate microstructure evolution in additive manufacturing, but its high computational cost limits large-scale simulations. This work introduces a machine learning (ML) surrogate model using a 3D U-Net architecture, trained on small-scale PF simulations, to predict grain orientations at high accuracy and significantly reduced computational cost.
<img width="1940" height="742" alt="image" src="https://github.com/user-attachments/assets/446afe00-7af5-4b6f-bb4f-223f175ec521" />

---

### 📖 Citation

If you use this code or data, please cite:

```bibtex
@article{CHOI2024103938,
title = {Accelerating phase-field simulation of three-dimensional microstructure evolution in laser powder bed fusion with composable machine learning predictions},
journal = {Additive Manufacturing},
volume = {79},
pages = {103938},
year = {2024},
issn = {2214-8604},
doi = {https://doi.org/10.1016/j.addma.2023.103938},
url = {https://www.sciencedirect.com/science/article/pii/S2214860423005511},
author = {Jin Young Choi and Tianju Xue and Shuheng Liao and Jian Cao},
keywords = {Phase-field modelling, Machine learning, Microstructure evolution, Grain orientation, Laser powder bed fusion, Additive manufacturing},
}
```

---

### 🔗 Related Project

- [jax-am](https://github.com/tianjuxue/jax-am): JAX-based simulation library for AM, with PF model used to generate data
