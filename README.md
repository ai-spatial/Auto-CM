# Auto-CM

**Code for paper**

*Xie, Y., Li, Z., Bao, H., Jia, X., Xu, D., Zhou, X. and Skakun, S., 2023, June. Auto-CM: Unsupervised deep learning for satellite imagery composition and cloud masking using spatio-temporal dynamics. In Proceedings of the AAAI Conference on Artificial Intelligence (Vol. 37, No. 12, pp. 14575-14583).* [link](https://ojs.aaai.org/index.php/AAAI/article/view/26704)

```
@inproceedings{xie2023auto,
  title={Auto-CM: Unsupervised deep learning for satellite imagery composition and cloud masking using spatio-temporal dynamics},
  author={Xie, Yiqun and Li, Zhili and Bao, Han and Jia, Xiaowei and Xu, Dongkuan and Zhou, Xun and Skakun, Sergii},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={12},
  pages={14575--14583},
  year={2023}
}
```

## Description

Cloud masking is both a fundamental and a critical task in the many Earth observation problems. This work aims to develop an **unsupervised** deep learning framework to detect clouds based on the differences in the spatio-temporal dynamics of the atmospheric events and land surface. 

## Key Features:

- **Label-free**: The training does not require any cloud labels.
- **Platform-independent**: The framework can be applied to any satellite platforms (e.g., PlanetScope, Landsat-8, and Sentinel-2).

## Example

The following figure shows an example of results comparing to deep clustering ([DEC](https://dl.acm.org/doi/10.5555/3045390.3045442)) and default cloud masks (Default) included in the imagery products.
<!--![Example](https://github.com/ai-spatial/Auto-CM/blob/main/visualization/Example_results.png)-->
<img src="https://github.com/ai-spatial/Auto-CM/blob/main/visualization/Example_results.png" width="600">

## Explanation of the code:

We provided the codes for training and testing with PlanetScope data. Codes for other satellite platforms may be uploaded later.

[AutoCM.ipynb](https://github.com/ai-spatial/Auto-CM/blob/main/AutoCM.ipynb) has everything together in one notebook.

[AutoCM_training.py](https://github.com/ai-spatial/Auto-CM/blob/main/AutoCM_training.py): Training Auto-CM model for 100 epochs using PlanetScope imagery time-series data. The model does not require time-series input during testing.

[AutoCM_testing.py](https://github.com/ai-spatial/Auto-CM/blob/main/AutoCM_testing.py): Generating cloud masks for new PlanetScope imagery tiles (TOA or surface reflectance) using the trained model. 
