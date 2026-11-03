# beyond-spectral-indices

**Disentangling Spectral and Spatial Evidence in Deep Burn-Scar and Forest Segmentation: A Factorial Ablation with Faithfulness-Checked Attributions**

Code and paper for a study of Sentinel-2 burn-scar delineation (dNBR vs. learned models) and
tree-cover mapping (NDVI vs. learned models). The study asks two questions:

- **When** do deep models beat the index? The answer is a stratified error analysis by burn
  severity, land cover, index ambiguity and object boundaries.
- **Why** do they beat it? The answer combines TreeSHAP, Integrated Gradients / GradientSHAP,
  band occlusion, a randomisation sanity check and a spatial "context radius".

## Method ladder

|                    | per pixel    | spatial      |
|--------------------|--------------|--------------|
| index bands only   | `lgbm_index` | `unet_index` |
| all 12 bands       | `lgbm_all`   | `unet_all`   |

Every method is compared with the index under three thresholds: literature, Otsu, and tuned on
the train split. The gap to the index splits into **rule**, **spectrum** and **space** effects.

## Data

| Task   | Imagery                                     | Labels                       | Source |
|--------|---------------------------------------------|------------------------------|--------|
| Burn   | Sentinel-2 L2A pre/post fire, California     | CAL FIRE perimeters (CaBuAr) | Hugging Face `DarthReca/california_burned_areas` |
| Forest | Sentinel-2 L2A 2021 growing-season composite | ESA WorldCover 2021, class 10 | Microsoft Planetary Computer (no account needed) |

## Running

Use a GPU machine, for example Google Colab. `notebooks/run_colab.ipynb` runs everything.

```bash
pip install -e ".[forest]"
python scripts/prepare_data.py configs/burn.yaml --inspect   # check the CaBuAr file layout once
bash scripts/run_all.sh                                       # both tasks, end to end
```

Or step by step:

```bash
python scripts/prepare_data.py configs/burn.yaml
python scripts/fit_index.py    configs/burn.yaml
python scripts/train.py        configs/burn.yaml unet_all     # also lgbm_index, lgbm_all, unet_index
python scripts/evaluate.py     configs/burn.yaml
python scripts/explain.py      configs/burn.yaml
python scripts/make_paper_assets.py configs/burn.yaml configs/forest.yaml
```

`make_paper_assets.py` writes `paper/tables/*.tex` and `paper/figures/*.pdf`, so no number in the
paper is typed by hand. To build the paper, upload `paper/` to Overleaf (it uses the ACM
`acmart` class) or run `latexmk -pdf main.tex`.

## Layout

```
configs/          task configs (data source, splits, training, explanation settings)
src/bsi/          library: data, indices, models, metrics, strata, explanations
scripts/          pipeline steps
paper/            ACM sigconf manuscript, bibliography, generated tables and figures
notebooks/        Colab runner
```

## License

MIT
