# Self-Supervised Dual Contouring

This repository contains the official implementation for the paper:

**Self-Supervised Dual Contouring**  
*Ramana Sundararaman, Roman Klokov, Maks Ovsjanikov*  
**arXiv:2405.18131 [cs.CV]**

The paper can be found on [arXiv](https://arxiv.org/abs/2405.18131).

## Installation

To install the required dependencies, run the following command:

```bash
pip install -r requirements.txt
```

## Usage

To train a new model, you can run the `train.py` script with a configuration file. For example:

```bash
python train.py --config configs/config_file.yaml --exp_name my_experiment
```

## Evaluation

To evaluate a trained model on a dataset, use the `single_eval.py` script. You will need to provide paths to the decoder weights and the data directory.

```bash
python single_eval.py --dec_weight /path/to/your/model.th --data_dir /path/to/your/data --save_dir ./results
```

Check the script for more options.

## Configuration

An example configuration file can be found at `Configs/example.yaml`. This file contains all the parameters for training and data loading.

## Citation

If you find our work useful, please consider citing our paper:

```bibtex
@misc{sundararaman2024selfsupervised,
      title={Self-Supervised Dual Contouring}, 
      author={Ramana Sundararaman and Roman Klokov and Maks Ovsjanikov},
      year={2024},
      eprint={2405.18131},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
