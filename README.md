# gpu-vintage

How long does a GPU remain economically useful for serving large language
models? This repository measures the throughput and energy of LLM inference on
seven NVIDIA GPU types from five hardware generations (Pascal to Ada Lovelace)
under one software stack, and feeds the measurements into a vintage-capital
model that derives each card's economic life, its value over that life, and the
electricity and carbon prices at which an older card should give way to new
capacity. It holds the code, the outputs of every run, and the scripts that
build every table and figure of the paper from them.

## Method

**Measurement.** Every GPU runs the same inference engine for the dense Qwen3
models (0.6, 1.7, 4, 8 and 14 billion parameters), written in plain PyTorch 2.6
with CUDA 12.4 so that it runs unchanged from Pascal (sm_60) to Ada (sm_89):
fused projections, a static key-value cache and CUDA graphs for decoding.
A run measures prefill and decoding separately, for static batches of 1 to 256
requests of synthetic prompts, in windows of at least 10 s after 2 s of warm-up,
plus 10 s of idle power. Energy is the GPU board energy read through NVML: the
cumulative energy counter where the card has one (Volta and later) and the
integral of power sampled every 20 ms on every card. A request of 512 prompt
tokens and 256 generated tokens is the main shape; its time and energy are
combined from the two phases. Each run writes three JSON sidecars:
`config.json` (what was run), `env.json` (library versions, thread counts, GPU
model) and `metrics.json` (every measurement window, with clocks,
temperatures and clock-event reasons).

**Economics.** Following Salter (1960) and Malcomson (1975), competition holds
the price of a token at the full cost of new capacity, which falls at the pace
θ of embodied technical progress. A card is scrapped when its operating cost
(electricity, carbon and, where data-centre capacity is scarce, the capacity it
occupies) reaches that price, and free entry pins the price level so that a new
card's discounted rents repay its cost. The pace θ is the slope, on the release
date, of the logarithm of the tokens that a dollar buys at the full cost of new
capacity, along the frontier of data-centre and workstation cards. The model
(`src/gpuvintage/econ.py`) gives the economic life, the value at each age, the
ratio of an older card's operating cost to the full cost of new capacity, the
break-even carbon price and the carbon payback of a replacement, for the
electricity prices and grid carbon intensities of six economies.

## Layout

| path | what |
|---|---|
| `src/gpuvintage/` | `model.py` (Qwen3 in PyTorch), `engine.py` (prefill and the CUDA-graph decoder), `power.py` (NVML meter), `bench.py` (one run), `analysis.py` (sidecars to rows), `econ.py` (the economic model) |
| `scripts/` | `bench.py` and `bench_hf.py` (measure), `make_campaigns.py` and `run_campaign.py` (campaigns), `build_results.py` with `build_tables.py` and `build_figures.py` (results) |
| `campaigns/` | the four campaign lists, one row per run |
| `data/` | the GPU panel, electricity prices and carbon intensities, the US consumer price index |
| `results/runs/` | the 141 runs of the paper, one directory each |
| `results/tables/` | the results as CSV |
| `paper/tables/`, `paper/figures/` | the paper's tables (LaTeX table bodies), the numbers its text quotes (`numbers.tex`) and its figures |
| `tests/` | unit tests; the GPU tests skip without a CUDA device |

## Installation

Python 3.10 to 3.13. With [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra analysis --extra dev
```

or with pip, in a virtual environment: `pip install -e ".[analysis,dev]"`. The
pinned torch 2.6.0 is the CUDA 12.4 build (`pyproject.toml` takes it from the
PyTorch cu124 index), which runs on every generation in the panel; measuring
needs an NVIDIA GPU and a driver that supports CUDA 12.4. Rebuilding the
results needs no GPU.

```bash
uv run pytest
```

## Usage

Rebuild every table, figure and quoted number from the shipped runs:

```bash
uv run python scripts/build_results.py
```

It reads `results/runs/` and writes `results/tables/`, `paper/tables/` and
`paper/figures/`, identical to the shipped files (the figures, given the same
matplotlib version and fonts: Liberation Sans).

The economic model on its own, for a card that serves 4,000 tokens per second
while busy:

```python
import math
from gpuvintage import econ

# 300 W busy and 60 W idle, rated 400 W, 22,500 USD with its share of the server;
# United States electricity, scarce data-centre capacity (1.32 USD per W and year).
op = econ.Operation(utilization=0.5, electricity=0.134, capacity_w_year=1.32)
Y = econ.mtok_per_year(4000, op.utilization)
o = econ.operating_cost_per_mtok(4000, 300, 60, op, rated_w=400)
for growth in (0.18, 0.33):   # yearly rise in the tokens a dollar of new capacity buys
    v = econ.solve_vintage(K=22_500, Y=Y, o=o, theta=math.log(1 + growth), r=0.08)
    print(f"progress {growth:.0%} a year: economic life {v.life:.1f} years")
print(f"{Y:,.0f} Mtok a year, {100 * o:.2f} US cents per Mtok to run")
```

prints

```
progress 18% a year: economic life 11.3 years
progress 33% a year: economic life 7.7 years
63,072 Mtok a year, 2.04 US cents per Mtok to run
```

Measure one model on a GPU of your own (the model must be in the local Hugging
Face cache, since runs never reach the network):

```bash
uv run huggingface-cli download Qwen/Qwen3-1.7B
uv run python scripts/bench.py --model Qwen/Qwen3-1.7B --dtype fp16 \
    --batches 1 4 16 64 --gpu-tag mygpu --run-dir out/runs/test-mygpu-r1
```

## What builds what

`scripts/build_results.py` builds everything below; the tables come from
`scripts/build_tables.py` and the figures from `scripts/build_figures.py`.

| output | content |
|---|---|
| `results/tables/runs.csv` | one row per run and batch size |
| `results/tables/agg.csv` | means and standard deviations over the three repetitions |
| `results/tables/econ.csv`, `lives.csv`, `keep.csv`, `payback.csv` | operating costs, economic lives and values, keep-or-replace ratios, carbon thresholds and paybacks |
| `paper/tables/gpus.tex` | the GPU panel: generation, release, segment, memory, bandwidth, FP32 rate, board power, launch price |
| `paper/tables/countries.tex` | electricity prices and grid carbon intensities of the six economies |
| `paper/tables/params.tex` | parameters of the economic model and their sensitivity ranges |
| `paper/tables/measured.tex` | throughput and energy per output token, Qwen3-1.7B and 4B, one request at a time and batched |
| `paper/tables/source.tex` | T4 against RTX 6000 Ada: these measurements beside those of Nguyen et al. (2024) |
| `paper/tables/rates.tex` | annual rates of change across vintages: tokens per dollar of full cost, per dollar, per joule, and from specifications |
| `paper/tables/keep.tex` | operating cost of each older card relative to the full cost of new capacity, by economy |
| `paper/tables/carbon.tex` | carbon price above which each older card should give way to new capacity |
| `paper/tables/precision.tex` | half against single precision on the Pascal cards |
| `paper/tables/software.tex` | Hugging Face `generate` against the engine |
| `paper/tables/shapes.tex` | energy per output token for three request shapes |
| `paper/tables/numbers.tex` | every number the text quotes, as LaTeX macros |
| `paper/figures/energy_vintage` | energy per output token by release date, one request at a time and batched |
| `paper/figures/life_vs_theta` | economic life of a new card against the pace of progress |
| `paper/figures/depreciation` | value of an A100 over its life, against straight-line depreciation and its second-hand price |

## Rerunning the measurements

The campaign lists in `campaigns/` (written by `scripts/make_campaigns.py`)
hold one row per run: its name, GPU, time limit and arguments. E1 runs the five
models on every card that holds them, at batches of 1 to 256; E2 runs single
precision on the two Pascal cards; E3 runs Hugging Face `generate` beside the
engine on four architectures; E4 runs two other request shapes (128 prompt
tokens and 1,024 generated; 2,048 and 64). Every run is repeated three times.
The GTX 1080 Ti runs in single precision, since its chip computes half
precision at 1/64 of its single-precision rate.

`scripts/run_campaign.py` runs the rows of one GPU, here or on any Slurm
cluster, and skips the runs that have finished:

```bash
uv run python scripts/run_campaign.py campaigns/e1.tsv --gpu a100                 # here
uv run python scripts/run_campaign.py campaigns/e1.tsv --gpu a100 \
    --sbatch "-p gpu --gres=gpu:1 -c 8 --mem=60G"                                 # Slurm
GPUVINTAGE_RUNS=out/runs uv run python scripts/build_results.py --tex out/paper --results out/tables
```

New runs go to `out/runs/` (a rerun into `results/runs/` would find every run
done). Download each model first, as above; every run records the model
revision it used in its `config.json`. A run that is interrupted resumes from
the batch sizes it has not measured. The 141 runs of the paper took about five
GPU-hours of measurement, from under a minute to about seven minutes each plus
model loading, one GPU per run, on a GeForce GTX 1080 Ti, a Tesla P100 PCIe 12 GB,
a Tesla V100 PCIe 16 GB, a Tesla T4, an A100 SXM4 40 GB, an A40 and an RTX 6000
Ada Generation.

## Data sources

| file | source | licence |
|---|---|---|
| `data/gpus.csv` | release dates, launch prices and specifications: Epoch AI's hardware database, and NVIDIA's whitepapers and datasheets (row by row in its `sources` column) | Epoch AI data: CC BY 4.0 |
| `data/countries.csv` | electricity prices: Vietnam's Ministry of Industry and Trade (Decision 1279/QĐ-BCT of 2025, Circular 60/2025/TT-BCT), the US Energy Information Administration (Electric Power Monthly), Eurostat (nrg_pc_205); exchange rates: World Bank; carbon intensities: Ember's yearly electricity data | EIA: public domain; Eurostat: reuse with acknowledgement; World Bank: CC BY 4.0; Ember: CC BY 4.0 |
| `data/cpi.csv` | US Bureau of Labor Statistics, CPI-U | public domain |

The Qwen3 models (Apache 2.0) are downloaded from Hugging Face at run time and
are not redistributed. The ratios of Nguyen et al. (2024) that
`scripts/build_tables.py` sets beside the measurements are quoted from their
paper.

## Licence

The code and the run outputs are released under the MIT licence (`LICENSE`).
The files in `data/` are compiled from the sources above, under their licences.
