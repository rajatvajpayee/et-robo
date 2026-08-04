# Project Instructions

## Environment Setup

1. **Create a new environment:**
    ```bash
    conda create -n <env_name> python=3.12
    conda activate <env_name>
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118
    ```
    - **Note:** If you have a different CUDA version, adjust the `--extra-index-url` accordingly (refer to [PyTorch official installation instructions](https://pytorch.org/get-started/locally/)).

3. **Checkpoints:**  
    The best checkpoint is saved in the `checkpoints/` folder.
    Alternatively, you can download the best checkpoint [here](https://drive.google.com/drive/folders/1pen5tLmVZ5IYvVNGEpWSxgi2f80F3onh?usp=sharing).

---

## Training, Evaluation, and Analysis

### 1. Train the Model
Run:
```bash
bash train.sh
```
- The best model checkpoint will be saved in the `checkpoints/` directory.
- **Tip:** You can adjust batch size, model capacity, or other parameters inside the `train.sh` script depending on your GPU memory or requirements.

### 2. Test / Evaluate the Model
After training, evaluate the model with:
```bash
bash eval.sh
```
- Loads the best checkpoint and evaluates on the test set.
- Results and logs are generated (see the script for output paths).

### 2b. Evaluate the Initial Baseline Model (from HuggingFace)
To evaluate the baseline results using the initial model provided by HuggingFace:
```bash
bash eval_baseline.sh
```
- This script evaluates the initial (pretrained, unfine-tuned) model checkpoint from HuggingFace on the test dataset.

### 3. Compare Multiple Models
To compare results from baseline and the finetuned model:
```bash
bash compare.sh
```
- Evaluates multiple checkpoints and provides comparison metrics.

### 4. Sampling Analysis (Top-k, p-value Studies)
To perform a sampling study (e.g., varying p values, top-k sampling):
```bash
bash sample.sh
```
- This script runs inference with different sampling parameters and generates output (e.g., multiple generations for various `top-k` and `top-p` values).

### 5. Saliency Visualizations (Attention, Grad-CAM, etc.)
To compute and save detailed saliency visualizations:
```bash
bash saliency.sh
```
- This will generate and save various interpretability outputs, such as attention rollout, Grad-CAM, and attention heatmaps.
- Results (images or matrices) are stored in an output directory for further analysis.

---

## Blind Ablation Study

To evaluate the model with *blank/noise* inputs (such as removing the image modality):

**Option 1: Run the Python script directly:**
```bash
python src/blind_ablation.py --checkpoint checkpoints
# Optional arguments:
#   --mode [zeros|noise]           # Ablation mode for input removal
#   --batch-size <int>             # Adjust batch size per your GPU capacity
#   --other-params ...
```

**Option 2: Use the provided shell script:**
```bash
bash blind_ablation.sh
```

**Note:**  
- All shell (`.sh`) scripts may require edits to paths or parameters for your specific environment or hardware configuration (such as batch size, checkpoint paths, CUDA device number, etc.).
- For advanced usage, open each script in an editor to adjust to your hardware or experiment requirements.

---