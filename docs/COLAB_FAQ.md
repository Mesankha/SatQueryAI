# SatQuery AI — Colab FAQ & Troubleshooting

## Q1: Do I really NOT need BigEarthNet?

**Correct. You don't need BigEarthNet.** Here's why:

| BigEarthNet limitation | Planetary Computer alternative |
|------------------------|-------------------------------|
| 15-68GB download | 0 GB (streamed) |
| European data only | Any AOI worldwide |
| No SAR | Sentinel-1 SAR included |
| Static (2017-2018) | Always current |
| Requires registration | No signup, no API key |

**When BigEarthNet IS useful:**
- Academic benchmarks (most papers use it)
- Reproducing published results
- Fine-tuning for European land cover

**For your use case (Indian cities + SAR), use Planetary Computer.**

## Q2: How much Colab time do I need?

| Colab tier | Time for 3 epochs (3000 samples) |
|-----------|----------------------------------|
| Free T4 (16GB VRAM) | 4-6 hours |
| Colab Pro T4 | 3-4 hours |
| Colab Pro A100 (40GB) | 30-45 min |

The free T4 is sufficient for LoRA training. Just be aware:
- Free tier disconnects after 12 hours (use checkpoints to resume)
- Free tier has daily GPU quotas (check your usage)

## Q3: My Colab disconnected mid-training. Now what?

**Good news**: The training script saves checkpoints every epoch (or every 500 steps).

To resume:
```python
!python -m m3_vlm.train_multimodal \\
    --data-dirs ./data/sar_lora ./data/optical_lora \\
    --output-dir ./models/geochat_india_sar \\
    --resume-from-checkpoint ./models/geochat_india_sar/checkpoint-XXXX \\
    --epochs 3
```

Replace `checkpoint-XXXX` with the actual checkpoint number (e.g., `checkpoint-500`).

## Q4: I see "CUDA Out of Memory". How to fix?

**Reduce memory usage in this order:**

1. **Reduce LoRA rank** (smaller adapter):
   ```python
   --lora-r 8 --lora-alpha 16  # instead of 16/32
   ```

2. **Reduce samples**:
   ```python
   --max-train-samples 1500  # instead of 3000
   ```

3. **Reduce gradient accumulation** (uses more memory):
   ```python
   --grad-accum 4  # instead of 8
   ```

4. **Enable 4-bit quantization** (requires code change in train_multimodal.py to add `--load-in-4bit` flag)

5. **Use smaller image size** (in m3_vlm/train_lora.py, find `load_image_safely` and change `max_side=1024` to `max_side=512`)

## Q5: How do I get the trained model to my local machine?

**Method 1: Google Drive (recommended)**
- Adapter is auto-saved to your Drive at `MyDrive/satquery/models/geochat_india_sar/adapter/`
- Download the whole `adapter/` folder to your local machine
- Set `LORA_ADAPTER_PATH` env var and start M3 service

**Method 2: Direct download from Colab**
```python
# In Colab, zip and download
!cd /content/satquery/models/geochat_india_sar && zip -r adapter.zip adapter/
from google.colab import files
files.download('adapter.zip')
```

**Method 3: Git push from Colab**
```python
!cd /content/satquery && git add models/geochat_india_sar/adapter/
!git commit -m "Add trained LoRA adapter"
!git push origin main  # requires GitHub auth setup
```

## Q6: What if the Planetary Computer API is slow or down?

**Backup plan: Use HuggingFace SEN12MS dataset**

```python
# Replace Cell 4 with this:
!python -m m3_vlm.acquire_sar huggingface \\
    --dataset blanchon/SEN12MS \\
    --output ./data/sen12ms_lora \\
    --max-samples 2000
```

This pulls 180k pre-paired Sentinel-1 + Sentinel-2 scenes from HuggingFace. No internet to Planetary Computer needed.

## Q7: How do I know if my LoRA is actually learning SAR?

**Three ways to verify:**

1. **Check TensorBoard loss curves**: Training loss should decrease steadily. If it plateaus or increases, something is wrong.

2. **Test before/after comparison**:
   ```python
   # Base GeoChat (no adapter) - should give bad SAR output
   !python -m m3_vlm.test_inference --image ./data/sar_lora/images/test_sar.tif
   
   # Fine-tuned (with adapter) - should give good SAR output
   !python -m m3_vlm.test_inference --image ./data/sar_lora/images/test_sar.tif \\
       --adapter ./models/geochat_india_sar/adapter
   ```

3. **Quantitative metrics**:
   ```python
   !python -m m3_vlm.evaluate --adapter ./models/geochat_india_sar/adapter \\
       --data-dir ./data/sar_lora --output ./eval_results.json
   ```
   - BLEU-4 > 0.25: Captioning is working
   - F1 > 0.7: VQA is working
   - IoU > 0.4: Grounding is working

## Q8: Can I use the base GeoChat without fine-tuning?

**Yes!** The base GeoChat works for:
- RGB optical captioning
- RGB optical VQA
- RGB optical grounding

**It does NOT work well for:**
- SAR images (will hallucinate)
- Multi-temporal analysis
- Fusion tasks

**To use base GeoChat** (no training needed):
```python
# In M3 service, just don't set LORA_ADAPTER_PATH
unset LORA_ADAPTER_PATH  # or just don't set it
python -m m3_vlm.service
```

## Q9: How long does data acquisition take?

| Step | Time | Notes |
|------|------|-------|
| Install packages | 3-5 min | One-time |
| BigEarthNet download | NOT NEEDED | Skip this |
| Get SAR data (5 cities × 100) | 10-20 min | Depends on network |
| Get optical data (3 cities × 50) | 5-10 min | Depends on network |
| Sanity check (dummy training) | 3-5 min | Validates pipeline |
| Real training (3 epochs) | 4-6 hours | The long step |
| Evaluation | 5-10 min | After training |
| **Total** | **~5-7 hours** | Mostly waiting |

## Q10: I want to use A100 (Colab Pro). Do I need to change anything?

**No code changes needed.** The training script auto-detects GPU.

But you can use larger batch sizes for speed:
```python
!python -m m3_vlm.train_multimodal \\
    --batch-size 2 --grad-accum 4  # larger effective batch
    --max-train-samples 5000  # more data
    ...
```

## Q11: What's the difference between `train_lora.py` and `train_multimodal.py`?

| File | Purpose | When to use |
|------|---------|-------------|
| `train_lora.py` | Original single-modality (optical) | If you only want optical |
| `train_multimodal.py` | Multi-modal (optical + SAR + fusion) | **Recommended for SAR understanding** |

**Use `train_multimodal.py` for your use case.**

## Q12: Can I train on only SAR (no optical)?

**Yes!** Modify the training command:
```python
!python -m m3_vlm.train_multimodal \\
    --data-dirs ./data/sar_lora \\
    --output-dir ./models/geochat_sar_only \\
    --modalities sar \\  # only SAR
    --epochs 5  # more epochs since less data
    ...
```

This is useful if you want to specialize the model only for SAR.

## Q13: Can I use the model on Hugging Face Spaces after training?

**Yes!** After training, push to HF:

```python
# In Colab:
!pip install huggingface_hub
!huggingface-cli login  # enter your HF token

from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="/content/satquery/models/geochat_india_sar/adapter",
    repo_id="YOUR_USERNAME/geochat-india-sar-lora",
    repo_type="model"
)
```

Then anyone can use your model with:
```python
from peft import PeftModel
from transformers import AutoModelForCausalLM
base = AutoModelForCausalLM.from_pretrained("mbzuai-oryx/GeoChat")
model = PeftModel.from_pretrained(base, "YOUR_USERNAME/geochat-india-sar-lora")
```

## Q14: How do I deploy for a demo (web app)?

**Two options:**

### Option A: Start M3 service in Colab, expose via ngrok
```python
!pip install pyngrok
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_TOKEN")
public_url = ngrok.connect(8001)
print(f"Public URL: {public_url}")
# Now your model is accessible at this URL from anywhere
```

### Option B: Download adapter to local machine, run there
```powershell
# On your local Windows machine:
$env:LORA_ADAPTER_PATH = "D:\path\to\downloaded\adapter"
python -m m3_vlm.service
# Then your local M3 service uses the fine-tuned adapter
```

## Q15: What if I want to add more cities later?

**Easy!** Just re-run Cell 4 with new cities:
```python
!python -m m3_vlm.acquire_sar cities \\
    --cities chennai kolkata hyderabad pune \\
    --output ./data/sar_lora \\
    --max-per-city 100
```

Then either:
- Continue training from your existing adapter (transfer learning)
- Re-train from scratch with all cities

For incremental training, use:
```python
!python -m m3_vlm.train_multimodal \\
    --data-dirs ./data/sar_lora \\
    --output-dir ./models/geochat_india_sar \\
    --resume-from-checkpoint ./models/geochat_india_sar/checkpoint-XXX
```

## Q16: Should I train on the 3050 laptop OR Colab?

| Aspect | 3050 laptop | Google Colab |
|--------|-------------|--------------|
| **VRAM** | 4GB | 16GB (T4 free) / 40GB (A100 Pro) |
| **Speed (3 epochs)** | 8-12 hours | 4-6 hours (T4) / 30-45 min (A100) |
| **Cost** | Free (you own it) | Free (with limits) or $10/mo Pro |
| **Stability** | Always available | 12-hour disconnect on free tier |
| **Internet** | Your connection | Google's fast connection |
| **Disk space** | 500GB+ available | 15GB free, 100GB on Pro |
| **Best for** | Multiple iterations, long training | Quick start, no local setup |

**Recommendation**: Start with Colab to verify the pipeline works, then move to 3050 for the real training (since you can leave it running overnight).

## Q17: What if I'm getting "Out of quota" or "Colab disconnected"?

**Free tier limits:**
- ~12 hours continuous GPU
- Daily GPU quota (varies)
- Idle timeout: 90 min

**Workarounds:**
1. **Save checkpoints frequently** (already configured in our training script)
2. **Use `--max-steps` instead of `--epochs`** for shorter, resumable runs
3. **Use Colab Pro** ($10/month for more reliable access)
4. **Use the 3050 laptop** for longer runs

## Q18: Final checklist before training

```
[ ] I have a Google account
[ ] I have ~5GB free space in Google Drive
[ ] I have the satquery code in Drive (or GitHub)
[ ] I understand the workflow takes 4-6 hours
[ ] I have a backup plan (3050 laptop or HF datasets)
[ ] I know how to resume from checkpoint if disconnected
[ ] I have ngrok token (optional, for remote access)
```

If all checked, you're ready to go!

## Q19: What if I get stuck?

**Debug order:**
1. Run `python -m m3_vlm.pipeline_check` to verify the code
2. Check the Colab cell output for specific error messages
3. Verify file paths (especially after Drive mount)
4. Check VRAM with `!nvidia-smi`
5. Check disk space with `!df -h`
6. Look at the training log in `/content/satquery/models/geochat_india_sar/logs/train.log`

**Common errors:**
- `ModuleNotFoundError: No module named 'm3_vlm'` → sys.path not set, add `sys.path.insert(0, '/content/satquery')`
- `RuntimeError: CUDA out of memory` → reduce batch size or use 4-bit
- `FileNotFoundError: data/sar_lora/images` → data acquisition didn't complete, re-run Cell 4
- `HF token required` → for private models, use `huggingface-cli login`

## Summary

The Colab workflow I provided is designed to be:
- **Simple**: 11 cells, clear progression
- **Resilient**: Checkpoint-based, can resume
- **Efficient**: 4-6 hours for a good model
- **Free**: No API keys, no paid services
- **Practical**: Real Indian city data, not synthetic

If you have specific issues, let me know and I'll help debug them.