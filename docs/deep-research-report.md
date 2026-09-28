# SatQuery AI: Vision-Language Assistant for Remote-Sensing Analysis

The SatQuery AI system is designed to answer complex remote-sensing queries by orchestrating specialized vision-language and image-processing models. It supports **single-image** and **paired-image** (cross-modal and bi-temporal) inputs, automatically selecting and composing task-specific models to generate evidence-backed answers. The system leverages large-scale remote-sensing datasets for training and evaluation, including co-registered optical (Sentinel-2) and SAR (Sentinel-1) imagery, with rich annotations (captions, Q&A, bounding-box references). In particular, the BigEarthNet.txt dataset (464,044 SAR–optical image pairs with ~9.6M text annotations) provides the multi-sensor image–text data needed to adapt vision-language models to Earth observation. Likewise, the VRSBench dataset (29,614 images with captions, object references, and Q&A pairs) enables training and evaluation of captioning, visual grounding, and VQA models in remote sensing. We also incorporate the CDVQA benchmark (“change detection meets VQA”) to train and assess change-based question answering on multi-temporal image pairs. In short, existing remote-sensing VLMs lag behind general-purpose models due to limited, narrow training data; SatQuery AI bridges this gap via domain adaptation and task-specific fine-tuning.

## 1. Data and Preprocessing

**Datasets:** We will utilize *BigEarthNet.txt* for model adaptation (training/fine-tuning). This large-scale multi-sensor dataset contains co-registered Sentinel-1 SAR and Sentinel-2 multispectral images with diverse annotations for tasks like land-cover captions, visual question-answer pairs, and referring expressions. For evaluation, we use *VRSBench* for single-image captioning, grounding, and VQA (29,614 images with detailed annotations), *RSVQA*-style datasets for generic VQA, and *CDVQA* for change-based VQA on multi-temporal pairs. The final ISRO/SAC test set includes Cartosat-2S optical and RISAT SAR images (co-registered), with reference masks, bounding boxes, and answers.

**Preprocessing Pipelines:** Input images (GeoTIFF/TIFF) are ingested via geospatial libraries (GDAL/rasterio). We validate projection and geotransform metadata to ensure co-registration of multi-modal/time series inputs. For multispectral optical images, bands are normalized (e.g. reflectance scaling) and cloud masks applied or cloudy areas flagged. SAR images (typically single-band or dual-polarization) are calibrated (e.g. radiometric correction) and converted to intensity. For multi-modal pairs (optical+SAR), we resample one to match the other’s resolution or use multi-resolution fusion. For temporal pairs, we ensure spatial alignment (same footprint/CRS) and, if necessary, register them precisely. Images are cropped or padded to square tiles (e.g. 512×512) as needed for model inputs. Common augmentations (random flips, rotations, slight intensity noise) are used during training to improve generalization. We check formats: only GeoTIFF/TIFF (and example PNG/JPEG for benchmarks) are accepted; others are rejected or converted. Input validation also verifies the expected number and types of images for the detected task (e.g. two images for change tasks).

## 2. Model Components and Adaptation

SatQuery AI uses **specialist vision-language components** for each task, all adapted or fine-tuned on remote-sensing data:

- **Single-Image Visual Question Answering (VQA):** A vision-language model (e.g. a transformer-based encoder + language decoder) is fine-tuned on remote-sensing Q&A pairs. BigEarthNet.txt provides millions of VQA pairs, and VRSBench offers open-ended Q&A in natural language. The model takes an input image and a user question (e.g. “Describe the land-cover” or “Is there water here?”) and produces an answer. We ensure the vocabulary includes RS-specific terms (e.g. “irrigated field”, “flooded”). Training uses cross-entropy or token-level loss, evaluated by answer accuracy or semantic metrics.

- **Image Captioning / Scene Description:** A captioning model generates a descriptive sentence for a single remote-sensing image. We can adapt existing captioners (like BLIP or ViT+Transformer) by fine-tuning on RS caption data. BigEarthNet.txt includes geographically grounded captions of land-use/cover and context, and VRSBench provides detailed object-rich captions. The model is trained with teacher forcing to maximize caption likelihood, and evaluated by language metrics (BLEU, CIDEr).

- **Text-Guided Region Grounding:** Given a sentence referring to an object (e.g. “the river” or “the large warehouse”), the grounding model outputs a bounding box or segmentation mask on the image. We implement this with a referring-expression model: either fine-tuning a detector (e.g. GLIP, Grounding DINO) on RS data or using a Vision-Language Transformer that outputs box coordinates. BigEarthNet.txt’s referring-expression annotations (instructions for bounding-box prediction) are directly applicable. The model is trained to minimize localization loss (e.g. IoU loss + classification). Performance is measured by IoU of predicted vs. ground-truth box.

- **Change Detection and Change Description:** For bi-temporal tasks, we use a Siamese or change-focused model. A *change detection* network (e.g. a U-Net or transformer taking two images) outputs a change map (binary mask or per-pixel change probability). Training data can come from standard change-detection sets (e.g. building change datasets) or synthesized via BigEarthNet temporal slices. After detecting changed areas, a *change description* module or LLM is used: it takes the change map (or images and change features) plus a query (“What changed between T1 and T2?”) and outputs natural-language descriptions. We can fine-tune on CDVQA data, which consists of multi-temporal image-question-answer triplets. A common pipeline is: encode each image separately (shared weights), fuse features (concatenate or via cross-attention), predict mask; then feed both images and mask to the VQA model.

- **Change-Based Visual Question Answering (CD-VQA):** A specialized VQA model answers questions about *changes* (e.g. “Has built-up area increased?”). This can be the above change detection model plus a language module. Yuan et al. introduce this CDVQA task explicitly for remote sensing. We train the model end-to-end or in two stages: first detect change map, then answer. Training uses the CDVQA dataset, and evaluation by answer accuracy on that benchmark.

- **Optical–SAR Joint Analysis:** For a co-registered optical+SAR pair, we build a *multi-modal fusion* model. One approach is a dual-branch network: an optical encoder (processing RGB/NIR bands) and a SAR encoder (processing radar intensity), whose feature maps are fused (e.g. concatenation, attention) before classification/segmentation. For example, to identify “built-up” vs. “water”, the fused model learns to use texture and structure (from SAR) plus spectral cues (from optical). We fine-tune this on tasks like land-cover segmentation where inputs include both modalities (BigEarthNet’s paired imagery can be used). Alternatively, we run separate single-modality classifiers and ensemble their outputs. In all cases, combining modalities yields more reliable classification than either alone, especially under clouds or at night.

All vision models are first pre-trained on large generic datasets (e.g. ImageNet, MS COCO), then fine-tuned with remote-sensing data. In particular, we adapt Vision-Language Foundation Models (e.g. CLIP, BLIP, or open VLMs) by continuing training on BigEarthNet.txt. The BigEarthNet.txt authors report that fine-tuning with it leads to “consistent performance gains” on Earth-observation tasks. Thus, SatQuery AI’s core vision-language module is a domain-adapted model that understands satellite imagery. For tasks like grounding and VQA, we may fine-tune separate head networks or adapters on the specific annotated tasks (captions, questions, expressions).

## 3. Agentic Controller and Task Orchestration

A central **agentic controller** parses the user’s natural-language query, validates inputs, selects models, and composes the workflow:

- **Query Interpretation:** An LLM (e.g. GPT or a fine-tuned BERT) or rule-based parser classifies the query into task(s). Keywords guide this: e.g. “describe”, “list”, “what is” → captioning/VQA; “where is X”, “highlight” → grounding; “change” or “between dates” → change detection/VQA; “optical and SAR” → multi-modal fusion tasks. The parser also recognizes target classes (“built-up”, “water”, etc.) and required outputs (text answer, highlighted region, mask). This step maps the query to one or more tasks.

- **Input Validation:** Based on the query, the controller checks the number and types of images provided. For example, if a change analysis is needed but only one image is uploaded, it will prompt the user to supply a second date or switch to single-image tasks. It verifies that any pair is co-registered (same projection, overlapping bounds) and the formats are supported. If an optical–SAR pair is intended, it ensures one is clearly SAR format (e.g. single band) and one optical; otherwise it warns.

- **Model/Tool Selection:** The system maintains a **registry** of specialized models/tools. Each entry in the registry includes: *task name*, *accepted input modalities*, *model name*, and *output type*. Based on the classified task, the controller selects the appropriate model(s). For example, for "single-image VQA" it might pick `RS_VQA_Model_v1` (a fine-tuned VQA net). For a composite query like “Describe the land-cover and highlight water”, it might route to *captioning* then to *grounding* with “water” as the object. For “two images: did built-up area change?”, it selects the *change detection* model plus either a *change captioning* or *change VQA* model. 

- **Workflow Execution:** The controller sets parameters (only within allowed ranges) and runs the model(s) in sequence or parallel. Intermediate outputs are cached as needed. For multi-step queries, the outputs of one model feed into the next: e.g., a change map from the change-detection model is fed into the VQA model to answer "where did change occur?". The system enforces that only registered models and permitted settings are used (no arbitrary code execution). 

- **Output Integration:** After models run, their outputs (text answers, bounding boxes, masks) are combined. The agentic controller assembles a final answer that includes: 
  - **Textual response**: the natural-language answer or caption, drawn from the VQA/captioning model.
  - **Spatial evidence**: an image with annotated overlays (e.g. mask or bounding boxes for relevant regions, change map heatmap).
  - **Confidence estimates**: each answer or detection is accompanied by a confidence score (e.g. VQA answer probability, IoU for boxes, probability for segmentation). 
  - **Execution summary**: a report logging which tasks were performed, model names and versions used, and key parameters (e.g. probability thresholds). This summary provides an auditable trace of the processing steps.

Importantly, SatQuery AI’s *internal reasoning* (e.g. LLM chain-of-thought) is not exposed; only the execution trace and final results are reported, as required.

## 4. Fusion and Multi-Modal/Temporal Reasoning

**Multi-Modal Fusion (Optical–SAR):** To combine optical and SAR imagery, SatQuery AI can fuse data at different levels. A simple approach is to concatenate SAR (single band) and RGB (or NIR) channels as input to a single encoder. More sophisticated is dual-stream: one CNN encoder processes optical, another processes SAR, and their features are merged in a fusion layer (e.g. concatenation or cross-attention). For example, to answer “identify built-up and water regions using optical+SAR together,” the model learns that bright SAR backscatter and certain textures indicate buildings, while low SAR and spectral blue indicate water. Fusion ensures we exploit complementary strengths: optical’s spectral signatures and SAR’s cloud-penetrating structure. SatQuery AI will implement and compare both early-fusion and late-fusion strategies, choosing the best for accuracy. In either case, we train on datasets with paired inputs and ground truth (BigEarthNet.txt provides co-registered pairs) so the model learns joint representations.

**Temporal Fusion (Change Detection):** For bi-temporal analysis, SatQuery AI uses techniques like image differencing and Siamese networks. One simple method is pixel-wise differencing (absolute difference) fed into a network for segmentation. A more advanced method stacks the two images as a multi-channel input (e.g. 6 channels: RGB at T1, RGB at T2) to a CNN or Vision Transformer. A change-dedicated model (like a UNet) predicts “change” vs. “no change” masks. Additionally, attention-based architectures can explicitly compare features across time (e.g. computing a change feature map). We also explore transformer models that treat time pairs like sequences. For example, the CDVQA approach includes a “change enhancing module” to focus on differences. SatQuery AI may adopt similar ideas: using explicit difference features or learned correlation layers to better capture change cues.

After obtaining change information (mask or features), temporal fusion continues in the language domain: the change map can be used by the question-answering model as an extra “image” highlighting changed areas. This helps answer questions like “How much did built-up increase?” by comparing masked regions over time. We will train the change-detection and change-VQA components jointly where possible, using combined loss (segmentation loss + answer classification loss), to encourage coherent fusion.

## 5. Training and Evaluation

**Model Training:** Each specialist model is trained or fine-tuned on appropriate labeled data. The vision-language models start from pre-trained backbones (e.g. CLIP, ViT+transformer) and are fine-tuned on remote-sensing datasets. For example:

- The captioning model is fine-tuned on VRSBench captions and BigEarthNet.txt caption texts.
- The VQA model is trained on VRSBench QA pairs and BigEarthNet.txt VQA pairs.
- The grounding model is trained on BigEarthNet.txt referring expressions (object references).
- The change-detection model uses bi-temporal labeled pairs; if an existing RS change dataset (like LEVIR-CD) is available, we incorporate it. The model is trained with a pixel-wise binary cross-entropy or Dice loss for change mask.
- The change-VQA model uses the generated change masks and CDVQA question/answer data, trained to minimize answer error.
- The optical-SAR fusion model is trained on segmentation or classification data (possibly repurposing a land-cover dataset but with dual inputs).

During training, we monitor performance on held-out validation splits of the same datasets. For VQA and captioning, we compute language-generation metrics (BLEU, ROUGE, CIDEr) and answer accuracy. For grounding, we compute Intersection-over-Union (IoU) and mean Average Precision. For change maps, we use IoU and F1 for the “changed” class. We will also track end-to-end question-answer accuracy (does the system answer correctly) on specific test questions.

**Evaluation Metrics:** At evaluation time, we follow the prescribed criteria. Public benchmarks (VRSBench, RSVQA, CDVQA, BigEarthNet tasks) use standard metrics:
- **VQA (single-image and change-based):** answer accuracy (percent correct) or exact match. Possibly also F1 if multi-answer.
- **Captioning:** BLEU-4, CIDEr, METEOR to measure similarity to ground truth captions.
- **Grounding:** IoU of predicted bounding boxes with ground-truth boxes, or mAP if multiple objects.
- **Change map (segmentation):** IoU (Jaccard) for changed vs. unchanged, F1-score of change class.
- **Optical–SAR analysis (e.g. built-up vs. water):** segmentation accuracy or IoU on those classes.
- **Overall system:** a weighted combination of the above, normalized per rules (scores are normalized before summing).

We will validate on the official test splits of the public datasets. For ISRO/SAC evaluation (Cartosat+RISAT), hidden labels include multi-class masks, boxes, and answers. We will submit our outputs (answers, masks, boxes) and compute metrics similarly. All scores are then normalized as per the guidelines. The following table summarizes the evaluation criteria:

| **Evaluation Item**       | **Benchmark/Task**               | **Metric**                         |
|---------------------------|----------------------------------|------------------------------------|
| *Single-Image VQA*        | VRSBench / RS VQA                | Answer accuracy (exact match)      |
| *Single-Image Captioning* | VRSBench captions                | BLEU/CIDEr (language metrics)      |
| *Region Grounding*        | VRSBench referring expressions   | IoU / mAP (box IoU to GT)          |
| *Change Map Generation*   | Public RS change datasets / SAC  | IoU, F1 for changed regions        |
| *Change-based VQA*        | CDVQA (change-VQA)               | Answer accuracy                    |
| *Optical–SAR Analysis*    | SAC test (built-up/water)        | Classification accuracy / IoU      |
| *Agentic Orchestration*   | N/A (system-level)               | Qualitative correctness (completeness of summary and evidence) |

All scores from these tasks are normalized and aggregated per the official evaluation procedure. Detailed evaluation annotations (answers, masks, boxes) will not be disclosed.

## 6. User Interface and Reporting

SatQuery AI will be delivered as an interactive web application (GUI) with a remote-sensing AI backend. The GUI allows users to:

- **Upload Inputs:** Drag-and-drop or file-picker for images. The interface will clearly list supported formats (GeoTIFF/TIFF for RS images; example PNG/JPEG allowed for benchmarks). It will display a preview of each uploaded image along with metadata (e.g. date, sensor type, resolution).
- **Enter Natural-Language Query:** A text box for the user question. The system can provide example queries or tooltips. Queries can be about any supported task (captioning, Q&A, change, etc.).
- **Results Display:** After submission, the GUI shows:
  - **Textual Answer:** The system’s answer or caption, shown prominently. This text is grounded in evidence (as described below).
  - **Visual Evidence:** The input image(s) are shown side-by-side. If a region is relevant (from grounding or change), it is overlaid with a colored mask or bounding box. For change tasks, the detected change map is overlayed (e.g. red for increased built-up, blue for decreased). We include a legend.
  - **Confidence Indicators:** Next to each answer segment or detected region, a confidence score (e.g. “Confidence: 87%”) is displayed.
  - **Execution Summary:** An expandable “Details” section lists the tasks invoked, model names/versions, key parameters (e.g. model thresholds), and a timestamp. This provides auditability. For example: “Task: Change Detection (Model: ChangeNet-v2, threshold=0.5); Task: VQA (Model: RS-VQA-v1)”.
- **Downloadable Report:** Users can export results as a report (PDF or JSON). The report includes the original query, timestamp, textual answer, annotated images, confidence scores, and the execution log for auditing. This fulfills the requirement for downloadable evidence and audit trail.
- **Interactive Follow-up:** While not required, the UI may allow users to refine queries or focus (e.g. clicking on “building” in the text to highlight buildings) to deepen exploration.

## 7. System Architecture and Deployment

The backend will be a modular **agent-based pipeline**. Possible implementation technologies include:

- **Orchestration:** A controller module (written in Python or Node.js) that handles query parsing, model dispatch, and output assembly. This could use an LLM API for natural language understanding (with a locked prompt to map to tasks), plus hard-coded rules for input validation.
- **Models as Services:** Each specialist model can be a REST/HTTP service or a callable function. For instance, a Flask or FastAPI microservice for VQA, another for change detection, etc. This allows independent development and scaling.
- **Data Flow:** The controller sends images and query text to the selected services, receives JSON/text outputs, and merges them.
- **GPU Acceleration:** Heavy models (CNNs, Transformers) will run on GPU servers. The web server can be on the same machine or separate.
- **Containerization:** All components will be containerized (e.g. Docker) for portability. The final deliverable is a deployable stack (e.g. using Docker Compose or Kubernetes).
- **Performance:** We will optimize for reasonable latency (few seconds per query), e.g. using batch processing or caching results when re-used.

**Testing and Audit:** We will implement unit and integration tests for each component. For example, sample images with known outputs will verify grounding, VQA, and change detection work. The agent’s task-routing logic will be tested with a variety of query patterns. Logging is critical: every query/run is logged with inputs, tasks chosen, and model outputs (for internal debugging, not shown to end user). During development, we will validate on held-out benchmark subsets to ensure metrics. Security checks will sanitize inputs (prevent path traversal) and limit model execution parameters.

## 8. Conclusion

SatQuery AI is an **agentic, query-driven vision-language system** specifically adapted to remote-sensing imagery. By combining domain-adapted VLMs and specialist models (for captioning, grounding, VQA, change detection, and fusion), it enables non-experts to extract meaningful information via natural language. The system’s novelty lies in its orchestration: automatically parsing the query, selecting models, and integrating their outputs with visual evidence. Drawing on recent datasets (BigEarthNet.txt, VRSBench, CDVQA) and state-of-the-art methods, SatQuery AI will be trained and tested to excel on tasks ranging from single-image analysis to cross-modal and temporal reasoning. The final product will be an interactive web app with transparent reporting, ready for evaluation on benchmark datasets and real satellite imagery.

**Evaluation/Judging Criteria:** The system will be evaluated on prescribed public benchmarks (using their official test splits) and the ISRO/SAC dataset. Metrics will include answer accuracy, captioning quality (BLEU/CIDEr), grounding IoU, segmentation IoU, etc., normalized across tasks. The table below summarizes the key evaluation items and metrics:

| **Evaluation Item**                | **Benchmark/Task**            | **Metric**                         |
|------------------------------------|-------------------------------|------------------------------------|
| Single-Image VQA (optical/SAR)     | VRSBench / RS VQA             | Answer accuracy (exact match)      |
| Single-Image Captioning            | VRSBench captions             | BLEU/CIDEr (language similarity)   |
| Text-Guided Region Grounding       | VRSBench referring expressions| IoU / mAP (bounding-box overlap)   |
| Change Map Generation (binary mask)| Public change datasets / SAC  | IoU, F1 for change vs. unchanged   |
| Change-Based VQA (bi-temporal)     | CDVQA                         | Answer accuracy                    |
| Optical–SAR Joint Analysis         | SAC test (built-up/water)     | Classification accuracy / IoU      |
| Agentic Orchestration (workflow)   | N/A (system-level)            | Completeness of execution summary  |

All scores from these categories will be normalized and combined per the official rules. The ISRO/SAC evaluation uses Cartosat-2S optical and RISAT SAR pairs, with task-specific reference masks, boxes, and answers (not disclosed). SatQuery AI aims to maximize performance on these metrics by leveraging specialized models, domain adaptation, and transparent evidence. 

**Sources:** Authoritative datasets and publications guide this design. For instance, VRSBench and BigEarthNet.txt are state-of-the-art remote-sensing vision-language benchmarks, and the CDVQA task is defined in recent literature. These inform our choice of tasks, data, and evaluation criteria.