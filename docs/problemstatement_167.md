# SatQuery AI — Problem Statement 167

## Background

Remote-sensing imagery is widely used for agricultural monitoring, disaster management, urban planning, forest monitoring, water-resource assessment, infrastructure mapping, and environmental analysis. However, most existing remote-sensing AI solutions are developed as isolated applications for a single predefined task, such as land-cover classification, object detection, visual question answering, or change detection. These systems often require users to understand satellite-data characteristics, GIS workflows, model selection, and task-specific parameters. Consequently, non-expert users may find it difficult to obtain meaningful information from satellite imagery through simple natural-language queries.

Many operational remote-sensing questions cannot always be answered reliably using a single optical image. Relevant information may be distributed across paired or multiple observations acquired at different times or by different sensors. Optical and multispectral imagery provides spectral and contextual information, whereas synthetic aperture radar (SAR) provides complementary structural information and supports day-and-night acquisition through cloud cover. Multitemporal image pairs are required to identify and interpret changes over time, while co-registered optical–SAR pairs can provide more complete and reliable information than either modality alone.

A general-purpose large language model (LLM) or vision-language model (VLM) cannot be expected to perform these specialised tasks reliably without adaptation to remote-sensing imagery, sensor characteristics, and domain-specific terminology. The proposed solution must therefore include remote-sensing fine-tuning or domain adaptation and may employ multiple specialised models for different tasks.

BigEarthNet.txt will serve as the primary dataset for adapting image–text representations to multisensor remote-sensing data. VRSBench and RSVQA will be used to evaluate single-image captioning, grounding, and visual question answering, while CDVQA will be used to evaluate multitemporal change-based visual question answering.

The novelty of SatQuery AI lies in its agentic, query-driven framework. Instead of applying a single generic VLM, the system selects and executes suitable remote-sensing specialist models, validates inputs, combines their outputs, and returns an evidence-grounded response.

## Description

The objective is to develop **SatQuery AI**, a software-based agentic vision-language assistant for analysing single and paired remote-sensing images through natural-language queries.

Single-image understanding is a mandatory baseline, while the principal focus is joint reasoning over paired cross-modal and multitemporal imagery.

## Defined Input Scope

### Single Image
One optical/multispectral or SAR image for:
- Captioning
- Visual question answering
- Text-guided region grounding

### Cross-Modal Pair
Co-registered optical/multispectral and SAR images of the same geographic area for joint information extraction and cross-modal analysis.

### Bi-Temporal Pair
Two spatially corresponding images of the same geographic area acquired at different times for:
- Change detection
- Change description
- Change-based visual question answering

### Supported Formats
- GeoTIFF or TIFF for geospatial imagery
- PNG and JPEG inputs may be accepted only for the prescribed public benchmark datasets

## Mandatory Functional Scope

### 1. Remote-Sensing Adaptation
At least one visual or vision-language component must be fine-tuned or otherwise adapted using BigEarthNet.txt or any open-source training data.

### 2. Single-Image Baseline
Visual question answering is mandatory.

Each solution must additionally implement either:
- Captioning/scene description, or
- Text-guided region grounding

### 3. Multi-Image Change Analysis
Change description or change-based visual question answering from a bi-temporal image pair is mandatory.

A spatial change map may also be generated where reference masks are available.

### 4. Cross-Modal Pair Analysis
The system must extract complementary information from a co-registered optical/multispectral and SAR image pair.

### 5. Agentic Orchestration
The system must automatically select, sequence, and execute the appropriate specialist models or tools according to the query and input configuration.

## Representative Queries

- “Describe the land-cover and major objects visible in this image.”
- “Highlight the water body referred to in the query.”
- “What changed between these two dates, and where did the change occur?”
- “Use the optical and SAR images together to identify built-up and water-covered regions.”
- “Has the built-up area increased, decreased, or remained unchanged?”

## Agentic Model and Tool Orchestration

The system may use multiple specialised components, such as:
- A remote-sensing VQA or captioning model
- A grounding model
- A change-understanding or change-VQA model
- An optical–SAR fusion or information-extraction model

The agentic controller should:

1. Interpret the query and classify the requested task.
2. Check the number, modality, format, metadata, and compatibility of the input images.
3. Select one or more models or tools from a predefined registry.
4. Configure only permitted task parameters and execute the selected workflow.
5. Combine textual and spatial outputs, estimate confidence, and return visual evidence.
6. Provide an auditable execution summary containing the selected task, model/tool names, and key parameters.

The controller may perform internal task planning; however, only the observable execution trace, including the selected task, models or tools, permitted parameters, and outputs will be evaluated.

**Internal reasoning text is neither required nor evaluated.**

## Expected Solution

The expected solution is an interactive GUI or web application with an agentic remote-sensing AI backend.

It should accept supported image inputs and natural-language queries, select the appropriate specialist workflow, and return evidence-grounded textual and visual results.

### The solution should include:

- Input upload and compatibility checking
- A remote-sensing-adapted vision-language component
- Specialist tools for VQA, captioning or grounding, change understanding, and optical–SAR analysis
- An agentic controller for task routing, tool execution, and output integration
- Visual evidence
- Confidence information
- Execution summaries
- Downloadable reports

Each solution must demonstrate:
1. Single-image VQA
2. One additional single-image task
3. Multitemporal change understanding
4. Optical–SAR paired-image analysis
5. Agentic model/tool orchestration

A generic LLM or VLM without remote-sensing adaptation will not satisfy the requirements.

## Deliverables

An interactive GUI or web application with an agentic remote-sensing AI backend, including:

- Codes
- Models
- Test cases
- Demonstration

## Implementation Scope

The system shall support:

- Single optical/multispectral or SAR images
- Co-registered optical–SAR pairs
- Bi-temporal image pairs
- GeoTIFF/TIFF inputs
- Approved benchmark formats

It must perform:

- Single-image VQA
- One additional single-image task
- Change analysis
- Optical–SAR joint analysis
- Agentic model/tool selection through an interactive GUI or web application

## Evaluation/Judging Criteria

Final evaluation will use prescribed public benchmark test subsets and an ISRO/SAC evaluation dataset. Scores will be normalised before combining different metrics.

### Evaluation/Judging Criteria Table

| Evaluation Area | Evaluation Scope | Potential Evidence / Output |
|---|---|---|
| Remote-Sensing Adaptation | Demonstration that at least one vision or vision-language component has been fine-tuned or domain-adapted using BigEarthNet.txt or other open-source remote-sensing training data | Training/adaptation configuration, model details, evaluation results |
| Single-Image VQA | Ability to answer questions about a single optical/multispectral or SAR image | Textual answer and supporting visual evidence |
| Captioning / Scene Description | Ability to describe the content of a single image, if selected as the additional single-image task | Generated caption/scene description |
| Text-Guided Grounding | Ability to identify image regions referred to by a natural-language query, if selected as the additional single-image task | Bounding box, mask, or highlighted region |
| Multitemporal Change Understanding | Ability to identify and describe changes between spatially corresponding images acquired at different times | Change description, answer, and optional change map |
| Change-Based VQA | Ability to answer natural-language questions concerning temporal changes | Question-answer output with evidence |
| Optical–SAR Joint Analysis | Ability to combine complementary information from co-registered optical/multispectral and SAR imagery | Joint analysis, extracted regions/classes, or evidence |
| Agentic Task Routing | Ability to interpret the query and automatically select appropriate specialist models/tools | Observable execution trace |
| Input Validation | Ability to verify image count, modality, format, metadata, and compatibility before execution | Validation status and error/warning information |
| Tool/Model Orchestration | Ability to sequence and execute multiple specialist components as required | Selected model/tool names and execution sequence |
| Evidence Grounding | Ability to connect textual responses with spatial or visual evidence | Highlighted regions, boxes, masks, or image evidence |
| Confidence Estimation | Ability to provide confidence information associated with outputs | Confidence values or calibrated confidence indicators |
| Auditability | Ability to provide an observable execution summary | Task, model/tool names, permitted parameters, and outputs |
| Public Benchmark Performance | Performance on prescribed public benchmark test subsets | Normalised benchmark metrics |
| ISRO/SAC Evaluation Performance | Performance on the official evaluation dataset | Task-specific scores against undisclosed references |
| GUI / Usability | Ability to upload inputs, enter natural-language queries, execute workflows, and inspect results | Functional interactive application |
| Reporting | Ability to export results and execution information | Downloadable report |

Public benchmarks will be evaluated using the prescribed test splits.

The ISRO/SAC evaluation set will contain pre-georeferenced and co-registered **Cartosat-2S optical** and **RISAT SAR** image pairs, with task-specific reference answers, labels, bounding boxes, or masks, as applicable.

Evaluation annotations will not be disclosed to participating teams.

## Key System Concept

**Natural-Language Query → Query Understanding → Input Validation → Task Classification → Specialist Model Selection → Tool/Model Execution → Multi-Modal/Temporal Fusion → Evidence Validation → Confidence Estimation → Auditable Response**

The core differentiator of SatQuery AI is not simply the use of a VLM, but the combination of:

- Remote-sensing domain adaptation
- Specialist model routing
- Single-image understanding
- Multitemporal reasoning
- Optical–SAR cross-modal reasoning
- Evidence-grounded outputs
- Observable and auditable agentic execution
