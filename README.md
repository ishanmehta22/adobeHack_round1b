# Enhanced Persona-Driven Document Intelligence System

A robust, containerized solution for analyzing PDF documents using persona-driven intelligence and semantic analysis.

---

## Features

- **Automatic PDF Processing:** Scans and analyzes all PDFs in the input collections.
- **Persona & Task Driven:** Customizes analysis based on persona and job-to-be-done from JSON config.
- **Semantic Section Ranking:** Uses sentence transformers and keyword extraction to rank document sections by relevance.
- **Fallback OCR:** Extracts text from scanned PDFs using Tesseract OCR if needed.
- **Multi-format Output:** Produces structured JSON output for downstream use.

---

## Folder Structure

```
round_1b/
  approach_explanation.md
  Dockerfile
  README.md
  requirements.txt
  input/
    Collection 1/
      challenge1b_input.json
      PDFs/
        *.pdf
    Collection 2/
      challenge1b_input.json
      PDFs/
        *.pdf
    Collection 3/
      challenge1b_input.json
      PDFs/
        *.pdf
  output/
    Collection 1/
      challenge1b_output.json
    Collection 2/
      challenge1b_output.json
    Collection 3/
      challenge1b_output.json
  src/
    persona_document_intelligence.py
```

---

## Quick Start

### 1. Build the Docker Image

```sh
docker build --platform linux/amd64 -t persona-doc-intel:latest .
```

### 2. Prepare Input

- Place PDFs in `input/Collection X/PDFs/`
- Add a `challenge1b_input.json` config file in each collection folder (see below for format)

### 3. Run the Container

```sh
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  persona-doc-intel:latest
```

---

## Input Format

Each collection folder must contain:

- `challenge1b_input.json` (see example below)
- `PDFs/` folder with relevant PDF files

**Sample `challenge1b_input.json`:**
```json
{
  "challenge_info": {
    "challenge_id": "round_1b_001",
    "test_case_name": "menu_planning",
    "description": "Dinner menu planning"
  },
  "documents": [
    { "filename": "Dinner Ideas - Sides_1.pdf", "title": "Dinner Ideas - Sides_1" }
  ],
  "persona": { "role": "Food Contractor" },
  "job_to_be_done": { "task": "Prepare a vegetarian buffet-style dinner menu for a corporate gathering, including gluten-free items." }
}
```

---

## Output

For each collection, the system generates:

- `output/Collection X/challenge1b_output.json`  
  Contains metadata, top extracted sections, and subsection analysis.

---

## Dependencies

All dependencies are installed via [requirements.txt](requirements.txt):

- PyMuPDF
- OpenCV
- Pillow
- pytesseract
- sentence-transformers
- scikit-learn
- yake
- summa
- nltk
- numpy
- torch
- transformers

System dependencies (installed in Dockerfile):

- tesseract-ocr
- tesseract-ocr-eng
- libgl1-mesa-glx
- libglib2.0-0
- libsm6
- libxext6
- libxrender-dev
- libgomp1
- libgcc-s1

---

## How It Works

1. **Startup:** Checks dependencies, downloads NLTK data, sets up folders.
2. **Input Discovery:** Finds all collections in `input/`.
3. **Config Loading:** Loads persona/task and document list from each collection's JSON.
4. **PDF Analysis:** Extracts text, identifies sections/headings, ranks by relevance.
5. **Semantic Scoring:** Uses sentence transformers and keyword matching for ranking.
6. **Output:** Saves results to `output/Collection X/challenge1b_output.json`.
