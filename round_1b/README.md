# Enhanced Persona-Driven Document Intelligence System

This is a containerized solution for analyzing PDF documents using persona-driven intelligence.

## Building the Docker Image

```bash
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .
```

## Running the Container

```bash
docker run --rm -v $(pwd)/input:/app/input -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier
```

## Input Requirements

### Directory Structure
- Place your PDF files in the `input/` directory
- Optionally, place a JSON configuration file in the `input/` directory

### Configuration File (Optional)
If no configuration file is provided, the system will automatically create a default configuration and process all PDF files found in the input directory.

Example configuration file (`config.json`):
```json
{
  "challenge_info": {
    "challenge_id": "analysis_task",
    "test_case_name": "document_analysis",
    "description": "Document analysis task"
  },
  "documents": [
    {
      "filename": "document1.pdf",
      "title": "Document 1"
    }
  ],
  "persona": {
    "role": "HR Professional"
  },
  "job_to_be_done": {
    "task": "Extract HR-related information from documents for employee management"
  }
}
```

## Output

The container will generate:
- `output.json` - Main analysis results in JSON format
- `readable_analysis_[timestamp].txt` - Human-readable analysis report

## Features

- **Automatic PDF Processing**: Processes all PDFs in the input directory
- **Semantic Analysis**: Uses sentence transformers for semantic understanding
- **Keyword Extraction**: Dynamically extracts relevant keywords
- **Section Analysis**: Identifies and ranks document sections by relevance
- **Multi-format Output**: JSON and readable text formats

## Dependencies

The container includes all necessary dependencies:
- PyMuPDF for PDF processing
- OpenCV for image processing
- Tesseract OCR for text extraction
- Sentence Transformers for semantic analysis
- scikit-learn for machine learning features
- YAKE for keyword extraction
- And more...

## System Requirements

- Docker with Linux/AMD64 platform support
- Sufficient memory for processing large PDF files
- No network access required (container runs with `--network none`)
