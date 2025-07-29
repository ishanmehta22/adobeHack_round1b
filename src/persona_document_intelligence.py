#!/usr/bin/env python3
# -- coding: utf-8 --
"""
Enhanced Persona-Driven Document Intelligence System - VS Code Version
Converted from Google Colab notebook for local VS Code execution
"""

import os
import json
import re
import time
import math
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import Counter, defaultdict


# Import required libraries
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
import pytesseract
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import yake
from summa.summarizer import summarize
import nltk
from typing import Any, Dict, List, Optional, Tuple


def setup_environment():
    """Setup environment and download required NLTK data"""
    print("ENHANCED PERSONA-DRIVEN DOCUMENT INTELLIGENCE SYSTEM")
    print("=" * 80)
    print("Setting up environment and dependencies...")
    
    # Check if CUDA is available
    try:
        import torch
        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            print(f"Current CUDA device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name()}")
    except ImportError:
        print("PyTorch not available - continuing without CUDA info")
    
    # Download NLTK data
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        print("✅ NLTK data downloaded")
    except Exception as e:
        print(f"⚠ NLTK download warning: {e}")
    
    print("✅ Environment setup complete!")

def create_project_structure():
    """Create necessary project directories"""
    # Determine project root from script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from src to project root
    
    project_dirs = [
        os.path.join(project_root, "input"),
        os.path.join(project_root, "output"),
        os.path.join(project_root, "temp")
    ]
    
    for directory in project_dirs:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    print("\n📁 Project structure ready!")

# Setup logging
def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized (console only)")
    return logger

class EnhancedDocumentAnalyzer:
    def __init__(self, logger, model_name='all-MiniLM-L6-v2'):
        """Initialize the enhanced analyzer with semantic understanding"""
        self.logger = logger
        self.model_name = model_name
        self.logger.info("Initializing Enhanced Document Analyzer...")

        # Load sentence transformer model (lightweight, fast)
        try:
            self.sentence_model = SentenceTransformer(model_name)
            self.logger.info(f"Loaded sentence transformer: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load sentence transformer: {e}")
            self.sentence_model = None

        # Initialize TF-IDF vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=0.1
        )

        # YAKE keyword extractor
        self.kw_extractor = yake.KeywordExtractor(
            lan="en",
            n=3,
            dedupLim=0.7,
            top=20
        )

        # Tesseract configuration
        self.tesseract_config = '--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?:;()[]{}"-\' '

        # Enhanced heading patterns
        self.heading_patterns = [
            r'^\d+\.\s+',  # "1. Introduction"
            r'^\d+\.\d+\s+',  # "1.1 Overview"
            r'^\d+\.\d+\.\d+\s+',  # "1.1.1 Details"
            r'^[A-Z][A-Z\s]+',  # "INTRODUCTION"
            r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:?',  # "Introduction" or "Background:"
            r'^Appendix\s+[A-Z0-9]',  # "Appendix A"
            r'^Chapter\s+\d+',  # "Chapter 1"
            r'^Section\s+\d+',  # "Section 1"
            r'^Phase\s+[IVX\d]+',  # "Phase I"
            r'^Part\s+[IVX\d]+',  # "Part I"
            r'^Step\s+\d+',  # "Step 1"
            r'^Task\s+\d+',  # "Task 1"
            r'^Lesson\s+\d+',  # "Lesson 1"
        ]

        self.logger.info("Enhanced Document Analyzer initialized successfully!")

    def safe_str_conversion(self, text: Any) -> str:
        """Safely convert any input to string"""
        if text is None:
            return ""
        if isinstance(text, (int, float, np.integer, np.floating)):
            return str(text)
        if isinstance(text, str):
            return text
        try:
            return str(text)
        except:
            return ""

    def extract_dynamic_keywords(self, persona_text: str, job_text: str) -> Dict[str, List[str]]:
        """Extract keywords dynamically from persona and job description"""
        # Safely convert inputs to strings
        persona_text = self.safe_str_conversion(persona_text)
        job_text = self.safe_str_conversion(job_text)

        combined_text = f"{persona_text} {job_text}".lower()

        # Extract keywords using YAKE
        try:
            keywords = self.kw_extractor.extract_keywords(combined_text)
            extracted_keywords = [kw[1] for kw in keywords[:15] if isinstance(kw[1], str)]
        except Exception as e:
            logger.error(f"YAKE keyword extraction failed: {e}")
            extracted_keywords = []

        # Additional manual extraction
        manual_keywords = []

        try:
            # Extract nouns and important terms
            words = re.findall(r'\b\w+\b', combined_text)
            word_freq = Counter(words)

            # Filter meaningful words (length > 3, not too common)
            for word, freq in word_freq.most_common(20):
                if len(word) > 3 and word not in ['with', 'from', 'that', 'this', 'have', 'will', 'been']:
                    manual_keywords.append(word)
        except Exception as e:
            logger.error(f"Manual keyword extraction failed: {e}")

        # Combine and deduplicate
        all_keywords = list(set(extracted_keywords + manual_keywords))

        # Categorize keywords by importance (simple heuristic)
        high_priority = []
        medium_priority = []
        context_words = []

        for keyword in all_keywords:
            keyword_str = self.safe_str_conversion(keyword).lower()
            if any(important in keyword_str for important in
                   ['create', 'manage', 'analyze', 'plan', 'design', 'develop', 'implement', 'forms', 'fillable', 'onboarding', 'compliance']):
                high_priority.append(keyword_str)
            elif any(context in keyword_str for context in
                     ['professional', 'business', 'corporate', 'academic', 'research']):
                context_words.append(keyword_str)
            else:
                medium_priority.append(keyword_str)

        return {
            'high_priority': high_priority[:10],
            'medium_priority': medium_priority[:15],
            'context_words': context_words[:10]
        }

    def extract_text_with_positions(self, pdf_path: str) -> List[Dict]:
        """Extract text with font information using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            all_text_blocks = []

            for page_num in range(min(50, len(doc))):
                page = doc[page_num]
                blocks = page.get_text("dict")

                for block in blocks["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                text = self.safe_str_conversion(span.get("text", "")).strip()
                                if text and len(text) > 2:
                                    all_text_blocks.append({
                                        'text': text,
                                        'page': page_num + 1,
                                        'font_size': float(span.get("size", 12)),
                                        'font_flags': int(span.get("flags", 0)),
                                        'bbox': span.get("bbox", [0, 0, 100, 20]),
                                        'font_name': self.safe_str_conversion(span.get("font", ""))
                                    })

            doc.close()
            return all_text_blocks
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return []

    def fallback_ocr_extraction(self, pdf_path: str) -> List[Dict]:
        """Fallback OCR method when PyMuPDF doesn't extract enough text"""
        try:
            doc = fitz.open(pdf_path)
            all_text_blocks = []

            for page_num in range(min(50, len(doc))):
                page = doc[page_num]
                mat = fitz.Matrix(1.5, 1.5)
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")

                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                try:
                    data = pytesseract.image_to_data(
                        gray,
                        config=self.tesseract_config,
                        output_type=pytesseract.Output.DICT
                    )

                    # Group words into lines
                    current_line = []
                    current_top = None
                    tolerance = 10

                    for i in range(len(data['text'])):
                        text = self.safe_str_conversion(data['text'][i]).strip()
                        conf = int(data.get('conf', [0])[i] if i < len(data.get('conf', [])) else 0)

                        if text and conf > 30:
                            top = int(data.get('top', [0])[i] if i < len(data.get('top', [])) else 0)

                            if current_top is None:
                                current_top = top
                                current_line = [text]
                            elif abs(top - current_top) <= tolerance:
                                current_line.append(text)
                            else:
                                if current_line:
                                    line_text = ' '.join(current_line)
                                    if len(line_text.strip()) > 2:
                                        height = int(data.get('height', [12])[i-1] if i > 0 and i-1 < len(data.get('height', [])) else 12)
                                        font_size = max(8, min(24, height * 0.75))

                                        all_text_blocks.append({
                                            'text': line_text,
                                            'page': page_num + 1,
                                            'font_size': float(font_size),
                                            'font_flags': 0,
                                            'bbox': [
                                                int(data.get('left', [0])[i-len(current_line)] if i-len(current_line) < len(data.get('left', [])) else 0),
                                                current_top,
                                                int(data.get('left', [0])[i-1] if i > 0 and i-1 < len(data.get('left', [])) else 100) +
                                                int(data.get('width', [100])[i-1] if i > 0 and i-1 < len(data.get('width', [])) else 100),
                                                current_top + height
                                            ],
                                            'font_name': 'OCR'
                                        })

                                current_line = [text]
                                current_top = top

                    if current_line:
                        line_text = ' '.join(current_line)
                        if len(line_text.strip()) > 2:
                            all_text_blocks.append({
                                'text': line_text,
                                'page': page_num + 1,
                                'font_size': 12.0,
                                'font_flags': 0,
                                'bbox': [0, 0, 100, 20],
                                'font_name': 'OCR'
                            })

                except Exception as e:
                    logger.error(f"OCR failed for page {page_num + 1}: {e}")
                    continue

            doc.close()
            return all_text_blocks
        except Exception as e:
            logger.error(f"Fallback OCR failed for {pdf_path}: {e}")
            return []

    def clean_text(self, text: Any) -> str:
        """Clean and normalize text, ensuring input is string"""
        text = self.safe_str_conversion(text)
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s\-\.\,\:\;\(\)\&\'\"\[\]]', '', text)
        return text

    def determine_heading_level(self, text: str, font_size: float, font_flags: int,
                              avg_font_size: float) -> Optional[str]:
        """Determine if text is a heading and what level"""
        text = self.safe_str_conversion(text).strip()
        font_size = float(font_size) if font_size is not None else 12.0
        font_flags = int(font_flags) if font_flags is not None else 0
        avg_font_size = float(avg_font_size) if avg_font_size is not None else 12.0

        if len(text) > 200 or len(text) < 3:
            return None

        # Check for specific patterns
        is_numbered_section = False
        level_from_pattern = None

        for pattern in self.heading_patterns:
            if re.match(pattern, text, re.IGNORECASE):
                is_numbered_section = True
                if re.match(r'^\d+\.\d+\.\d+', text):
                    level_from_pattern = "H3"
                elif re.match(r'^\d+\.\d+', text):
                    level_from_pattern = "H2"
                elif re.match(r'^\d+\.', text):
                    level_from_pattern = "H1"
                elif re.match(r'^Appendix|^Chapter|^Section|^Part', text, re.IGNORECASE):
                    level_from_pattern = "H1"
                elif re.match(r'^Phase|^Step|^Task|^Lesson', text, re.IGNORECASE):
                    level_from_pattern = "H2"
                break

        # Check font properties
        is_likely_bold = font_flags & 2**4
        is_larger_font = font_size > avg_font_size * 1.1

        # Heading indicators
        heading_score = 0

        if is_numbered_section:
            heading_score += 3
        if is_likely_bold:
            heading_score += 2
        if is_larger_font:
            heading_score += 2
        if text.isupper() and len(text) > 3:
            heading_score += 2
        if text.endswith(':'):
            heading_score += 1

        # Determine level
        if heading_score >= 3:
            if level_from_pattern:
                return level_from_pattern
            elif font_size > avg_font_size * 1.5:
                return "H1"
            elif font_size > avg_font_size * 1.3:
                return "H2"
            else:
                return "H3"

        return None

    def extract_sections_from_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract sections with their content from a PDF"""
        # Extract text blocks
        text_blocks = self.extract_text_with_positions(pdf_path)

        if len(text_blocks) < 10:
            logger.info(f"Low text extraction, using OCR fallback for {os.path.basename(pdf_path)}")
            text_blocks = self.fallback_ocr_extraction(pdf_path)

        if not text_blocks:
            return []

        # Calculate average font size
        font_sizes = [float(block.get('font_size', 12)) for block in text_blocks]
        avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 12.0

        # Extract headings and their content
        sections = []
        current_section = None

        for block in text_blocks:
            # Ensure text is string before cleaning
            text = self.clean_text(block.get('text', ''))

            if len(text) < 3:
                continue

            level = self.determine_heading_level(
                text,
                float(block.get('font_size', 12)),
                int(block.get('font_flags', 0)),
                avg_font_size
            )

            if level:  # This is a heading
                # Save previous section if exists
                if current_section:
                    sections.append(current_section)

                # Start new section
                current_section = {
                    'title': text,
                    'page': str(block.get('page', 'Unknown')),
                    'level': level,
                    'content': []
                }
            else:  # This is content
                if current_section:
                    current_section['content'].append({
                        'text': text,
                        'page': str(block.get('page', 'Unknown'))
                    })

        # Don't forget the last section
        if current_section:
            sections.append(current_section)

        return sections

    def calculate_semantic_relevance(self, section: Dict, persona_embedding: np.ndarray,
                                   job_embedding: np.ndarray) -> float:
        """Calculate semantic relevance using embeddings"""
        if self.sentence_model is None or persona_embedding.size == 0 or job_embedding.size == 0:
            return 0.0

        try:
            # Combine title and content text, ensuring they are strings
            all_text = self.safe_str_conversion(section.get('title', ''))
            for content_block in section.get('content', []):
                content_text = self.safe_str_conversion(content_block.get('text', ''))
                all_text += ' ' + content_text

            # Limit text length for embedding
            all_text = all_text[:2000]

            if not all_text.strip():
                return 0.0

            # Get section embedding
            section_embedding = self.sentence_model.encode([all_text])

            # Calculate cosine similarities
            persona_similarity = cosine_similarity(section_embedding, persona_embedding.reshape(1, -1))[0][0]
            job_similarity = cosine_similarity(section_embedding, job_embedding.reshape(1, -1))[0][0]

            # Weighted combination
            semantic_score = (persona_similarity * 0.4 + job_similarity * 0.6) * 10

            return max(0, float(semantic_score))
        except Exception as e:
            logger.error(f"Error calculating semantic relevance: {e}")
            return 0.0

    def calculate_keyword_relevance(self, section: Dict, keywords: Dict[str, List[str]]) -> float:
        """Calculate relevance score based on keyword matching"""
        score = 0.0

        try:
            # Combine title and content text, ensuring they are strings
            all_text = self.safe_str_conversion(section.get('title', '')).lower()

            for content_block in section.get('content', []):
                content_text = self.safe_str_conversion(content_block.get('text', ''))
                all_text += ' ' + content_text.lower()

            # Score based on high-priority keywords
            for keyword in keywords.get('high_priority', []):
                keyword_str = self.safe_str_conversion(keyword).lower()
                if keyword_str in all_text:
                    score += 3.0

            # Score based on medium-priority keywords
            for keyword in keywords.get('medium_priority', []):
                keyword_str = self.safe_str_conversion(keyword).lower()
                if keyword_str in all_text:
                    score += 1.5

            # Score based on context words
            for word in keywords.get('context_words', []):
                word_str = self.safe_str_conversion(word).lower()
                if word_str in all_text:
                    score += 0.5

        except Exception as e:
            logger.error(f"Error calculating keyword relevance: {e}")

        return float(score)

    def calculate_relevance_score(self, section: Dict, persona: str, job_description: str,
                                persona_embedding: np.ndarray, job_embedding: np.ndarray,
                                keywords: Dict[str, List[str]]) -> float:
        """Calculate comprehensive relevance score"""

        try:
            # Semantic relevance (if available)
            semantic_score = self.calculate_semantic_relevance(section, persona_embedding, job_embedding)

            # Keyword-based relevance
            keyword_score = self.calculate_keyword_relevance(section, keywords)

            # Content length bonus
            content_length = 0
            for block in section.get('content', []):
                content_text = self.safe_str_conversion(block.get('text', ''))
                content_length += len(content_text)

            length_bonus = 0.0
            if content_length > 500:
                length_bonus = 2.0
            elif content_length > 200:
                length_bonus = 1.0
            elif content_length > 100:
                length_bonus = 0.5

            # Heading level bonus
            level_bonus = 0.0
            level = section.get('level', '')
            if level == 'H1':
                level_bonus = 1.0
            elif level == 'H2':
                level_bonus = 0.7
            elif level == 'H3':
                level_bonus = 0.4

            # Combine scores
            total_score = semantic_score + keyword_score + length_bonus + level_bonus

            return float(total_score)

        except Exception as e:
            logger.error(f"Error calculating relevance score: {e}")
            return 0.0

    def improve_section_title(self, section: Dict) -> str:
        """Improve section title by cleaning and making it more meaningful"""
        title = self.safe_str_conversion(section.get('title', ''))

        # If title is too short or incomplete, try to enhance it with content
        if len(title) < 10 or title.lower().startswith(('the', 'a ', 'an ')):
            content_blocks = section.get('content', [])
            if content_blocks:
                # Get first meaningful sentence from content
                first_content = self.safe_str_conversion(content_blocks[0].get('text', ''))
                sentences = re.split(r'[.!?]+', first_content)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 10 and len(sentence) < 100:
                        # Use this as enhanced title
                        return sentence

        # Clean up the existing title
        title = re.sub(r'^[^\w]*', '', title)  # Remove leading non-word chars
        title = re.sub(r'[^\w]*$', '', title)  # Remove trailing non-word chars
        title = title.strip()

        # Capitalize properly
        if title and not title.isupper():
            title = title.title()

        return title if len(title) > 3 else f"Section from page {section.get('page', 'N/A')}"

    def extract_meaningful_content(self, section: Dict, max_length: int = 500) -> str:
        """Extract meaningful content from a section, not just the title"""
        title = self.safe_str_conversion(section.get('title', ''))
        content_blocks = section.get('content', [])

        # Combine all content
        all_content = []

        # Add title if meaningful
        if len(title) > 10:
            all_content.append(title)

        # Add content blocks
        for block in content_blocks:
            content_text = self.safe_str_conversion(block.get('text', ''))
            content_text = content_text.strip()
            if len(content_text) > 20:  # Only meaningful content
                all_content.append(content_text)

        if not all_content:
            return f"Section: {title}" if title else "No content available"

        # Join and clean
        combined_content = ' '.join(all_content)
        combined_content = re.sub(r'\s+', ' ', combined_content).strip()

        # Truncate if too long
        if len(combined_content) > max_length:
            # Try to break at sentence boundary
            sentences = re.split(r'[.!?]+', combined_content)
            truncated = []
            current_length = 0

            for sentence in sentences:
                sentence = sentence.strip()
                if current_length + len(sentence) + 1 <= max_length - 3:  # Leave room for "..."
                    truncated.append(sentence)
                    current_length += len(sentence) + 1
                else:
                    break

            if truncated:
                combined_content = '. '.join(truncated) + '.'
                if current_length < max_length - 3:
                    combined_content += '..'
            else:
                combined_content = combined_content[:max_length-3] + '...'

        return combined_content

    def extract_refined_subsections(self, sections: List[Dict], top_n: int = 5) -> List[Dict]:
        """Extract refined text from top sections for subsection analysis"""
        subsections = []

        logger.info(f"Generating subsections from top {top_n} sections out of {len(sections)} total sections")

        try:
            for i, section in enumerate(sections[:top_n]):
                logger.info(f"Processing subsection {i+1}/{min(top_n, len(sections))}")

                # Get basic section info
                document = self.safe_str_conversion(section.get('document', 'Unknown'))
                page = self.safe_str_conversion(section.get('page', 'Unknown'))

                # Extract meaningful content instead of just title
                refined_text = self.extract_meaningful_content(section, max_length=800)

                logger.info(f"  Document: {document}")
                logger.info(f"  Page: {page}")
                logger.info(f"  Content length: {len(refined_text)} characters")

                # Create subsection entry (matching the expected format)
                subsection = {
                    'document': document,
                    'refined_text': refined_text,
                    'page_number': page
                }

                subsections.append(subsection)
                logger.info(f"  ✓ Successfully added subsection {i+1}")

        except Exception as e:
            logger.error(f"Error in extract_refined_subsections: {e}")
            logger.error(f"Sections data type: {type(sections)}")
            if sections:
                logger.error(f"First section keys: {list(sections[0].keys()) if sections[0] else 'None'}")

        logger.info(f"Subsection extraction completed: {len(subsections)} subsections generated")

        return subsections

    def process_documents(self, collection_path: str, input_config: Dict) -> Dict:
        """Process all documents from a collection and generate output"""
        logger.info("Starting document processing...")

        try:
            # Extract configuration
            persona = self.safe_str_conversion(input_config.get('persona', {}).get('role', ''))
            job_task = self.safe_str_conversion(input_config.get('job_to_be_done', {}).get('task', ''))
            documents = input_config.get('documents', [])

            logger.info(f"Persona: {persona}")
            logger.info(f"Task: {job_task}")
            logger.info(f"Processing {len(documents)} documents...")

            # Generate dynamic keywords
            keywords = self.extract_dynamic_keywords(persona, job_task)
            logger.info(f"Extracted keywords: {len(keywords.get('high_priority', []))} high priority, {len(keywords.get('medium_priority', []))} medium priority")

            # Generate embeddings for persona and job (if model available)
            persona_embedding = np.array([])
            job_embedding = np.array([])
            if self.sentence_model and persona.strip() and job_task.strip():
                try:
                    persona_embedding = self.sentence_model.encode([persona])
                    job_embedding = self.sentence_model.encode([job_task])
                    logger.info("Generated semantic embeddings")
                except Exception as e:
                    logger.error(f"Error generating embeddings: {e}")

            # Process documents from PDFs folder
            start_time = time.time()
            all_sections = []
            pdfs_folder = os.path.join(collection_path, "PDFs")

            for i, doc_info in enumerate(documents, 1):
                filename = self.safe_str_conversion(doc_info.get('filename', ''))
                pdf_path = os.path.join(pdfs_folder, filename)

                logger.info(f"Processing {i}/{len(documents)}: {filename}")

                if not os.path.exists(pdf_path):
                    logger.warning(f"File not found - skipping {filename}")
                    continue

                try:
                    # Extract sections from PDF
                    sections = self.extract_sections_from_pdf(pdf_path)
                    logger.info(f"Extracted {len(sections)} sections from {filename}")

                    # Add document info and calculate relevance scores
                    for section in sections:
                        # Ensure necessary fields are strings before calculating relevance
                        section['document'] = filename
                        section['title'] = self.safe_str_conversion(section.get('title', ''))
                        section['page'] = self.safe_str_conversion(section.get('page', 'Unknown'))

                        # Ensure content text is properly formatted
                        cleaned_content = []
                        for content_block in section.get('content', []):
                            content_text = self.safe_str_conversion(content_block.get('text', ''))
                            page_num = self.safe_str_conversion(content_block.get('page', 'Unknown'))
                            cleaned_content.append({'text': content_text, 'page': page_num})
                        section['content'] = cleaned_content

                        section['relevance_score'] = self.calculate_relevance_score(
                            section, persona, job_task,
                            persona_embedding[0] if persona_embedding.size > 0 else np.array([]),
                            job_embedding[0] if job_embedding.size > 0 else np.array([]),
                            keywords
                        )

                    all_sections.extend(sections)

                except Exception as e:
                    logger.error(f"Error processing {filename}: {e}")
                    continue

            # Sort sections by relevance score (descending)
            all_sections.sort(key=lambda x: float(x.get('relevance_score', 0)), reverse=True)

            processing_time = time.time() - start_time
            logger.info(f"Processing completed in {processing_time:.2f} seconds")
            logger.info(f"Total sections extracted: {len(all_sections)}")

            # Extract top sections for output with improved titles
            top_sections = []
            for i, section in enumerate(all_sections[:5]): # Top 5 sections
                improved_title = self.improve_section_title(section)
                top_sections.append({
                    'document': self.safe_str_conversion(section.get('document', 'Unknown')),
                    'section_title': improved_title,
                    'importance_rank': i + 1,
                    'page_number': self.safe_str_conversion(section.get('page', 'Unknown'))
                })

            # Extract subsection analysis - This is the key fix
            subsection_analysis = self.extract_refined_subsections(all_sections, top_n=5)
            logger.info(f"Generated {len(subsection_analysis)} subsection analyses")

            # Prepare final output
            output_result = {
                'metadata': {
                    'input_documents': [self.safe_str_conversion(doc.get('filename', 'Unknown')) for doc in documents],
                    'persona': persona,
                    'job_to_be_done': job_task,
                    'processing_timestamp': datetime.now().isoformat()
                },
                'extracted_sections': top_sections,
                'subsection_analysis': subsection_analysis
            }

            return output_result

        except Exception as e:
            logger.error(f"Error in process_documents: {e}")
            return {
                'metadata': {
                    'input_documents': [],
                    'persona': '',
                    'job_to_be_done': '',
                    'processing_timestamp': datetime.now().isoformat(),
                    'error': str(e)
                },
                'extracted_sections': [],
                'subsection_analysis': []
            }


def get_collections_from_input(input_base_folder: str) -> List[str]:
    """Get all collection folders from input directory"""
    collections = []
    if os.path.exists(input_base_folder):
        for item in os.listdir(input_base_folder):
            item_path = os.path.join(input_base_folder, item)
            if os.path.isdir(item_path) and item.startswith('Collection'):
                collections.append(item)
    return sorted(collections)


def load_collection_config(collection_path: str) -> Optional[Dict]:
    """Load challenge1b_input.json configuration from collection folder"""
    config_file = os.path.join(collection_path, "challenge1b_input.json")
    
    if not os.path.exists(config_file):
        print(f"No challenge1b_input.json found in {collection_path}")
        return None
    
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"Successfully loaded config from: {config_file}")
        return config
    except Exception as e:
        print(f"Error loading config file {config_file}: {e}")
        return None


def get_pdfs_from_collection(collection_path: str) -> List[str]:
    """Get all PDF files from the PDFs folder in a collection"""
    pdfs_folder = os.path.join(collection_path, "PDFs")
    pdf_files = []
    
    if os.path.exists(pdfs_folder):
        for file in os.listdir(pdfs_folder):
            if file.endswith('.pdf'):
                pdf_files.append(file)
    
    return sorted(pdf_files)


def save_collection_results(result: Dict, output_base_folder: str, collection_name: str) -> str:
    """Save analysis results for a specific collection"""
    collection_output_dir = os.path.join(output_base_folder, collection_name)
    os.makedirs(collection_output_dir, exist_ok=True)
    
    # Save as challenge1b_output.json
    json_path = os.path.join(collection_output_dir, "challenge1b_output.json")
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    
    print(f"Collection {collection_name} output saved to: {json_path}")
    
    return json_path


def load_config_from_file(input_folder: str) -> Optional[Dict]:
    """Load configuration from JSON file in input folder"""
    config_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]
    
    if not config_files:
        print(f"No JSON config files found in {input_folder}")
        print("Please create a config file based on the sample_config.json")
        return None
    
    if len(config_files) == 1:
        config_file = config_files[0]
        print(f"Using config file: {config_file}")
    else:
        print(f"Found {len(config_files)} config files:")
        for i, filename in enumerate(config_files, 1):
            print(f"  {i}. {filename}")
        
        while True:
            try:
                choice = int(input("Select config file (number): ")) - 1
                if 0 <= choice < len(config_files):
                    config_file = config_files[choice]
                    break
                else:
                    print("Invalid choice. Please try again.")
            except ValueError:
                print("Please enter a valid number.")
    
    config_path = os.path.join(input_folder, config_file)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        print(f"Successfully loaded config from: {config_file}")
        return config
    except Exception as e:
        print(f"Error loading config file {config_file}: {e}")
        return None


def validate_config(config: Dict, input_folder: str) -> bool:
    """Validate configuration and check if PDF files exist"""
    try:
        # Check required fields
        if 'persona' not in config or 'role' not in config['persona']:
            print("Config missing persona.role")
            return False
        
        if 'job_to_be_done' not in config or 'task' not in config['job_to_be_done']:
            print("Config missing job_to_be_done.task")
            return False
        
        if 'documents' not in config or not config['documents']:
            print("Config missing documents list")
            return False
        
        # Check if PDF files exist
        available_pdfs = set(os.listdir(input_folder))
        missing_files = []
        
        for doc in config['documents']:
            filename = doc.get('filename', '')
            if filename not in available_pdfs:
                missing_files.append(filename)
        
        if missing_files:
            print(f"Missing PDF files: {', '.join(missing_files)}")
            print(f"Please place these files in the '{input_folder}' directory")
            return False

        print("Configuration validated successfully")
        return True
        
    except Exception as e:
        print(f"Error validating config: {e}")
        return False


def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        'fitz',  # PyMuPDF
        'cv2',   # opencv-python
        'PIL',   # Pillow
        'pytesseract',
        'sentence_transformers',
        'sklearn',
        'yake',
        'summa',
        'nltk',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'fitz':
                import fitz
            elif package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            elif package == 'pytesseract':
                import pytesseract
            elif package == 'sentence_transformers':
                from sentence_transformers import SentenceTransformer
            elif package == 'sklearn':
                from sklearn.metrics.pairwise import cosine_similarity
            elif package == 'yake':
                import yake
            elif package == 'summa':
                from summa.summarizer import summarize
            elif package == 'nltk':
                import nltk
            elif package == 'numpy':
                import numpy
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for package in missing_packages:
            if package == 'fitz':
                print("  - PyMuPDF (install with: pip install PyMuPDF)")
            elif package == 'cv2':
                print("  - opencv-python (install with: pip install opencv-python)")
            elif package == 'PIL':
                print("  - Pillow (install with: pip install Pillow)")
            else:
                print(f"  - {package} (install with: pip install {package})")
        return False
    
    print("All required dependencies are installed")
    return True


def main():
    """Main execution function"""
    print("ENHANCED PERSONA-DRIVEN DOCUMENT INTELLIGENCE SYSTEM")
    print("=" * 80)
    
    # Check dependencies first
    if not check_dependencies():
        print("\nPlease install missing dependencies and run again.")
        print("You can install all at once with:")
        print("pip install PyMuPDF opencv-python Pillow pytesseract sentence-transformers scikit-learn yake summa-eval nltk numpy")
        return
    
    # Setup environment and project structure
    setup_environment()
    create_project_structure()
    
    # Setup logging
    global logger
    logger = setup_logging()
    
    # Define folder paths - adjust for running from src folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Go up one level from src to project root
    input_base_folder = os.path.join(project_root, "input")
    output_base_folder = os.path.join(project_root, "output")
    
    print(f"Script directory: {script_dir}")
    print(f"Project root: {project_root}")
    print(f"Input folder: {input_base_folder}")
    print(f"Output folder: {output_base_folder}")
    
    # Get all collections from input directory
    collections = get_collections_from_input(input_base_folder)
    
    if not collections:
        print(f"\nNo collection folders found in '{input_base_folder}' directory")
        print("Please create collection folders (Collection 1, Collection 2, etc.) with PDFs folder and challenge1b_input.json")
        return
    
    print(f"\nFound {len(collections)} collections:")
    for i, collection in enumerate(collections, 1):
        print(f"  {i}. {collection}")
    
    # Process each collection
    total_collections = len(collections)
    successful_collections = 0
    
    for collection_num, collection_name in enumerate(collections, 1):
        print(f"\n{'='*60}")
        print(f"🔍 PROCESSING {collection_name.upper()} ({collection_num}/{total_collections})")
        print(f"{'='*60}")
        
        collection_path = os.path.join(input_base_folder, collection_name)
        
        # Load configuration for this collection
        config = load_collection_config(collection_path)
        if not config:
            print(f"Skipping {collection_name} due to config issues")
            continue
        
        # Get PDFs for this collection
        pdf_files = get_pdfs_from_collection(collection_path)
        if not pdf_files:
            print(f"No PDF files found in {collection_name}/PDFs folder")
            continue
        
        print(f"Found {len(pdf_files)} PDF files in {collection_name}/PDFs:")
        for i, filename in enumerate(pdf_files, 1):
            print(f"  {i}. {filename}")
        
        # Validate configuration against actual PDFs
        documents_in_config = config.get('documents', [])
        config_pdfs = [doc.get('filename', '') for doc in documents_in_config]
        
        # Check if all PDFs in config exist in PDFs folder
        missing_pdfs = [pdf for pdf in config_pdfs if pdf not in pdf_files]
        if missing_pdfs:
            print(f"Warning: Some PDFs in config not found in PDFs folder: {missing_pdfs}")
        
        # Update config to include all PDFs found (in case config is incomplete)
        all_documents = []
        for pdf_file in pdf_files:
            # Check if this PDF is already in config
            existing_doc = next((doc for doc in documents_in_config if doc.get('filename') == pdf_file), None)
            if existing_doc:
                all_documents.append(existing_doc)
            else:
                # Add missing PDF with default title
                all_documents.append({
                    'filename': pdf_file,
                    'title': pdf_file.replace('.pdf', '').replace('_', ' ').title()
                })
        
        config['documents'] = all_documents
        
        # Display configuration summary
        print(f"\nCONFIGURATION SUMMARY FOR {collection_name}")
        print("=" * 50)
        print(f"Persona: {config.get('persona', {}).get('role', 'Not specified')}")
        print(f"Task: {config.get('job_to_be_done', {}).get('task', 'Not specified')}")
        print(f"Documents to process: {len(config['documents'])}")
        
        for i, doc in enumerate(config['documents'], 1):
            print(f"  {i}. {doc['filename']} - {doc.get('title', 'No title')}")
        
        # Initialize analyzer and process documents
        print(f"\n🔍 Starting document analysis for {collection_name}...")
        analyzer = EnhancedDocumentAnalyzer(logger)
        
        start_time = time.time()
        result = analyzer.process_documents(collection_path, config)
        processing_time = time.time() - start_time
        
        if result and result.get('extracted_sections'):
            print(f"\nProcessing time for {collection_name}: {processing_time:.2f} seconds")
            
            # Display results summary
            print(f"\nANALYSIS RESULTS FOR {collection_name}")
            print("=" * 50)
            print(f"Persona: {result['metadata']['persona']}")
            print(f"Job to be done: {result['metadata']['job_to_be_done']}")
            print(f"Documents processed: {len(result['metadata']['input_documents'])}")
            print(f"Top sections extracted: {len(result['extracted_sections'])}")
            print(f"Subsections analyzed: {len(result['subsection_analysis'])}")

            print(f"\nEXTRACTED SECTIONS (Top 5):")
            print("-" * 50)
            for section in result['extracted_sections']:
                print(f"Rank {section['importance_rank']}: {section['section_title']}")
                print(f"{section['document']} (Page {section['page_number']})")
            
            # Save results for this collection
            output_path = save_collection_results(result, output_base_folder, collection_name)
            
            print(f"\n{collection_name} analysis complete!")
            print(f"Results saved to: output/{collection_name}/challenge1b_output.json")
            successful_collections += 1
            
        else:
            print(f"Analysis failed for {collection_name}")
            if result.get('metadata', {}).get('error'):
                print(f"Error: {result['metadata']['error']}")
    
    # Final summary
    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETE!")
    print(f"Successfully processed: {successful_collections}/{total_collections} collections")
    print(f"All results saved in '{output_base_folder}' directory")

    if successful_collections > 0:
        print(f"\nOutput structure:")
        for collection in collections:
            output_file = os.path.join(output_base_folder, collection, "challenge1b_output.json")
            if os.path.exists(output_file):
                print(f"{output_base_folder}/{collection}/challenge1b_output.json")
            else:
                print(f"{output_base_folder}/{collection}/challenge1b_output.json (failed)")

    print(f"{'='*80}")


if __name__ == "__main__":
    main()