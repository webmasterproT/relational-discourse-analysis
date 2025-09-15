# Relational Discourse Analysis Project

## Overview
This project analyzes interpersonal communication patterns using discourse analysis techniques grounded in relational theory. The focus is on identifying evidence-based patterns in power dynamics, consent negotiation, trauma-informed communication, and relational repair processes.

## Methodology
- **Theoretical Framework**: Relational theory, discourse analysis, trauma-informed communication principles
- **Data Processing**: Automated speaker segmentation, feature extraction, pattern recognition
- **Validation**: Human review loop with audit trail for reliability
- **Output**: JSON-structured data suitable for quantitative analysis

## Project Structure
```
relational_discourse_project/
├── data/
│   ├── raw/           # Original transcript data
│   └── processed/     # Parsed and feature-enriched data
├── schemas/           # JSON schema definitions
├── scripts/           # Analysis pipeline scripts
├── analyses/          # Human review and validation
├── outputs/           # Final results and reports
└── docs/              # Documentation and methodology
```

## Key Features
- Speaker differentiation and turn-taking analysis
- Power dynamic quantification
- Consent and boundary negotiation tracking
- Trauma-responsive communication pattern detection
- Evidence linking with character-level precision

## Usage
1. Place raw conversation data in `data/raw/`
2. Run processing pipeline: `python scripts/run_pipeline.py`
3. Review flagged patterns in `analyses/REVIEW.md`
4. Generate final reports in `outputs/reports/`