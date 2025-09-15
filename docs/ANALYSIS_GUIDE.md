# Advanced Relational Discourse Analysis Guide

## Overview

This framework provides sophisticated analysis capabilities for understanding relational communication patterns, power dynamics, and trauma-informed interactions. The system goes beyond basic pattern detection to offer temporal analysis, longitudinal tracking, and interactive exploration tools.

## Table of Contents

1. [Basic Analysis Workflow](#basic-analysis-workflow)
2. [Advanced Analysis Modules](#advanced-analysis-modules)
3. [Further Analysis Possibilities](#further-analysis-possibilities)
4. [Research Applications](#research-applications)
5. [Integration with Other Systems](#integration-with-other-systems)

---

## Basic Analysis Workflow

### 1. Initial Setup & Pattern Detection

```bash
# 1. Parse conversation data
python scripts/01_parse_split.py

# 2. Run pattern detection
python scripts/03_pattern_tagging.py

# 3. Generate basic report
# Report is automatically created in outputs/reports/
```

**Output:** Speaker-differentiated JSON files, pattern annotations, comprehensive analysis report

### 2. Launch Interactive Dashboard

```bash
# Install dashboard dependencies
pip install streamlit plotly

# Launch interactive dashboard  
python scripts/launch_dashboard.py
```

**Features:**
- Pattern timeline visualization
- Evidence viewer with highlighting
- Safety assessment panel
- Speaker comparison charts
- Statistical analysis tables

---

## Advanced Analysis Modules

### 1. Temporal Pattern Analysis

**Purpose:** Analyze how communication patterns evolve within a single conversation.

```bash
python scripts/05_temporal_analysis.py
```

**Key Capabilities:**
- **Escalation Detection:** Identifies moments where pattern severity increases
- **Conversation Phases:** Segments conversation into distinct communicative phases  
- **Pattern Clustering:** Detects which patterns tend to co-occur
- **Speaker Dynamics:** Tracks how each speaker's patterns change over time

**Output Example:**
```json
{
  "escalation_points": [
    {
      "turn": 13,
      "escalation_type": "severity",
      "from_severity": 2,
      "to_severity": 4
    }
  ],
  "conversation_phases": [
    {
      "phase_number": 1,
      "turn_start": 1,
      "turn_end": 6,
      "dominant_pattern": "accountability_taking"
    }
  ]
}
```

### 2. Longitudinal Relationship Analysis

**Purpose:** Track relationship patterns across multiple conversations over time.

```python
from scripts.longitudinal_analysis import LongitudinalAnalyzer, ConversationSession

analyzer = LongitudinalAnalyzer("/path/to/project")

# Add multiple conversation sessions
session1 = ConversationSession(
    session_id="conversation_001",
    date=datetime(2024, 1, 15),
    participants=["PERSON_A", "PERSON_B"],
    utterances_file="data/processed/conv_001_utterances.json",
    annotations_file="outputs/json/conv_001_annotations.json",
    context={"setting": "therapy", "topic": "boundary_setting"}
)

analyzer.add_conversation_session(session1)
# ... add more sessions

# Analyze evolution
evolution = analyzer.analyze_pattern_evolution()
```

**Key Capabilities:**
- **Repair Progress Tracking:** Measures improvement in accountability/trauma responsiveness
- **Deterioration Signals:** Detects increasing power imbalances or boundary violations
- **Critical Incident Detection:** Identifies conversations with unusual pattern activity
- **Relationship Phase Identification:** Maps relationship through distinct phases

### 3. Statistical Significance Testing

**Add to existing analysis:**

```python
from scipy.stats import chi2_contingency, fisher_exact
import numpy as np

def test_pattern_significance(pattern_summary):
    """Test if pattern distribution differs significantly between speakers."""
    
    # Create contingency table
    patterns = list(pattern_summary.keys())
    person_a_counts = [pattern_summary[p]['speakers']['PERSON_A'] for p in patterns]
    person_b_counts = [pattern_summary[p]['speakers']['PERSON_B'] for p in patterns]
    
    contingency_table = np.array([person_a_counts, person_b_counts])
    
    # Chi-square test
    chi2, p_value, dof, expected = chi2_contingency(contingency_table)
    
    return {
        'chi2_statistic': chi2,
        'p_value': p_value,
        'significant': p_value < 0.05,
        'effect_size': np.sqrt(chi2 / np.sum(contingency_table))
    }
```

---

## Further Analysis Possibilities

### 1. Cross-Pattern Correlation Analysis

Identify which patterns predict or correlate with others:

```python
def analyze_pattern_correlations(annotations):
    """Analyze correlations between different pattern types."""
    
    # Group patterns by utterance/turn
    turn_patterns = {}
    for ann in annotations:
        turn = ann['utterance_id']
        if turn not in turn_patterns:
            turn_patterns[turn] = []
        turn_patterns[turn].append(ann['pattern_type'])
    
    # Calculate co-occurrence matrix
    pattern_types = list(set(ann['pattern_type'] for ann in annotations))
    cooccurrence = np.zeros((len(pattern_types), len(pattern_types)))
    
    for patterns in turn_patterns.values():
        for i, p1 in enumerate(pattern_types):
            for j, p2 in enumerate(pattern_types):
                if p1 in patterns and p2 in patterns:
                    cooccurrence[i, j] += 1
    
    return cooccurrence, pattern_types
```

### 2. Semantic Analysis Integration

Enhance pattern detection with semantic understanding:

```python
from transformers import pipeline, AutoTokenizer, AutoModel
import torch

class SemanticPatternAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = AutoModel.from_pretrained("bert-base-uncased")
    
    def analyze_utterance_semantics(self, text):
        # Sentiment analysis
        sentiment = self.sentiment_analyzer(text)[0]
        
        # Semantic embeddings
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return {
            'sentiment': sentiment,
            'semantic_embedding': embeddings.numpy(),
            'emotional_intensity': sentiment['score']
        }
```

### 3. Predictive Modeling

Build models to predict conversation outcomes:

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def build_outcome_predictor(conversation_features):
    """Predict conversation outcomes based on early patterns."""
    
    # Features: early conversation patterns (first 25% of turns)
    # Target: overall conversation outcome (repair success, escalation, etc.)
    
    X = []  # Feature vectors (pattern counts, speaker ratios, etc.)
    y = []  # Outcome labels
    
    # Extract features and train model
    for conv in conversation_features:
        early_patterns = extract_early_patterns(conv, proportion=0.25)
        outcome = classify_outcome(conv)
        
        X.append(early_patterns)
        y.append(outcome)
    
    # Train predictive model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    return model
```

### 4. Network Analysis of Communication Patterns

Analyze conversation as a network of connected utterances:

```python
import networkx as nx

def create_conversation_network(utterances, annotations):
    """Create network representation of conversation flow."""
    
    G = nx.DiGraph()
    
    # Add nodes (utterances)
    for utterance in utterances:
        G.add_node(
            utterance['utterance_id'],
            speaker=utterance['speaker'],
            word_count=utterance['word_count'],
            patterns=[ann['pattern_type'] for ann in annotations 
                     if ann['utterance_id'] == utterance['utterance_id']]
        )
    
    # Add edges (conversation flow)
    for i in range(len(utterances) - 1):
        current_id = utterances[i]['utterance_id']
        next_id = utterances[i + 1]['utterance_id']
        G.add_edge(current_id, next_id)
    
    # Calculate network metrics
    centrality = nx.betweenness_centrality(G)
    clustering = nx.clustering(G)
    
    return G, {'centrality': centrality, 'clustering': clustering}
```

---

## Research Applications

### 1. Clinical Research

**Therapeutic Relationship Assessment:**
- Track repair attempts over multiple therapy sessions
- Identify patterns that predict therapeutic alliance strength
- Monitor trauma-informed practice adherence

**Implementation:**
```python
def assess_therapeutic_relationship(session_data):
    """Assess therapeutic relationship quality from conversation patterns."""
    
    therapeutic_indicators = {
        'empathy_score': count_pattern_proportion(session_data, 'trauma_responsive'),
        'power_balance': calculate_speaker_equality(session_data),
        'boundary_respect': 1 - count_pattern_proportion(session_data, 'boundary_violation'),
        'repair_capacity': count_pattern_proportion(session_data, 'accountability_taking')
    }
    
    # Composite therapeutic relationship score
    tr_score = np.mean(list(therapeutic_indicators.values()))
    
    return therapeutic_indicators, tr_score
```

### 2. Legal/Forensic Applications  

**Evidence-Based Pattern Documentation:**
- Character-level evidence linking for legal proceedings
- Objective measurement of communication dynamics
- Trend analysis for pattern of behavior documentation

### 3. Relationship Counseling

**Couple Communication Assessment:**
- Real-time pattern feedback during sessions
- Progress tracking between sessions  
- Identification of recurring problematic patterns

### 4. Training and Education

**Communication Skills Training:**
- Objective measurement of trauma-informed communication
- Before/after training effectiveness assessment
- Skill development tracking over time

---

## Integration with Other Systems

### 1. Real-Time Analysis Integration

```python
class RealTimeAnalyzer:
    def __init__(self, pattern_analyzer):
        self.analyzer = pattern_analyzer
        self.current_conversation = []
    
    def process_utterance(self, speaker, text, timestamp):
        """Process utterance in real-time."""
        
        utterance = {
            'speaker': speaker,
            'text': text,
            'timestamp': timestamp,
            'utterance_id': f"UTT_{len(self.current_conversation):04d}"
        }
        
        # Detect patterns immediately
        patterns = self.analyzer.detect_patterns(utterance)
        
        # Alert for high-risk patterns
        for pattern in patterns:
            if pattern.confidence > 0.8 and pattern.pattern_type in ['boundary_violation', 'power_imbalance']:
                self.send_alert(pattern)
        
        self.current_conversation.append(utterance)
        return patterns
    
    def send_alert(self, pattern):
        """Send real-time alert for concerning patterns."""
        print(f"🚨 ALERT: {pattern.pattern_type} detected with {pattern.confidence:.2f} confidence")
```

### 2. Database Integration

```python
import sqlite3
import json

class DiscourseDatabase:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.create_tables()
    
    def create_tables(self):
        """Create database schema for discourse analysis."""
        
        self.conn.executescript('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                date TEXT,
                participants TEXT,
                context TEXT
            );
            
            CREATE TABLE IF NOT EXISTS utterances (
                id TEXT PRIMARY KEY,
                conversation_id TEXT,
                speaker TEXT,
                text TEXT,
                turn_number INTEGER,
                word_count INTEGER,
                timestamp TEXT,
                FOREIGN KEY (conversation_id) REFERENCES conversations (id)
            );
            
            CREATE TABLE IF NOT EXISTS patterns (
                id TEXT PRIMARY KEY,
                utterance_id TEXT,
                pattern_type TEXT,
                confidence REAL,
                severity TEXT,
                evidence_start INTEGER,
                evidence_end INTEGER,
                theoretical_basis TEXT,
                FOREIGN KEY (utterance_id) REFERENCES utterances (id)
            );
        ''')
    
    def store_analysis(self, conversation_id, utterances, annotations):
        """Store complete analysis in database."""
        
        for utterance in utterances:
            self.conn.execute('''
                INSERT OR REPLACE INTO utterances 
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                utterance['utterance_id'],
                conversation_id,
                utterance['speaker'],
                utterance['text'],
                utterance['turn_number'],
                utterance['word_count'],
                utterance.get('timestamp')
            ))
        
        for annotation in annotations:
            evidence = annotation['evidence_span']
            self.conn.execute('''
                INSERT OR REPLACE INTO patterns
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                annotation['annotation_id'],
                annotation['utterance_id'],
                annotation['pattern_type'],
                annotation['confidence'],
                annotation['severity'],
                evidence.get('start_char'),
                evidence.get('end_char'),
                annotation['theoretical_basis']
            ))
        
        self.conn.commit()
```

### 3. API Integration

```python
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze_conversation():
    """API endpoint for conversation analysis."""
    
    data = request.get_json()
    conversation_text = data['conversation']
    
    # Parse and analyze
    parser = ConversationParser()
    utterances = parser.parse_conversation_text(conversation_text)
    
    analyzer = RelationalPatternAnalyzer("schemas/pattern_rules.yaml")
    annotations = []
    
    for utterance in utterances:
        matches = analyzer.detect_patterns(utterance)
        annotations.extend(matches)
    
    # Return results
    return jsonify({
        'utterances': len(utterances),
        'patterns_detected': len(annotations),
        'safety_score': calculate_safety_score(annotations),
        'repair_score': calculate_repair_score(annotations),
        'detailed_results': {
            'utterances': utterances,
            'annotations': [ann.__dict__ for ann in annotations]
        }
    })

if __name__ == '__main__':
    app.run(debug=True)
```

---

## Conclusion

This framework provides a foundation for sophisticated relational discourse analysis with multiple pathways for extension and integration. The modular design allows researchers and practitioners to:

1. **Start Simple:** Basic pattern detection and evidence linking
2. **Scale Up:** Add temporal analysis and longitudinal tracking  
3. **Integrate:** Connect with databases, real-time systems, and APIs
4. **Specialize:** Adapt for clinical, legal, educational, or research contexts

The evidence-based approach ensures that all findings are traceable to specific text segments, making the analysis transparent and verifiable for both research and applied contexts.

For questions, contributions, or custom implementations, see the GitHub repository:
**https://github.com/webmasterproT/relational-discourse-analysis**