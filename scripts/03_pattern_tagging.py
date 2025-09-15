#!/usr/bin/env python3
"""
Pattern detection and annotation for relational discourse analysis.
Applies theory-based rules to identify communication patterns.
"""

import json
import yaml
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import uuid

@dataclass
class PatternMatch:
    pattern_type: str
    confidence: float
    evidence_span: Tuple[int, int]
    text_segment: str
    contextual_factors: List[str]
    theoretical_basis: str

class RelationalPatternAnalyzer:
    def __init__(self, rules_path: str):
        with open(rules_path, 'r') as f:
            self.rules = yaml.safe_load(f)
        
        self.patterns = self.rules['patterns']
        self.linguistic_markers = self.rules['linguistic_markers']
        self.contextual_modifiers = self.rules['contextual_modifiers']
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Pre-compile regex patterns for faster matching."""
        self.compiled_patterns = {}
        
        for pattern_name, pattern_config in self.patterns.items():
            if 'keyword_patterns' in str(pattern_config.get('indicators', [])):
                # Extract keyword patterns and compile them
                keywords = []
                for indicator in pattern_config.get('indicators', []):
                    if isinstance(indicator, dict) and 'keyword_patterns' in indicator:
                        keywords.extend(indicator['keyword_patterns'])
                    elif isinstance(indicator, str) and indicator.startswith('keyword_patterns:'):
                        # Handle YAML list format
                        continue
                
                # Look for actual keyword lists in the indicators
                for item in pattern_config.get('indicators', []):
                    if isinstance(item, dict):
                        for key, value in item.items():
                            if key == 'keyword_patterns' and isinstance(value, list):
                                keywords.extend(value)
                
                if keywords:
                    # Create regex pattern that matches any of the keywords (case insensitive)
                    pattern_str = '|'.join(re.escape(kw) for kw in keywords)
                    self.compiled_patterns[pattern_name] = re.compile(f'({pattern_str})', re.IGNORECASE)
    
    def extract_linguistic_features(self, text: str) -> Dict[str, int]:
        """Extract linguistic markers from text."""
        features = {
            'hedging_words': 0,
            'intensifiers': 0,
            'modal_verbs': 0,
            'imperatives': 0,
            'first_person': 0,
            'second_person': 0,
            'third_person': 0
        }
        
        text_lower = text.lower()
        
        # Count linguistic markers
        for marker_type, words in self.linguistic_markers.items():
            for word in words:
                features[marker_type] += len(re.findall(r'\b' + re.escape(word) + r'\b', text_lower))
        
        # Count pronouns
        first_person = ['i', 'me', 'my', 'mine', 'myself']
        second_person = ['you', 'your', 'yours', 'yourself']
        third_person = ['he', 'she', 'it', 'they', 'them', 'their', 'his', 'her', 'its']
        
        for pronoun in first_person:
            features['first_person'] += len(re.findall(r'\b' + pronoun + r'\b', text_lower))
        for pronoun in second_person:
            features['second_person'] += len(re.findall(r'\b' + pronoun + r'\b', text_lower))
        for pronoun in third_person:
            features['third_person'] += len(re.findall(r'\b' + pronoun + r'\b', text_lower))
        
        return features
    
    def detect_patterns(self, utterance: Dict[str, Any]) -> List[PatternMatch]:
        """Detect relational patterns in an utterance."""
        matches = []
        text = utterance['text']
        text_lower = text.lower()
        
        # Extract contextual factors
        contextual_factors = self._extract_contextual_factors(text)
        
        # Check each pattern
        for pattern_name, pattern_config in self.patterns.items():
            confidence = 0.0
            evidence_spans = []
            
            # Pattern-specific detection logic
            if pattern_name == 'memory_dispute':
                confidence, spans = self._detect_memory_dispute(text)
            elif pattern_name == 'boundary_violation':
                confidence, spans = self._detect_boundary_violation(text)
            elif pattern_name == 'accountability_taking':
                confidence, spans = self._detect_accountability_taking(text)
            elif pattern_name == 'accountability_avoiding':
                confidence, spans = self._detect_accountability_avoiding(text)
            elif pattern_name == 'power_imbalance':
                confidence, spans = self._detect_power_imbalance(text, utterance)
            elif pattern_name == 'trauma_responsive':
                confidence, spans = self._detect_trauma_responsive(text)
            elif pattern_name == 'invalidation':
                confidence, spans = self._detect_invalidation(text)
            elif pattern_name == 'consent_negotiation':
                confidence, spans = self._detect_consent_negotiation(text)
            
            # Apply contextual modifiers
            for factor in contextual_factors:
                if factor in self.contextual_modifiers:
                    multiplier = self.contextual_modifiers[factor]['weight_multiplier']
                    confidence *= multiplier
            
            # If confidence exceeds threshold, record the match
            threshold = pattern_config.get('threshold', 0.5)
            if confidence >= threshold:
                for start, end in spans:
                    match = PatternMatch(
                        pattern_type=pattern_name,
                        confidence=min(confidence, 1.0),  # Cap at 1.0
                        evidence_span=(start, end),
                        text_segment=text[start:end],
                        contextual_factors=contextual_factors,
                        theoretical_basis=pattern_config.get('theoretical_basis', '')
                    )
                    matches.append(match)
        
        return matches
    
    def _extract_contextual_factors(self, text: str) -> List[str]:
        """Identify contextual factors that may affect pattern interpretation."""
        factors = []
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['trauma', 'ptsd', 'abuse', 'hospital']):
            factors.append('trauma_history')
        if any(word in text_lower for word in ['asd', 'adhd', 'autistic']):
            factors.append('neurodivergence')
        if any(word in text_lower for word in ['help', 'support', 'vulnerable', 'problems']):
            factors.append('power_differential')
        if any(word in text_lower for word in ['hospital', 'confined', 'crisis']):
            factors.append('vulnerability_context')
            
        return factors
    
    def _detect_memory_dispute(self, text: str) -> Tuple[float, List[Tuple[int, int]]]:
        """Detect memory/reality disputes."""
        confidence = 0.0
        spans = []
        
        dispute_patterns = [
            r"you said\s+['\"]([^'\"]*)['\"]",
            r"don't remember",
            r"didn't say",
            r"deny",
            r"that's not what happened",
            r"twice.*in reference",
            r"exact phrase"
        ]
        
        for pattern in dispute_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                confidence += 0.2
                spans.append((match.start(), match.end()))
        
        return min(confidence, 1.0), spans
    
    def _detect_boundary_violation(self, text: str) -> Tuple[float, List[Tuple[int, int]]]:
        """Detect descriptions of boundary violations."""
        confidence = 0.0
        spans = []
        
        violation_patterns = [
            r"no indicator from me",
            r"would not stop",
            r"did not respond",
            r"uncomfortable",
            r"without.*consent",
            r"touching.*with no",
            r"rubbing.*harder and harder",
            r"did not feel.*until.*uncomfortable"
        ]
        
        for pattern in violation_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                confidence += 0.3  # Higher weight for serious violations
                spans.append((match.start(), match.end()))
        
        return min(confidence, 1.0), spans
    
    def _detect_accountability_taking(self, text: str) -> Tuple[float, List[Tuple[int, int]]]:
        """Detect genuine accountability."""
        confidence = 0.0
        spans = []
        
        accountability_patterns = [
            r"this is on me",
            r"my failure",
            r"I am sorry",
            r"I was wrong",
            r"I made.*mistake",
            r"I failed to",
            r"I should have",
            r"I apologize"
        ]
        
        for pattern in accountability_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                confidence += 0.25
                spans.append((match.start(), match.end()))
        
        return min(confidence, 1.0), spans
    
    def _detect_accountability_avoiding(self, text: str) -> Tuple[float, List[Tuple[int, int]]]:
        """Detect accountability avoidance."""
        confidence = 0.0
        spans = []
        
        avoidance_patterns = [
            r"I don't remember",
            r"didn't mean",
            r"wasn't intentional",
            r"never said",
            r"that's not accurate",
            r"you misunderstood"
        ]
        
        for pattern in avoidance_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                confidence += 0.3
                spans.append((match.start(), match.end()))
        
        return min(confidence, 1.0), spans
    
    def _detect_power_imbalance(self, text: str, utterance: Dict) -> Tuple[float, List[Tuple[int, int]]]:
        """Detect power imbalance indicators."""
        confidence = 0.0
        spans = []
        
        power_patterns = [
            r"I can solve",
            r"I helped",
            r"you needed",
            r"you were vulnerable",
            r"confident.*solve.*problems",
            r"I understood",
            r"you would have thought"
        ]
        
        for pattern in power_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                confidence += 0.2
                spans.append((match.start(), match.end()))
        
        # Check pronoun asymmetry
        features = utterance.get('features', {}).get('pronouns', {})
        first_person = features.get('first_person', 0)
        second_person = features.get('second_person', 0)
        
        if first_person > 0 and second_person > 0:
            ratio = first_person / second_person
            if ratio > 2.0:  # High I/you ratio suggests power assertion
                confidence += 0.3
                # Find first "I" for span
                i_match = re.search(r'\bi\b', text, re.IGNORECASE)
                if i_match:
                    spans.append((i_match.start(), i_match.end()))
        
        return min(confidence, 1.0), spans
    
    def _detect_trauma_responsive(self, text: str) -> Tuple[float, List[Tuple[int, int]]]:
        """Detect trauma-informed language."""
        confidence = 0.0
        spans = []
        
        trauma_patterns = [
            r"trauma",
            r"vulnerable",
            r"survival strategies",
            r"obligation",
            r"power differential",
            r"abuse.*experienced",
            r"blind spots"
        ]
        
        for pattern in trauma_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                confidence += 0.2
                spans.append((match.start(), match.end()))
        
        return min(confidence, 1.0), spans
    
    def _detect_invalidation(self, text: str) -> Tuple[float, List[Tuple[int, int]]]:
        """Detect invalidation of experience."""
        confidence = 0.0
        spans = []
        
        invalidation_patterns = [
            r"you would have thought",
            r"that's not what happened",
            r"you're wrong",
            r"that's not accurate",
            r"you misunderstood"
        ]
        
        for pattern in invalidation_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                confidence += 0.4  # High weight for invalidation
                spans.append((match.start(), match.end()))
        
        return min(confidence, 1.0), spans
    
    def _detect_consent_negotiation(self, text: str) -> Tuple[float, List[Tuple[int, int]]]:
        """Detect consent-related discussion."""
        confidence = 0.0
        spans = []
        
        consent_patterns = [
            r"consent",
            r"what you wanted",
            r"how you felt",
            r"obligation",
            r"autonomy",
            r"respectful",
            r"asking.*felt"
        ]
        
        for pattern in consent_patterns:
            matches = list(re.finditer(pattern, text, re.IGNORECASE))
            for match in matches:
                confidence += 0.15
                spans.append((match.start(), match.end()))
        
        return min(confidence, 1.0), spans

def main():
    """Run pattern analysis on conversation data."""
    
    # Initialize analyzer
    rules_path = "/Users/tiaastor/relational_discourse_project/schemas/pattern_rules.yaml"
    analyzer = RelationalPatternAnalyzer(rules_path)
    
    # Load conversation data
    data_dir = Path("/Users/tiaastor/relational_discourse_project/data/processed")
    with open(data_dir / "complete_conversation.json", 'r') as f:
        utterances = json.load(f)
    
    # Create output directories
    output_dir = Path("/Users/tiaastor/relational_discourse_project/outputs")
    json_dir = output_dir / "json"
    json_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze patterns
    annotations = []
    pattern_summary = {}
    
    for utterance in utterances:
        # Extract linguistic features
        linguistic_features = analyzer.extract_linguistic_features(utterance['text'])
        
        # Update utterance with enhanced features
        utterance['features']['pronouns'] = {
            'first_person': linguistic_features['first_person'],
            'second_person': linguistic_features['second_person'],
            'third_person': linguistic_features['third_person']
        }
        utterance['features']['linguistic_markers'] = {
            'hedging_words': linguistic_features['hedging_words'],
            'intensifiers': linguistic_features['intensifiers'],
            'modal_verbs': linguistic_features['modal_verbs'],
            'imperatives': linguistic_features['imperatives']
        }
        
        # Detect patterns
        matches = analyzer.detect_patterns(utterance)
        
        # Create annotations
        for match in matches:
            annotation_id = f"ANN_{len(annotations)+1:04d}"
            annotation = {
                "annotation_id": annotation_id,
                "utterance_id": utterance['utterance_id'],
                "pattern_type": match.pattern_type,
                "evidence_span": {
                    "start_char": match.evidence_span[0],
                    "end_char": match.evidence_span[1],
                    "text_segment": match.text_segment
                },
                "confidence": round(match.confidence, 3),
                "theoretical_basis": match.theoretical_basis,
                "contextual_factors": match.contextual_factors,
                "severity": "high" if match.confidence > 0.8 else "moderate" if match.confidence > 0.6 else "low",
                "validated_by_human": False
            }
            annotations.append(annotation)
            
            # Update summary
            if match.pattern_type not in pattern_summary:
                pattern_summary[match.pattern_type] = {
                    "count": 0,
                    "avg_confidence": 0,
                    "speakers": {"PERSON_A": 0, "PERSON_B": 0}
                }
            
            pattern_summary[match.pattern_type]["count"] += 1
            pattern_summary[match.pattern_type]["speakers"][utterance["speaker"]] += 1
    
    # Calculate average confidences
    for pattern_type in pattern_summary:
        pattern_annotations = [a for a in annotations if a["pattern_type"] == pattern_type]
        if pattern_annotations:
            avg_confidence = sum(a["confidence"] for a in pattern_annotations) / len(pattern_annotations)
            pattern_summary[pattern_type]["avg_confidence"] = round(avg_confidence, 3)
    
    # Save results
    with open(json_dir / "annotations.json", 'w') as f:
        json.dump(annotations, f, indent=2)
    
    with open(json_dir / "pattern_summary.json", 'w') as f:
        json.dump(pattern_summary, f, indent=2)
    
    with open(data_dir / "enhanced_conversation.json", 'w') as f:
        json.dump(utterances, f, indent=2)
    
    # Print summary
    print(f"✅ Analyzed {len(utterances)} utterances")
    print(f"📋 Generated {len(annotations)} pattern annotations")
    print(f"🔍 Detected {len(pattern_summary)} different pattern types")
    
    print("\\n📊 Pattern Summary:")
    for pattern, stats in pattern_summary.items():
        print(f"  • {pattern}: {stats['count']} instances (avg confidence: {stats['avg_confidence']})")
        print(f"    - Person A: {stats['speakers']['PERSON_A']}, Person B: {stats['speakers']['PERSON_B']}")

if __name__ == "__main__":
    main()