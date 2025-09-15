#!/usr/bin/env python3
"""
Temporal Pattern Analysis for Relational Discourse
Analyzes how communication patterns change over the course of conversations
and across multiple conversation sessions.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta

class TemporalPatternAnalyzer:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        
    def analyze_conversation_flow(self, annotations_file: str, utterances_file: str) -> Dict[str, Any]:
        """Analyze how patterns change throughout a single conversation."""
        
        # Load data
        with open(self.data_dir / annotations_file, 'r') as f:
            annotations = json.load(f)
        
        with open(self.data_dir / utterances_file, 'r') as f:
            utterances = json.load(f)
        
        # Create timeline analysis
        timeline_data = []
        
        for annotation in annotations:
            # Find corresponding utterance
            utterance = next((u for u in utterances if u['utterance_id'] == annotation['utterance_id']), None)
            if utterance:
                timeline_data.append({
                    'turn_number': utterance['turn_number'],
                    'speaker': utterance['speaker'], 
                    'pattern_type': annotation['pattern_type'],
                    'confidence': annotation['confidence'],
                    'severity': annotation['severity'],
                    'word_count': utterance['word_count']
                })
        
        df = pd.DataFrame(timeline_data)
        
        # Calculate pattern evolution metrics
        analysis = {
            'pattern_timeline': self._create_pattern_timeline(df),
            'escalation_points': self._detect_escalation_points(df),
            'pattern_clustering': self._analyze_pattern_clustering(df),
            'speaker_dynamics': self._analyze_speaker_dynamics(df),
            'conversation_phases': self._identify_conversation_phases(df)
        }
        
        return analysis
    
    def _create_pattern_timeline(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create turn-by-turn pattern analysis."""
        timeline = []
        
        for turn in sorted(df['turn_number'].unique()):
            turn_patterns = df[df['turn_number'] == turn]
            
            timeline.append({
                'turn': turn,
                'speaker': turn_patterns.iloc[0]['speaker'],
                'pattern_count': len(turn_patterns),
                'pattern_types': turn_patterns['pattern_type'].tolist(),
                'avg_confidence': turn_patterns['confidence'].mean(),
                'max_severity': turn_patterns['severity'].mode().iloc[0] if not turn_patterns.empty else 'none',
                'word_count': turn_patterns.iloc[0]['word_count']
            })
        
        return {
            'timeline': timeline,
            'total_turns': len(timeline),
            'pattern_density': len(df) / len(timeline) if timeline else 0
        }
    
    def _detect_escalation_points(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify moments where pattern severity or frequency increases."""
        escalations = []
        
        # Group by turn and calculate severity scores
        severity_map = {'low': 1, 'moderate': 2, 'high': 3, 'critical': 4}
        df['severity_score'] = df['severity'].map(severity_map)
        
        turn_severity = df.groupby('turn_number').agg({
            'severity_score': 'max',
            'confidence': 'mean',
            'pattern_type': 'count'
        }).rename(columns={'pattern_type': 'pattern_count'})
        
        # Detect escalation points (significant increases)
        for i in range(1, len(turn_severity)):
            current_turn = turn_severity.index[i]
            prev_turn = turn_severity.index[i-1]
            
            current_severity = turn_severity.loc[current_turn, 'severity_score']
            prev_severity = turn_severity.loc[prev_turn, 'severity_score']
            
            current_count = turn_severity.loc[current_turn, 'pattern_count']
            prev_count = turn_severity.loc[prev_turn, 'pattern_count']
            
            # Escalation criteria
            severity_increase = current_severity > prev_severity
            pattern_spike = current_count > prev_count * 1.5
            
            if severity_increase or pattern_spike:
                escalations.append({
                    'turn': current_turn,
                    'escalation_type': 'severity' if severity_increase else 'frequency',
                    'from_severity': prev_severity,
                    'to_severity': current_severity,
                    'from_count': prev_count,
                    'to_count': current_count
                })
        
        return escalations
    
    def _analyze_pattern_clustering(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze if certain patterns cluster together in time."""
        pattern_cooccurrence = {}
        
        # Group patterns by turn
        turn_patterns = df.groupby('turn_number')['pattern_type'].apply(list).to_dict()
        
        # Calculate co-occurrence matrix
        pattern_types = df['pattern_type'].unique()
        cooccurrence_matrix = np.zeros((len(pattern_types), len(pattern_types)))
        pattern_to_idx = {p: i for i, p in enumerate(pattern_types)}
        
        for turn, patterns in turn_patterns.items():
            for i, p1 in enumerate(patterns):
                for j, p2 in enumerate(patterns):
                    if i != j:  # Don't count self-occurrence
                        idx1, idx2 = pattern_to_idx[p1], pattern_to_idx[p2]
                        cooccurrence_matrix[idx1, idx2] += 1
        
        # Find significant clusters (patterns that frequently appear together)
        clusters = []
        threshold = np.mean(cooccurrence_matrix) + np.std(cooccurrence_matrix)
        
        for i, p1 in enumerate(pattern_types):
            for j, p2 in enumerate(pattern_types):
                if i != j and cooccurrence_matrix[i, j] > threshold:
                    clusters.append({
                        'pattern_1': p1,
                        'pattern_2': p2,
                        'cooccurrence_count': int(cooccurrence_matrix[i, j]),
                        'cluster_strength': float(cooccurrence_matrix[i, j] / np.max(cooccurrence_matrix))
                    })
        
        return {
            'cooccurrence_matrix': cooccurrence_matrix.tolist(),
            'pattern_types': list(pattern_types),
            'significant_clusters': clusters
        }
    
    def _analyze_speaker_dynamics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how pattern usage differs between speakers over time."""
        speaker_evolution = {}
        
        for speaker in df['speaker'].unique():
            speaker_df = df[df['speaker'] == speaker].copy()
            speaker_df = speaker_df.sort_values('turn_number')
            
            # Calculate running averages
            speaker_df['cumulative_confidence'] = speaker_df['confidence'].expanding().mean()
            speaker_df['cumulative_pattern_count'] = speaker_df.groupby('speaker').cumcount() + 1
            
            # Pattern type evolution
            pattern_evolution = []
            window_size = 3
            
            for i in range(len(speaker_df) - window_size + 1):
                window = speaker_df.iloc[i:i+window_size]
                pattern_evolution.append({
                    'turn_range': f"{window.iloc[0]['turn_number']}-{window.iloc[-1]['turn_number']}",
                    'dominant_pattern': window['pattern_type'].mode().iloc[0],
                    'avg_confidence': window['confidence'].mean(),
                    'pattern_diversity': len(window['pattern_type'].unique())
                })
            
            speaker_evolution[speaker] = {
                'total_patterns': len(speaker_df),
                'pattern_types': speaker_df['pattern_type'].value_counts().to_dict(),
                'avg_confidence': speaker_df['confidence'].mean(),
                'confidence_trend': self._calculate_trend(speaker_df['cumulative_confidence'].tolist()),
                'pattern_evolution': pattern_evolution
            }
        
        return speaker_evolution
    
    def _identify_conversation_phases(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Identify distinct phases in conversation based on pattern changes."""
        phases = []
        
        # Sort by turn number
        df_sorted = df.sort_values('turn_number')
        
        # Use sliding window to detect phase changes
        window_size = 4
        phase_start = df_sorted.iloc[0]['turn_number']
        current_dominant_pattern = None
        
        for i in range(0, len(df_sorted), window_size):
            window = df_sorted.iloc[i:i+window_size]
            
            if len(window) < 2:
                continue
                
            # Identify dominant pattern in window
            dominant_pattern = window['pattern_type'].mode().iloc[0]
            avg_confidence = window['confidence'].mean()
            speaker_balance = window['speaker'].value_counts()
            
            # Detect phase change
            if current_dominant_pattern and dominant_pattern != current_dominant_pattern:
                phases.append({
                    'phase_number': len(phases) + 1,
                    'turn_start': phase_start,
                    'turn_end': window.iloc[0]['turn_number'] - 1,
                    'dominant_pattern': current_dominant_pattern,
                    'avg_confidence': avg_confidence,
                    'primary_speaker': speaker_balance.index[0] if not speaker_balance.empty else 'unknown'
                })
                phase_start = window.iloc[0]['turn_number']
            
            current_dominant_pattern = dominant_pattern
        
        # Add final phase
        if df_sorted.iloc[-1]['turn_number'] > phase_start:
            final_window = df_sorted[df_sorted['turn_number'] >= phase_start]
            phases.append({
                'phase_number': len(phases) + 1,
                'turn_start': phase_start,
                'turn_end': df_sorted.iloc[-1]['turn_number'],
                'dominant_pattern': current_dominant_pattern,
                'avg_confidence': final_window['confidence'].mean(),
                'primary_speaker': final_window['speaker'].mode().iloc[0]
            })
        
        return phases
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate if trend is increasing, decreasing, or stable."""
        if len(values) < 2:
            return 'insufficient_data'
            
        slope, _, r_value, p_value, _ = stats.linregress(range(len(values)), values)
        
        if p_value > 0.05:  # Not significant
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def generate_temporal_report(self, output_file: str, analysis: Dict[str, Any]) -> None:
        """Generate comprehensive temporal analysis report."""
        
        report = f"""# Temporal Pattern Analysis Report

## Conversation Flow Analysis

### Pattern Timeline
- **Total Turns Analyzed:** {analysis['pattern_timeline']['total_turns']}
- **Pattern Density:** {analysis['pattern_timeline']['pattern_density']:.2f} patterns per turn

### Escalation Points
Detected {len(analysis['escalation_points'])} escalation points:
"""
        
        for escalation in analysis['escalation_points']:
            report += f"- **Turn {escalation['turn']}:** {escalation['escalation_type']} escalation "
            report += f"(severity: {escalation['from_severity']} → {escalation['to_severity']})\n"
        
        report += f"""
### Conversation Phases
Identified {len(analysis['conversation_phases'])} distinct phases:
"""
        
        for phase in analysis['conversation_phases']:
            report += f"- **Phase {phase['phase_number']}** (Turns {phase['turn_start']}-{phase['turn_end']}): "
            report += f"Dominant pattern: {phase['dominant_pattern']}, "
            report += f"Primary speaker: {phase['primary_speaker']}\n"
        
        report += """
### Pattern Clustering
Significant pattern co-occurrences:
"""
        
        for cluster in analysis['pattern_clustering']['significant_clusters']:
            report += f"- **{cluster['pattern_1']}** + **{cluster['pattern_2']}**: "
            report += f"{cluster['cooccurrence_count']} co-occurrences "
            report += f"(strength: {cluster['cluster_strength']:.2f})\n"
        
        report += """
### Speaker Dynamics
"""
        
        for speaker, dynamics in analysis['speaker_dynamics'].items():
            report += f"""
#### {speaker}
- Total patterns: {dynamics['total_patterns']}
- Average confidence: {dynamics['avg_confidence']:.2f}
- Confidence trend: {dynamics['confidence_trend']}
- Pattern distribution: {dynamics['pattern_types']}
"""
        
        # Save report
        with open(output_file, 'w') as f:
            f.write(report)

def main():
    """Run temporal pattern analysis."""
    analyzer = TemporalPatternAnalyzer("/Users/tiaastor/relational_discourse_project/outputs/json")
    
    # Analyze conversation flow
    analysis = analyzer.analyze_conversation_flow(
        "annotations.json", 
        "../data/processed/complete_conversation.json"
    )
    
    # Generate report
    output_dir = Path("/Users/tiaastor/relational_discourse_project/outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analyzer.generate_temporal_report(
        str(output_dir / "temporal_analysis_report.md"), 
        analysis
    )
    
    # Save analysis data
    with open(output_dir.parent / "json" / "temporal_analysis.json", 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print("✅ Temporal analysis complete")
    print(f"📊 Detected {len(analysis['escalation_points'])} escalation points")
    print(f"🔍 Identified {len(analysis['conversation_phases'])} conversation phases")
    print(f"🔗 Found {len(analysis['pattern_clustering']['significant_clusters'])} pattern clusters")

if __name__ == "__main__":
    main()