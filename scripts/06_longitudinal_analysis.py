#!/usr/bin/env python3
"""
Longitudinal Analysis for Relational Discourse
Tracks pattern changes across multiple conversations over time
to identify relationship repair progress or deterioration.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass 
class ConversationSession:
    session_id: str
    date: datetime
    participants: List[str] 
    utterances_file: str
    annotations_file: str
    context: Dict[str, Any]

class LongitudinalAnalyzer:
    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.sessions: List[ConversationSession] = []
        
    def add_conversation_session(self, session: ConversationSession) -> None:
        """Add a conversation session to the longitudinal analysis."""
        self.sessions.append(session)
        self.sessions.sort(key=lambda s: s.date)  # Keep chronological order
    
    def analyze_pattern_evolution(self) -> Dict[str, Any]:
        """Analyze how patterns change across multiple conversations."""
        
        if len(self.sessions) < 2:
            raise ValueError("Need at least 2 sessions for longitudinal analysis")
        
        # Load all session data
        session_data = []
        for session in self.sessions:
            session_patterns = self._load_session_patterns(session)
            session_data.append({
                'session': session,
                'patterns': session_patterns
            })
        
        analysis = {
            'pattern_trajectories': self._analyze_pattern_trajectories(session_data),
            'relationship_phases': self._identify_relationship_phases(session_data), 
            'repair_indicators': self._assess_repair_indicators(session_data),
            'deterioration_signals': self._assess_deterioration_signals(session_data),
            'speaker_evolution': self._analyze_speaker_evolution(session_data),
            'critical_incidents': self._identify_critical_incidents(session_data),
            'progress_metrics': self._calculate_progress_metrics(session_data)
        }
        
        return analysis
    
    def _load_session_patterns(self, session: ConversationSession) -> Dict[str, Any]:
        """Load pattern data for a single session."""
        
        # Load utterances
        utterances_path = self.project_dir / session.utterances_file
        with open(utterances_path, 'r') as f:
            utterances = json.load(f)
        
        # Load annotations  
        annotations_path = self.project_dir / session.annotations_file
        with open(annotations_path, 'r') as f:
            annotations = json.load(f)
        
        # Calculate session metrics
        pattern_counts = {}
        total_confidence = 0
        severity_scores = {'low': 1, 'moderate': 2, 'high': 3, 'critical': 4}
        total_severity = 0
        speaker_patterns = {'PERSON_A': {}, 'PERSON_B': {}}
        
        for annotation in annotations:
            pattern_type = annotation['pattern_type']
            confidence = annotation['confidence']
            severity = annotation['severity']
            
            # Get speaker for this annotation
            utterance = next((u for u in utterances if u['utterance_id'] == annotation['utterance_id']), None)
            if utterance:
                speaker = utterance['speaker']
                
                # Track speaker-specific patterns
                if pattern_type not in speaker_patterns[speaker]:
                    speaker_patterns[speaker][pattern_type] = 0
                speaker_patterns[speaker][pattern_type] += 1
            
            # Overall pattern tracking
            if pattern_type not in pattern_counts:
                pattern_counts[pattern_type] = 0
            pattern_counts[pattern_type] += 1
            
            total_confidence += confidence
            total_severity += severity_scores.get(severity, 0)
        
        return {
            'session_id': session.session_id,
            'date': session.date,
            'total_patterns': len(annotations),
            'pattern_counts': pattern_counts,
            'avg_confidence': total_confidence / len(annotations) if annotations else 0,
            'avg_severity': total_severity / len(annotations) if annotations else 0,
            'speaker_patterns': speaker_patterns,
            'total_utterances': len(utterances),
            'conversation_length': sum(u.get('word_count', 0) for u in utterances),
            'turn_ratio': self._calculate_turn_ratio(utterances)
        }
    
    def _calculate_turn_ratio(self, utterances: List[Dict]) -> float:
        """Calculate speaking turn ratio between speakers."""
        speaker_counts = {}
        for utterance in utterances:
            speaker = utterance['speaker']
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
        if 'PERSON_A' in speaker_counts and 'PERSON_B' in speaker_counts:
            return speaker_counts['PERSON_B'] / speaker_counts['PERSON_A']
        return 1.0
    
    def _analyze_pattern_trajectories(self, session_data: List[Dict]) -> Dict[str, Any]:
        """Analyze how each pattern type changes over time."""
        
        # Get all unique pattern types across sessions
        all_patterns = set()
        for data in session_data:
            all_patterns.update(data['patterns']['pattern_counts'].keys())
        
        trajectories = {}
        
        for pattern_type in all_patterns:
            trajectory = []
            for data in session_data:
                count = data['patterns']['pattern_counts'].get(pattern_type, 0)
                total = data['patterns']['total_patterns']
                proportion = count / total if total > 0 else 0
                
                trajectory.append({
                    'session_id': data['patterns']['session_id'],
                    'date': data['patterns']['date'].isoformat(),
                    'count': count,
                    'proportion': proportion,
                    'total_patterns': total
                })
            
            # Calculate trajectory statistics
            counts = [t['count'] for t in trajectory]
            proportions = [t['proportion'] for t in trajectory]
            
            # Linear trend analysis
            if len(counts) >= 2:
                time_indices = list(range(len(counts)))
                count_slope, _, count_r, count_p, _ = stats.linregress(time_indices, counts)
                prop_slope, _, prop_r, prop_p, _ = stats.linregress(time_indices, proportions)
                
                trajectories[pattern_type] = {
                    'timeline': trajectory,
                    'trend': {
                        'count_slope': count_slope,
                        'count_r_squared': count_r ** 2,
                        'count_p_value': count_p,
                        'proportion_slope': prop_slope, 
                        'proportion_r_squared': prop_r ** 2,
                        'proportion_p_value': prop_p,
                        'direction': 'increasing' if prop_slope > 0 else 'decreasing' if prop_slope < 0 else 'stable',
                        'significance': prop_p < 0.05
                    },
                    'statistics': {
                        'mean_count': np.mean(counts),
                        'std_count': np.std(counts),
                        'mean_proportion': np.mean(proportions),
                        'std_proportion': np.std(proportions),
                        'min_count': min(counts),
                        'max_count': max(counts)
                    }
                }
        
        return trajectories
    
    def _identify_relationship_phases(self, session_data: List[Dict]) -> List[Dict[str, Any]]:
        """Identify distinct phases in the relationship based on pattern changes."""
        
        phases = []
        
        if len(session_data) < 3:
            return phases  # Need minimum sessions for phase detection
        
        # Create composite metrics for each session
        session_metrics = []
        for data in session_data:
            patterns = data['patterns']
            
            # Safety concern score (boundary violations + power imbalances)
            safety_patterns = ['boundary_violation', 'power_imbalance', 'invalidation']
            safety_score = sum(patterns['pattern_counts'].get(p, 0) for p in safety_patterns)
            
            # Repair attempt score (accountability + trauma responsive)
            repair_patterns = ['accountability_taking', 'trauma_responsive', 'consent_negotiation']
            repair_score = sum(patterns['pattern_counts'].get(p, 0) for p in repair_patterns)
            
            # Communication quality score 
            quality_score = patterns['avg_confidence'] * (1 / patterns['avg_severity'] if patterns['avg_severity'] > 0 else 1)
            
            session_metrics.append({
                'session_id': patterns['session_id'],
                'date': patterns['date'],
                'safety_score': safety_score,
                'repair_score': repair_score,
                'quality_score': quality_score,
                'total_patterns': patterns['total_patterns']
            })
        
        # Detect phase changes using clustering on these metrics
        # Simple approach: look for significant changes in the composite scores
        current_phase_start = 0
        
        for i in range(1, len(session_metrics)):
            current = session_metrics[i]
            previous = session_metrics[i-1]
            
            # Check for significant changes (>50% change in any metric)
            safety_change = abs(current['safety_score'] - previous['safety_score']) / max(previous['safety_score'], 1)
            repair_change = abs(current['repair_score'] - previous['repair_score']) / max(previous['repair_score'], 1)
            quality_change = abs(current['quality_score'] - previous['quality_score']) / max(previous['quality_score'], 0.1)
            
            if safety_change > 0.5 or repair_change > 0.5 or quality_change > 0.5:
                # Phase boundary detected
                phase_sessions = session_metrics[current_phase_start:i]
                
                if phase_sessions:
                    phases.append({
                        'phase_number': len(phases) + 1,
                        'start_session': phase_sessions[0]['session_id'],
                        'end_session': phase_sessions[-1]['session_id'],
                        'start_date': phase_sessions[0]['date'].isoformat(),
                        'end_date': phase_sessions[-1]['date'].isoformat(),
                        'session_count': len(phase_sessions),
                        'characteristics': {
                            'avg_safety_score': np.mean([s['safety_score'] for s in phase_sessions]),
                            'avg_repair_score': np.mean([s['repair_score'] for s in phase_sessions]),
                            'avg_quality_score': np.mean([s['quality_score'] for s in phase_sessions]),
                            'dominant_pattern': self._get_dominant_pattern_for_phase(session_data[current_phase_start:i])
                        }
                    })
                
                current_phase_start = i
        
        # Add final phase
        if current_phase_start < len(session_metrics):
            phase_sessions = session_metrics[current_phase_start:]
            phases.append({
                'phase_number': len(phases) + 1,
                'start_session': phase_sessions[0]['session_id'],
                'end_session': phase_sessions[-1]['session_id'], 
                'start_date': phase_sessions[0]['date'].isoformat(),
                'end_date': phase_sessions[-1]['date'].isoformat(),
                'session_count': len(phase_sessions),
                'characteristics': {
                    'avg_safety_score': np.mean([s['safety_score'] for s in phase_sessions]),
                    'avg_repair_score': np.mean([s['repair_score'] for s in phase_sessions]),
                    'avg_quality_score': np.mean([s['quality_score'] for s in phase_sessions]),
                    'dominant_pattern': self._get_dominant_pattern_for_phase(session_data[current_phase_start:])
                }
            })
        
        return phases
    
    def _get_dominant_pattern_for_phase(self, phase_data: List[Dict]) -> str:
        """Get the most common pattern type in a relationship phase."""
        pattern_totals = {}
        
        for data in phase_data:
            for pattern, count in data['patterns']['pattern_counts'].items():
                pattern_totals[pattern] = pattern_totals.get(pattern, 0) + count
        
        if pattern_totals:
            return max(pattern_totals, key=pattern_totals.get)
        return 'none'
    
    def _assess_repair_indicators(self, session_data: List[Dict]) -> Dict[str, Any]:
        """Assess positive indicators of relationship repair progress."""
        
        repair_indicators = {
            'accountability_trend': self._calculate_pattern_trend(session_data, 'accountability_taking'),
            'trauma_responsiveness_trend': self._calculate_pattern_trend(session_data, 'trauma_responsive'),
            'consent_awareness_trend': self._calculate_pattern_trend(session_data, 'consent_negotiation'),
            'boundary_violation_reduction': -self._calculate_pattern_trend(session_data, 'boundary_violation'),
            'overall_repair_score': 0
        }
        
        # Calculate composite repair score
        positive_trends = [
            repair_indicators['accountability_trend'],
            repair_indicators['trauma_responsiveness_trend'], 
            repair_indicators['consent_awareness_trend'],
            repair_indicators['boundary_violation_reduction']
        ]
        
        repair_indicators['overall_repair_score'] = np.mean([t for t in positive_trends if t is not None])
        
        # Assess repair quality
        if repair_indicators['overall_repair_score'] > 0.5:
            repair_indicators['repair_assessment'] = 'strong_progress'
        elif repair_indicators['overall_repair_score'] > 0:
            repair_indicators['repair_assessment'] = 'moderate_progress'
        elif repair_indicators['overall_repair_score'] > -0.5:
            repair_indicators['repair_assessment'] = 'minimal_progress'
        else:
            repair_indicators['repair_assessment'] = 'deteriorating'
        
        return repair_indicators
    
    def _assess_deterioration_signals(self, session_data: List[Dict]) -> Dict[str, Any]:
        """Assess negative indicators suggesting relationship deterioration."""
        
        deterioration_signals = {
            'power_imbalance_trend': self._calculate_pattern_trend(session_data, 'power_imbalance'),
            'invalidation_trend': self._calculate_pattern_trend(session_data, 'invalidation'),
            'memory_dispute_trend': self._calculate_pattern_trend(session_data, 'memory_dispute'),
            'accountability_avoidance_trend': self._calculate_pattern_trend(session_data, 'accountability_avoiding'),
            'overall_severity_trend': self._calculate_severity_trend(session_data),
            'communication_breakdown_risk': 'low'
        }
        
        # Assess breakdown risk
        concerning_trends = [
            deterioration_signals['power_imbalance_trend'] or 0,
            deterioration_signals['invalidation_trend'] or 0,
            deterioration_signals['memory_dispute_trend'] or 0,
            deterioration_signals['accountability_avoidance_trend'] or 0
        ]
        
        risk_score = np.mean(concerning_trends)
        
        if risk_score > 0.5:
            deterioration_signals['communication_breakdown_risk'] = 'high'
        elif risk_score > 0.2:
            deterioration_signals['communication_breakdown_risk'] = 'moderate'
        else:
            deterioration_signals['communication_breakdown_risk'] = 'low'
        
        return deterioration_signals
    
    def _calculate_pattern_trend(self, session_data: List[Dict], pattern_type: str) -> Optional[float]:
        """Calculate trend for a specific pattern type across sessions."""
        
        values = []
        for data in session_data:
            count = data['patterns']['pattern_counts'].get(pattern_type, 0)
            total = data['patterns']['total_patterns']
            proportion = count / total if total > 0 else 0
            values.append(proportion)
        
        if len(values) < 2:
            return None
            
        # Calculate linear trend
        time_indices = list(range(len(values)))
        slope, _, _, p_value, _ = stats.linregress(time_indices, values)
        
        return slope if p_value < 0.05 else 0  # Return 0 if not significant
    
    def _calculate_severity_trend(self, session_data: List[Dict]) -> float:
        """Calculate trend in average pattern severity across sessions."""
        
        severity_values = [data['patterns']['avg_severity'] for data in session_data]
        
        if len(severity_values) < 2:
            return 0
            
        time_indices = list(range(len(severity_values)))
        slope, _, _, p_value, _ = stats.linregress(time_indices, severity_values)
        
        return slope if p_value < 0.05 else 0
    
    def _analyze_speaker_evolution(self, session_data: List[Dict]) -> Dict[str, Any]:
        """Analyze how each speaker's communication patterns evolve."""
        
        speaker_evolution = {'PERSON_A': {}, 'PERSON_B': {}}
        
        for speaker in ['PERSON_A', 'PERSON_B']:
            # Track pattern evolution for this speaker
            speaker_patterns_over_time = []
            
            for data in session_data:
                speaker_data = data['patterns']['speaker_patterns'].get(speaker, {})
                total_speaker_patterns = sum(speaker_data.values())
                
                speaker_patterns_over_time.append({
                    'session_id': data['patterns']['session_id'],
                    'date': data['patterns']['date'],
                    'pattern_counts': speaker_data,
                    'total_patterns': total_speaker_patterns
                })
            
            # Calculate trends for each pattern type for this speaker
            pattern_trends = {}
            all_speaker_patterns = set()
            for session in speaker_patterns_over_time:
                all_speaker_patterns.update(session['pattern_counts'].keys())
            
            for pattern in all_speaker_patterns:
                values = []
                for session in speaker_patterns_over_time:
                    count = session['pattern_counts'].get(pattern, 0)
                    total = session['total_patterns']
                    proportion = count / total if total > 0 else 0
                    values.append(proportion)
                
                if len(values) >= 2:
                    trend_slope = self._calculate_trend_slope(values)
                    pattern_trends[pattern] = trend_slope
            
            speaker_evolution[speaker] = {
                'timeline': speaker_patterns_over_time,
                'pattern_trends': pattern_trends,
                'overall_pattern_activity': [s['total_patterns'] for s in speaker_patterns_over_time]
            }
        
        return speaker_evolution
    
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate slope of linear trend."""
        if len(values) < 2:
            return 0
            
        time_indices = list(range(len(values)))
        slope, _, _, _, _ = stats.linregress(time_indices, values)
        return slope
    
    def _identify_critical_incidents(self, session_data: List[Dict]) -> List[Dict[str, Any]]:
        """Identify conversation sessions with unusual pattern activity."""
        
        critical_incidents = []
        
        # Calculate baseline metrics
        total_patterns = [data['patterns']['total_patterns'] for data in session_data]
        avg_severity = [data['patterns']['avg_severity'] for data in session_data]
        
        mean_patterns = np.mean(total_patterns)
        std_patterns = np.std(total_patterns)
        mean_severity = np.mean(avg_severity)
        std_severity = np.std(avg_severity)
        
        # Identify outliers
        for data in session_data:
            patterns = data['patterns']
            
            z_score_patterns = (patterns['total_patterns'] - mean_patterns) / std_patterns if std_patterns > 0 else 0
            z_score_severity = (patterns['avg_severity'] - mean_severity) / std_severity if std_severity > 0 else 0
            
            # Flag as critical incident if >2 standard deviations from mean
            if abs(z_score_patterns) > 2 or abs(z_score_severity) > 2:
                incident_type = []
                if z_score_patterns > 2:
                    incident_type.append('high_pattern_activity')
                elif z_score_patterns < -2:
                    incident_type.append('low_pattern_activity')
                    
                if z_score_severity > 2:
                    incident_type.append('high_severity')
                elif z_score_severity < -2:
                    incident_type.append('low_severity')
                
                critical_incidents.append({
                    'session_id': patterns['session_id'],
                    'date': patterns['date'].isoformat(),
                    'incident_types': incident_type,
                    'pattern_count_z_score': z_score_patterns,
                    'severity_z_score': z_score_severity,
                    'dominant_patterns': sorted(patterns['pattern_counts'].items(), key=lambda x: x[1], reverse=True)[:3]
                })
        
        return critical_incidents
    
    def _calculate_progress_metrics(self, session_data: List[Dict]) -> Dict[str, Any]:
        """Calculate overall progress metrics for the relationship."""
        
        if len(session_data) < 2:
            return {'insufficient_data': True}
        
        # Compare first and last sessions
        first_session = session_data[0]['patterns']
        last_session = session_data[-1]['patterns']
        
        # Safety metrics (should decrease)
        safety_patterns = ['boundary_violation', 'power_imbalance', 'invalidation']
        first_safety = sum(first_session['pattern_counts'].get(p, 0) for p in safety_patterns)
        last_safety = sum(last_session['pattern_counts'].get(p, 0) for p in safety_patterns)
        
        # Repair metrics (should increase)
        repair_patterns = ['accountability_taking', 'trauma_responsive', 'consent_negotiation']
        first_repair = sum(first_session['pattern_counts'].get(p, 0) for p in repair_patterns)
        last_repair = sum(last_session['pattern_counts'].get(p, 0) for p in repair_patterns)
        
        # Calculate progress scores
        safety_improvement = (first_safety - last_safety) / max(first_safety, 1)  # Positive is good
        repair_improvement = (last_repair - first_repair) / max(first_repair, 1)  # Positive is good
        
        overall_progress = (safety_improvement + repair_improvement) / 2
        
        return {
            'timespan_days': (last_session['date'] - first_session['date']).days,
            'total_sessions': len(session_data),
            'safety_improvement': safety_improvement,
            'repair_improvement': repair_improvement,
            'overall_progress_score': overall_progress,
            'severity_change': last_session['avg_severity'] - first_session['avg_severity'],
            'communication_efficiency_change': (last_session['total_patterns'] / last_session['conversation_length']) - 
                                             (first_session['total_patterns'] / first_session['conversation_length'])
        }

def main():
    """Example usage of longitudinal analysis."""
    
    analyzer = LongitudinalAnalyzer("/Users/tiaastor/relational_discourse_project")
    
    # Add example session (you would add multiple real sessions)
    session1 = ConversationSession(
        session_id="session_001",
        date=datetime.now() - timedelta(days=30),
        participants=["PERSON_A", "PERSON_B"],
        utterances_file="data/processed/complete_conversation.json",
        annotations_file="outputs/json/annotations.json", 
        context={"setting": "initial_discussion", "topic": "boundary_violations"}
    )
    
    analyzer.add_conversation_session(session1)
    
    # Would add more sessions here for real analysis
    # analyzer.add_conversation_session(session2)
    # analyzer.add_conversation_session(session3)
    
    print("✅ Longitudinal analyzer ready")
    print("📊 Add multiple conversation sessions to analyze relationship evolution")
    print("🔍 Tracks repair progress, deterioration signals, and critical incidents")

if __name__ == "__main__":
    main()