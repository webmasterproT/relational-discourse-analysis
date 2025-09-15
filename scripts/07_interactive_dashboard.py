#!/usr/bin/env python3
"""
Interactive Analysis Dashboard for Relational Discourse
Web-based interface for exploring patterns, viewing evidence, and adjusting parameters.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, List, Any
from datetime import datetime

class InteractiveDashboard:
    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.data_loaded = False
        
        # Set page config
        if not hasattr(st, '_is_running_with_streamlit'):
            st.set_page_config(
                page_title="Relational Discourse Analysis Dashboard",
                page_icon="🔍",
                layout="wide",
                initial_sidebar_state="expanded"
            )
    
    def load_data(self) -> Dict[str, Any]:
        """Load all analysis data files."""
        try:
            # Load main data files
            with open(self.project_dir / "outputs/json/annotations.json", 'r') as f:
                annotations = json.load(f)
            
            with open(self.project_dir / "data/processed/complete_conversation.json", 'r') as f:
                utterances = json.load(f)
            
            with open(self.project_dir / "outputs/json/pattern_summary.json", 'r') as f:
                pattern_summary = json.load(f)
            
            # Load temporal analysis if available
            temporal_analysis = {}
            temporal_path = self.project_dir / "outputs/json/temporal_analysis.json"
            if temporal_path.exists():
                with open(temporal_path, 'r') as f:
                    temporal_analysis = json.load(f)
            
            self.data_loaded = True
            
            return {
                'annotations': annotations,
                'utterances': utterances,
                'pattern_summary': pattern_summary,
                'temporal_analysis': temporal_analysis
            }
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return {}
    
    def create_pattern_overview_chart(self, pattern_summary: Dict[str, Any]) -> go.Figure:
        """Create pattern distribution overview chart."""
        
        patterns = list(pattern_summary.keys())
        counts = [pattern_summary[p]['count'] for p in patterns]
        confidences = [pattern_summary[p]['avg_confidence'] for p in patterns]
        
        # Create subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Bar chart for counts
        fig.add_trace(
            go.Bar(
                x=patterns,
                y=counts,
                name="Pattern Count",
                marker_color='lightblue'
            ),
            secondary_y=False,
        )
        
        # Line chart for confidence
        fig.add_trace(
            go.Scatter(
                x=patterns,
                y=confidences,
                mode='lines+markers',
                name="Avg Confidence",
                marker_color='red',
                line=dict(width=3)
            ),
            secondary_y=True,
        )
        
        fig.update_xaxes(title_text="Pattern Type", tickangle=45)
        fig.update_yaxes(title_text="Pattern Count", secondary_y=False)
        fig.update_yaxes(title_text="Average Confidence", secondary_y=True)
        
        fig.update_layout(
            title="Pattern Distribution and Confidence Levels",
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    def create_speaker_comparison_chart(self, pattern_summary: Dict[str, Any]) -> go.Figure:
        """Create speaker comparison chart."""
        
        patterns = list(pattern_summary.keys())
        person_a_counts = [pattern_summary[p]['speakers']['PERSON_A'] for p in patterns]
        person_b_counts = [pattern_summary[p]['speakers']['PERSON_B'] for p in patterns]
        
        fig = go.Figure(data=[
            go.Bar(name='Person A', x=patterns, y=person_a_counts, marker_color='lightcoral'),
            go.Bar(name='Person B', x=patterns, y=person_b_counts, marker_color='skyblue')
        ])
        
        fig.update_layout(
            title="Pattern Distribution by Speaker",
            xaxis_title="Pattern Type",
            yaxis_title="Count",
            barmode='group',
            xaxis_tickangle=-45,
            height=500
        )
        
        return fig
    
    def create_timeline_visualization(self, annotations: List[Dict], utterances: List[Dict]) -> go.Figure:
        """Create timeline visualization of patterns."""
        
        # Create timeline data
        timeline_data = []
        for annotation in annotations:
            # Find corresponding utterance
            utterance = next((u for u in utterances if u['utterance_id'] == annotation['utterance_id']), None)
            if utterance:
                timeline_data.append({
                    'turn': utterance['turn_number'],
                    'speaker': utterance['speaker'],
                    'pattern': annotation['pattern_type'],
                    'confidence': annotation['confidence'],
                    'severity': annotation['severity'],
                    'text_preview': utterance['text'][:100] + "..." if len(utterance['text']) > 100 else utterance['text']
                })
        
        df = pd.DataFrame(timeline_data)
        
        # Color mapping for patterns
        pattern_colors = {
            'boundary_violation': '#ff6b6b',
            'power_imbalance': '#feca57',
            'accountability_taking': '#48dbfb',
            'trauma_responsive': '#0abde3',
            'consent_negotiation': '#006ba6'
        }
        
        # Create scatter plot
        fig = px.scatter(
            df,
            x='turn',
            y='pattern',
            color='pattern',
            size='confidence',
            symbol='speaker',
            hover_data=['confidence', 'severity', 'text_preview'],
            color_discrete_map=pattern_colors,
            title="Pattern Timeline Across Conversation"
        )
        
        fig.update_layout(
            xaxis_title="Turn Number",
            yaxis_title="Pattern Type", 
            height=600,
            showlegend=True
        )
        
        return fig
    
    def create_evidence_viewer(self, annotations: List[Dict], utterances: List[Dict]) -> None:
        """Create interactive evidence viewer."""
        
        st.subheader("🔍 Evidence Viewer")
        
        # Pattern type filter
        pattern_types = list(set(ann['pattern_type'] for ann in annotations))
        selected_pattern = st.selectbox("Select Pattern Type", pattern_types)
        
        # Filter annotations by selected pattern
        filtered_annotations = [ann for ann in annotations if ann['pattern_type'] == selected_pattern]
        
        # Display filtered annotations
        for i, annotation in enumerate(filtered_annotations):
            # Find corresponding utterance
            utterance = next((u for u in utterances if u['utterance_id'] == annotation['utterance_id']), None)
            
            if utterance:
                with st.expander(f"Instance {i+1}: Turn {utterance['turn_number']} ({utterance['speaker']})"):
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Confidence", f"{annotation['confidence']:.2f}")
                    with col2:
                        st.metric("Severity", annotation['severity'])
                    with col3:
                        st.metric("Word Count", utterance['word_count'])
                    
                    # Show full text with evidence highlighted
                    evidence_span = annotation['evidence_span']
                    text = utterance['text']
                    
                    if 'start_char' in evidence_span and 'end_char' in evidence_span:
                        start = evidence_span['start_char']
                        end = evidence_span['end_char']
                        
                        highlighted_text = (
                            text[:start] + 
                            f"**:red[{text[start:end]}]**" + 
                            text[end:]
                        )
                    else:
                        highlighted_text = text
                    
                    st.markdown("**Full Text:**")
                    st.markdown(highlighted_text)
                    
                    # Show theoretical basis
                    st.markdown(f"**Theoretical Basis:** {annotation['theoretical_basis']}")
                    
                    # Show contextual factors
                    if annotation['contextual_factors']:
                        st.markdown(f"**Contextual Factors:** {', '.join(annotation['contextual_factors'])}")
    
    def create_pattern_statistics_table(self, pattern_summary: Dict[str, Any]) -> None:
        """Create detailed pattern statistics table."""
        
        st.subheader("📊 Pattern Statistics")
        
        # Convert to DataFrame for better display
        stats_data = []
        for pattern, data in pattern_summary.items():
            stats_data.append({
                'Pattern': pattern.replace('_', ' ').title(),
                'Total Count': data['count'],
                'Avg Confidence': f"{data['avg_confidence']:.3f}",
                'Person A': data['speakers']['PERSON_A'],
                'Person B': data['speakers']['PERSON_B'],
                'Dominance': 'Person A' if data['speakers']['PERSON_A'] > data['speakers']['PERSON_B'] else 'Person B' if data['speakers']['PERSON_B'] > data['speakers']['PERSON_A'] else 'Equal'
            })
        
        df = pd.DataFrame(stats_data)
        
        # Style the dataframe
        styled_df = df.style.format({
            'Total Count': '{:d}',
            'Person A': '{:d}',
            'Person B': '{:d}'
        }).apply(lambda x: ['background-color: #ffcccb' if x.name == 'boundary_violation' else 
                          'background-color: #add8e6' if 'accountability' in str(x.name).lower() else 
                          '' for i in x], axis=1)
        
        st.dataframe(styled_df, use_container_width=True)
    
    def create_safety_assessment_panel(self, pattern_summary: Dict[str, Any]) -> None:
        """Create safety assessment panel."""
        
        st.subheader("⚠️ Safety Assessment")
        
        # Calculate safety scores
        safety_patterns = ['boundary_violation', 'power_imbalance', 'invalidation']
        repair_patterns = ['accountability_taking', 'trauma_responsive', 'consent_negotiation']
        
        safety_score = sum(pattern_summary.get(p, {}).get('count', 0) for p in safety_patterns)
        repair_score = sum(pattern_summary.get(p, {}).get('count', 0) for p in repair_patterns)
        
        # Create safety metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Safety Concerns", 
                safety_score,
                help="Total instances of boundary violations, power imbalances, and invalidation"
            )
        
        with col2:
            st.metric(
                "Repair Attempts", 
                repair_score,
                help="Total instances of accountability taking, trauma responsiveness, and consent negotiation"
            )
        
        with col3:
            repair_ratio = repair_score / max(safety_score, 1)
            st.metric(
                "Repair Ratio", 
                f"{repair_ratio:.2f}",
                help="Ratio of repair attempts to safety concerns"
            )
        
        # Safety assessment
        if safety_score > repair_score * 2:
            st.error("🚨 HIGH RISK: Safety concerns significantly outweigh repair attempts")
        elif safety_score > repair_score:
            st.warning("⚠️ MODERATE RISK: Safety concerns outweigh repair attempts")
        elif repair_score > safety_score * 1.5:
            st.success("✅ POSITIVE: Strong repair attempts relative to safety concerns")
        else:
            st.info("ℹ️ BALANCED: Moderate balance between safety concerns and repair attempts")
    
    def run_dashboard(self):
        """Main dashboard interface."""
        
        st.title("🔍 Relational Discourse Analysis Dashboard")
        st.markdown("---")
        
        # Sidebar
        st.sidebar.title("Navigation")
        st.sidebar.markdown("Use this dashboard to explore communication patterns and evidence.")
        
        # Load data
        data = self.load_data()
        
        if not data:
            st.error("Could not load analysis data. Please ensure analysis has been run first.")
            return
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Overview", 
            "📈 Timeline", 
            "🔍 Evidence", 
            "📋 Statistics", 
            "⚠️ Safety"
        ])
        
        with tab1:
            st.header("Pattern Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_overview = self.create_pattern_overview_chart(data['pattern_summary'])
                st.plotly_chart(fig_overview, use_container_width=True)
            
            with col2:
                fig_speakers = self.create_speaker_comparison_chart(data['pattern_summary'])
                st.plotly_chart(fig_speakers, use_container_width=True)
            
            # Key insights
            st.subheader("Key Insights")
            total_patterns = sum(data['pattern_summary'][p]['count'] for p in data['pattern_summary'])
            total_utterances = len(data['utterances'])
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Patterns", total_patterns)
            with col2:
                st.metric("Total Utterances", total_utterances)
            with col3:
                pattern_density = total_patterns / total_utterances
                st.metric("Pattern Density", f"{pattern_density:.2f}")
            with col4:
                person_a_turns = len([u for u in data['utterances'] if u['speaker'] == 'PERSON_A'])
                person_b_turns = len([u for u in data['utterances'] if u['speaker'] == 'PERSON_B'])
                turn_ratio = person_b_turns / max(person_a_turns, 1)
                st.metric("Turn Ratio (B:A)", f"{turn_ratio:.2f}")
        
        with tab2:
            st.header("Pattern Timeline")
            fig_timeline = self.create_timeline_visualization(data['annotations'], data['utterances'])
            st.plotly_chart(fig_timeline, use_container_width=True)
            
            st.markdown("""
            **Timeline Interpretation:**
            - Each dot represents a detected pattern instance
            - Dot size indicates confidence level
            - Colors represent different pattern types
            - Shapes distinguish between speakers
            """)
        
        with tab3:
            self.create_evidence_viewer(data['annotations'], data['utterances'])
        
        with tab4:
            self.create_pattern_statistics_table(data['pattern_summary'])
        
        with tab5:
            self.create_safety_assessment_panel(data['pattern_summary'])
            
            # Additional safety insights
            st.markdown("### Detailed Safety Analysis")
            
            boundary_violations = data['pattern_summary'].get('boundary_violation', {}).get('count', 0)
            power_imbalances = data['pattern_summary'].get('power_imbalance', {}).get('count', 0)
            
            if boundary_violations > 0:
                st.error(f"⚠️ {boundary_violations} boundary violation instances detected")
                st.markdown("This indicates serious consent and safety concerns that require immediate attention.")
            
            if power_imbalances > 0:
                st.warning(f"⚡ {power_imbalances} power imbalance instances detected")
                st.markdown("Power dynamics may be impacting communication equality and consent.")
        
        # Footer
        st.markdown("---")
        st.markdown("*Relational Discourse Analysis Framework - Evidence-based pattern detection*")

def create_dashboard_launcher():
    """Create a simple launcher script for the dashboard."""
    
    launcher_content = '''#!/usr/bin/env python3
"""
Dashboard Launcher
Run this script to start the interactive analysis dashboard.
"""

import subprocess
import sys
from pathlib import Path

def main():
    dashboard_script = Path(__file__).parent / "07_interactive_dashboard.py"
    
    try:
        # Launch Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_script),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\\n📊 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")

if __name__ == "__main__":
    main()
'''
    
    with open("/Users/tiaastor/relational_discourse_project/scripts/launch_dashboard.py", 'w') as f:
        f.write(launcher_content)

def main():
    """Main entry point for dashboard."""
    dashboard = InteractiveDashboard("/Users/tiaastor/relational_discourse_project")
    dashboard.run_dashboard()

if __name__ == "__main__":
    # Create launcher script
    create_dashboard_launcher()
    
    # Run dashboard if called directly
    main()