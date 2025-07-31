import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
import anthropic
import requests
import json
import base64
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from enum import Enum
import re
from datetime import datetime, timedelta
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

class ChartType(Enum):
    BAR = "bar"
    LINE = "line"
    SCATTER = "scatter"
    HISTOGRAM = "histogram"
    BOX = "box"
    HEATMAP = "heatmap"
    PIE = "pie"
    TREEMAP = "treemap"
    VIOLIN = "violin"
    CORRELATION = "correlation"
    DISTRIBUTION = "distribution"
    TIME_SERIES = "time_series"
    GEOGRAPHICAL = "geographical"

@dataclass
class VisualizationRecommendation:
    chart_type: ChartType
    title: str
    description: str
    x_column: str
    y_column: str = None
    color_column: str = None
    size_column: str = None
    confidence_score: float = 0.0
    reasoning: str = ""
    parameters: Dict[str, Any] = None

class IntelligentVisualizer:
    def __init__(self, anthropic_api_key: str = None, gemini_api_key: str = None):
        self.anthropic_api_key = anthropic_api_key
        self.gemini_api_key = gemini_api_key
        
    def analyze_data_characteristics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyzes the dataset to understand its characteristics for visualization."""
        analysis = {
            "shape": df.shape,
            "columns": list(df.columns),
            "dtypes": df.dtypes.to_dict(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "categorical_columns": df.select_dtypes(include=['object']).columns.tolist(),
            "datetime_columns": df.select_dtypes(include=['datetime64']).columns.tolist(),
            "missing_values": df.isnull().sum().to_dict(),
            "unique_counts": {col: df[col].nunique() for col in df.columns},
            "sample_data": df.head(3).to_dict('records'),
            "statistical_summary": df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
        }
        
        # Detect potential time series
        for col in df.columns:
            if df[col].dtype == 'object':
                # Try to detect dates
                sample_values = df[col].dropna().astype(str).head(10).tolist()
                date_patterns = [
                    r'\d{4}-\d{2}-\d{2}',
                    r'\d{1,2}/\d{1,2}/\d{4}',
                    r'\d{1,2}-\d{1,2}-\d{4}',
                    r'\w+ \d{4}',  # Month Year
                    r'Q[1-4] \d{4}'  # Quarter Year
                ]
                
                for pattern in date_patterns:
                    matches = sum(1 for val in sample_values if re.search(pattern, str(val)))
                    if matches > len(sample_values) * 0.7:
                        analysis["potential_time_columns"] = analysis.get("potential_time_columns", []) + [col]
                        break
        
        return analysis

    def create_ai_visualization_prompt(self, df: pd.DataFrame, context: str = "") -> str:
        """Creates a comprehensive prompt for AI-driven visualization recommendations."""
        analysis = self.analyze_data_characteristics(df)
        
        return f"""
        You are an expert data visualization consultant. Analyze this dataset and recommend the most effective visualizations for scientific/research presentation.

        **Dataset Context:** {context}

        **Dataset Characteristics:**
        - Shape: {analysis['shape'][0]} rows × {analysis['shape'][1]} columns
        - Numeric columns: {analysis['numeric_columns']}
        - Categorical columns: {analysis['categorical_columns']}
        - Unique value counts: {analysis['unique_counts']}
        
        **Sample Data:**
        {pd.DataFrame(analysis['sample_data']).to_string()}

        **Statistical Summary:**
        {json.dumps(analysis['statistical_summary'], indent=2)}

        **Task:** Recommend 5-8 different visualizations that would be most effective for this data. Consider:
        1. The academic/research context (this is from PDF appendix tables)
        2. Statistical relationships that should be highlighted
        3. Patterns that might be hidden in the data
        4. Comparisons that would be meaningful
        5. Distribution analysis for key variables

        **Return a JSON response with this exact structure:**
        {{
            "dataset_insights": {{
                "data_type": "research/financial/survey/experimental/other",
                "primary_focus": "descriptive analysis of the main theme",
                "key_variables": ["most important columns"],
                "relationships_to_explore": ["suggested variable pairs/groups"]
            }},
            "visualization_recommendations": [
                {{
                    "chart_type": "bar|line|scatter|histogram|box|heatmap|pie|treemap|correlation|distribution",
                    "title": "Descriptive title for the visualization",
                    "description": "What insights this chart will reveal",
                    "x_column": "column_name",
                    "y_column": "column_name_or_null",
                    "color_column": "column_name_or_null",
                    "size_column": "column_name_or_null",
                    "confidence_score": 0.0-1.0,
                    "reasoning": "Why this visualization is recommended",
                    "parameters": {{
                        "aggregation": "sum|mean|count|none",
                        "sort_by": "column_name_or_null",
                        "filter_conditions": "any specific filters to apply",
                        "additional_settings": "other plotly/chart specific settings"
                    }}
                }}
            ],
            "story_narrative": {{
                "introduction": "Brief overview of what the data shows",
                "key_findings": ["3-5 main insights the visualizations will reveal"],
                "recommended_order": ["suggested order of charts for presentation"]
            }}
        }}
        """

    def get_ai_visualization_recommendations(self, df: pd.DataFrame, context: str = "", 
                                          model: str = "claude-3-5-sonnet-20240620") -> Dict[str, Any]:
        """Gets AI-powered visualization recommendations."""
        prompt = self.create_ai_visualization_prompt(df, context)
        
        try:
            if "claude" in model and self.anthropic_api_key:
                client = anthropic.Anthropic(api_key=self.anthropic_api_key)
                message = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}]
                )
                response_text = message.content[0].text
                
            elif "gemini" in model and self.gemini_api_key:
                payload = {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"responseMimeType": "application/json"}
                }
                api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.gemini_api_key}"
                response = requests.post(api_url, headers={'Content-Type': 'application/json'}, 
                                       json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                response_text = result['candidates'][0]['content']['parts'][0]['text']
            else:
                return {"error": "No valid API key configured"}
            
            # Clean markdown formatting
            if response_text.strip().startswith("```json"):
                response_text = response_text.strip()[7:-3].strip()
            
            return json.loads(response_text)
            
        except Exception as e:
            return {"error": f"AI recommendation failed: {str(e)}"}

    def create_visualization(self, df: pd.DataFrame, recommendation: Dict[str, Any]) -> go.Figure:
        """Creates a plotly visualization based on AI recommendation."""
        chart_type = recommendation.get("chart_type", "bar")
        x_col = recommendation.get("x_column")
        y_col = recommendation.get("y_column")
        color_col = recommendation.get("color_column")
        size_col = recommendation.get("size_column")
        title = recommendation.get("title", "Visualization")
        params = recommendation.get("parameters", {})
        
        # Apply any filters
        plot_df = df.copy()
        if params.get("filter_conditions"):
            # This would need more sophisticated parsing in production
            pass
        
        # Apply aggregation if needed
        if params.get("aggregation") and params["aggregation"] != "none":
            if params["aggregation"] == "mean" and y_col:
                plot_df = plot_df.groupby(x_col)[y_col].mean().reset_index()
            elif params["aggregation"] == "sum" and y_col:
                plot_df = plot_df.groupby(x_col)[y_col].sum().reset_index()
            elif params["aggregation"] == "count":
                plot_df = plot_df.groupby(x_col).size().reset_index(name='count')
                y_col = 'count'
        
        # Sort if specified
        if params.get("sort_by") and params["sort_by"] in plot_df.columns:
            plot_df = plot_df.sort_values(params["sort_by"])
        
        # Create the appropriate chart
        fig = None
        
        if chart_type == "bar":
            fig = px.bar(plot_df, x=x_col, y=y_col, color=color_col, 
                        title=title, template="plotly_white")
        
        elif chart_type == "line":
            fig = px.line(plot_df, x=x_col, y=y_col, color=color_col, 
                         title=title, template="plotly_white")
        
        elif chart_type == "scatter":
            fig = px.scatter(plot_df, x=x_col, y=y_col, color=color_col, 
                           size=size_col, title=title, template="plotly_white")
        
        elif chart_type == "histogram":
            fig = px.histogram(plot_df, x=x_col, color=color_col, 
                             title=title, template="plotly_white")
        
        elif chart_type == "box":
            fig = px.box(plot_df, x=x_col, y=y_col, color=color_col, 
                        title=title, template="plotly_white")
        
        elif chart_type == "pie":
            if params.get("aggregation") == "count":
                value_counts = plot_df[x_col].value_counts()
                fig = px.pie(values=value_counts.values, names=value_counts.index, 
                           title=title, template="plotly_white")
            else:
                fig = px.pie(plot_df, names=x_col, values=y_col, 
                           title=title, template="plotly_white")
        
        elif chart_type == "heatmap":
            # Create correlation heatmap for numeric columns
            numeric_cols = plot_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = plot_df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                              title=title, template="plotly_white")
        
        elif chart_type == "correlation":
            numeric_cols = plot_df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = plot_df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                              title=title, template="plotly_white",
                              color_continuous_scale="RdBu_r")
        
        elif chart_type == "distribution":
            if plot_df[x_col].dtype in ['int64', 'float64']:
                fig = px.histogram(plot_df, x=x_col, marginal="box",
                                 title=title, template="plotly_white")
        
        # Default fallback
        if fig is None:
            fig = px.bar(plot_df, x=x_col, y=y_col, title=title, template="plotly_white")
        
        # Apply consistent styling
        fig.update_layout(
            font=dict(size=12),
            title_font_size=16,
            showlegend=True if color_col else False,
            height=500,
            margin=dict(t=60, b=40, l=40, r=40)
        )
        
        return fig

    def create_statistical_summary_viz(self, df: pd.DataFrame) -> List[go.Figure]:
        """Creates statistical summary visualizations."""
        figures = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            # Distribution overview
            n_cols = min(3, len(numeric_cols))
            n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
            
            fig = make_subplots(
                rows=n_rows, cols=n_cols,
                subplot_titles=list(numeric_cols),
                vertical_spacing=0.1
            )
            
            for i, col in enumerate(numeric_cols):
                row = i // n_cols + 1
                col_idx = i % n_cols + 1
                
                # Add histogram
                fig.add_trace(
                    go.Histogram(x=df[col], name=col, showlegend=False),
                    row=row, col=col_idx
                )
            
            fig.update_layout(
                title="Distribution Analysis of Numeric Variables",
                height=300 * n_rows,
                template="plotly_white"
            )
            figures.append(fig)
        
        return figures

def main():
    st.title("Step 4: 📊 AI-Powered Data Visualization")
    
    # Check if we have data
    if 'extracted_df' not in st.session_state or st.session_state.extracted_df is None:
        st.warning("⚠️ Please extract and validate a table first.")
        return
    
    # Use cleaned data if available, otherwise use extracted data
    if 'cleaned_df' in st.session_state and st.session_state.cleaned_df is not None:
        df = st.session_state.cleaned_df
        st.info("📋 Using cleaned and validated data")
    else:
        df = st.session_state.extracted_df
        st.info("📋 Using original extracted data")
    
    # Initialize visualizer
    visualizer = IntelligentVisualizer(
        anthropic_api_key=st.session_state.get('anthropic_api_key'),
        gemini_api_key=st.session_state.get('gemini_api_key')
    )
    
    # Sidebar for context and settings
    with st.sidebar:
        st.header("🎯 Visualization Settings")
        
        context = st.text_area(
            "Data Context (helps AI make better recommendations):",
            placeholder="e.g., 'Financial performance data from company annual report appendix', 'Clinical trial results', 'Survey responses about customer satisfaction'",
            height=100
        )
        
        ai_model = st.selectbox(
            "AI Model for Recommendations:",
            ["claude-3-5-sonnet-20240620", "claude-3-opus-20240229", "gemini-2.5-pro"]
        )
        
        auto_generate = st.checkbox("Auto-generate visualizations", value=True)
    
    # Data overview
    st.header("📈 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        numeric_cols = len(df.select_dtypes(include=[np.number]).columns)
        st.metric("Numeric Columns", numeric_cols)
    with col4:
        categorical_cols = len(df.select_dtypes(include=['object']).columns)
        st.metric("Categorical Columns", categorical_cols)
    
    # Quick data preview
    with st.expander("📋 Data Preview", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
    
    # AI Recommendations Section
    st.header("🤖 AI Visualization Recommendations")
    
    if st.button("🎨 Get AI Recommendations", type="primary"):
        with st.spinner("🧠 AI is analyzing your data and generating visualization recommendations..."):
            recommendations = visualizer.get_ai_visualization_recommendations(df, context, ai_model)
            st.session_state.viz_recommendations = recommendations
            
            if 'error' not in recommendations:
                st.success("✅ AI recommendations generated!")
            else:
                st.error(f"❌ {recommendations['error']}")
    
    # Display recommendations and generate visualizations
    if 'viz_recommendations' in st.session_state and 'error' not in st.session_state.viz_recommendations:
        recommendations = st.session_state.viz_recommendations
        
        # Display insights
        if 'dataset_insights' in recommendations:
            insights = recommendations['dataset_insights']
            st.subheader("🔍 Dataset Insights")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Data Type**: {insights.get('data_type', 'Unknown')}")
                st.info(f"**Primary Focus**: {insights.get('primary_focus', 'General analysis')}")
            with col2:
                if insights.get('key_variables'):
                    st.success(f"**Key Variables**: {', '.join(insights['key_variables'])}")
        
        # Story narrative
        if 'story_narrative' in recommendations:
            narrative = recommendations['story_narrative']
            with st.expander("📖 Data Story", expanded=True):
                st.markdown(f"**Overview**: {narrative.get('introduction', '')}")
                if narrative.get('key_findings'):
                    st.markdown("**Key Findings**:")
                    for finding in narrative['key_findings']:
                        st.markdown(f"• {finding}")
        
        # Generate visualizations
        st.header("📊 AI-Generated Visualizations")
        
        if 'visualization_recommendations' in recommendations:
            viz_recs = recommendations['visualization_recommendations']
            
            # Create tabs for different visualizations
            tab_names = [f"{i+1}. {rec.get('title', f'Chart {i+1}')}" for i, rec in enumerate(viz_recs)]
            tabs = st.tabs(tab_names)
            
            for i, (tab, rec) in enumerate(zip(tabs, viz_recs)):
                with tab:
                    col1, col2 = st.columns([3, 1])
                    
                    with col2:
                        st.metric("Confidence", f"{rec.get('confidence_score', 0):.1%}")
                        with st.expander("💡 AI Reasoning"):
                            st.write(rec.get('reasoning', 'No reasoning provided'))
                            st.write(f"**Description**: {rec.get('description', '')}")
                            
                            # Show parameters
                            if rec.get('parameters'):
                                st.json(rec['parameters'])
                    
                    with col1:
                        try:
                            fig = visualizer.create_visualization(df, rec)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Download button for the chart
                            img_bytes = fig.to_image(format="png", width=1200, height=600)
                            st.download_button(
                                label="📥 Download Chart",
                                data=img_bytes,
                                file_name=f"chart_{i+1}_{rec.get('chart_type', 'viz')}.png",
                                mime="image/png"
                            )
                            
                        except Exception as e:
                            st.error(f"Error creating visualization: {str(e)}")
                            st.code(f"Recommendation: {json.dumps(rec, indent=2)}")
    
    # Manual visualization creation
    st.header("🛠️ Custom Visualization Builder")
    
    with st.expander("Create Custom Charts", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            chart_type = st.selectbox(
                "Chart Type:",
                ["bar", "line", "scatter", "histogram", "box", "pie", "heatmap"]
            )
        
        with col2:
            x_column = st.selectbox("X-axis:", df.columns)
        
        with col3:
            numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            y_column = st.selectbox("Y-axis:", [None] + list(df.columns), index=0)
        
        color_column = st.selectbox("Color by:", [None] + list(df.columns), index=0)
        
        if st.button("Create Custom Chart"):
            custom_rec = {
                "chart_type": chart_type,
                "title": f"Custom {chart_type.title()} Chart",
                "x_column": x_column,
                "y_column": y_column,
                "color_column": color_column,
                "parameters": {}
            }
            
            try:
                fig = visualizer.create_visualization(df, custom_rec)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error creating custom chart: {str(e)}")
    
    # Statistical Analysis Section
    st.header("📈 Statistical Analysis")
    
    if st.button("Generate Statistical Summary"):
        summary_figs = visualizer.create_statistical_summary_viz(df)
        for fig in summary_figs:
            st.plotly_chart(fig, use_container_width=True)
    
    # Export options
    st.header("📤 Export Options")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("📊 Generate Visualization Report"):
            # Create a comprehensive report
            st.info("This would generate a comprehensive PDF report with all visualizations and insights")
    
    with col2:
        if st.button("🔗 Share Dashboard"):
            st.info("This would create a shareable link to this dashboard")

if __name__ == "__main__":
    main()