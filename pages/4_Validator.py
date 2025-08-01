import streamlit as st
import pandas as pd
import numpy as np
import json
import anthropic
import openai
import requests
from typing import Dict, List, Tuple, Any, Optional
import re
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

class ValidationSeverity(Enum):
    CRITICAL = "critical"
    WARNING = "warning" 
    INFO = "info"
    SUCCESS = "success"

@dataclass
class ValidationResult:
    check_name: str
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any] = None
    suggested_fix: str = None
    auto_fixable: bool = False

class SmartTableValidator:
    def __init__(self, anthropic_api_key: str = None, gemini_api_key: str = None, openai_api_key: str = None):
        self.anthropic_api_key = anthropic_api_key
        self.gemini_api_key = gemini_api_key
        self.openai_api_key = openai_api_key
        self.validation_cache = {}
        
    def quick_health_check(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Lightning-fast initial assessment of data quality."""
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        
        # Quick metrics
        health_score = max(0, 1 - (missing_cells / total_cells) * 2)
        
        issues = []
        if missing_cells > total_cells * 0.3:
            issues.append("High missing data")
        if len(df.columns) != len(set(df.columns)):
            issues.append("Duplicate columns")
        if df.duplicated().sum() > len(df) * 0.1:
            issues.append("Many duplicates")
            
        return {
            "health_score": health_score,
            "critical_issues": issues,
            "completeness": 1 - (missing_cells / total_cells),
            "shape_quality": "good" if len(df) > 0 and len(df.columns) > 0 else "poor"
        }
    
    def smart_type_detection(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Intelligent data type detection with confidence scores."""
        type_suggestions = {}
        
        for col in df.columns:
            series = df[col].dropna()
            if len(series) == 0:
                continue
                
            suggestions = {
                'current': str(df[col].dtype),
                'recommended': None,
                'confidence': 0.0,
                'sample_values': series.head(3).tolist(),
                'issues': []
            }
            
            # Convert to string for analysis
            str_series = series.astype(str)
            
            # Check for numeric
            numeric_converted = pd.to_numeric(series, errors='coerce')
            numeric_ratio = numeric_converted.notna().sum() / len(series)
            
            # Check for dates
            date_patterns = [
                (r'^\d{4}-\d{2}-\d{2}$', 'YYYY-MM-DD'),
                (r'^\d{1,2}/\d{1,2}/\d{4}$', 'MM/DD/YYYY'),
                (r'^\d{1,2}-\d{1,2}-\d{4}$', 'MM-DD-YYYY'),
            ]
            
            date_matches = 0
            detected_pattern = None
            for pattern, format_name in date_patterns:
                matches = str_series.str.match(pattern, na=False).sum()
                if matches > date_matches:
                    date_matches = matches
                    detected_pattern = format_name
            
            date_ratio = date_matches / len(series)
            
            # Determine best type
            if numeric_ratio > 0.9:
                if numeric_converted.notna().sum() > 0:
                    if (numeric_converted % 1 == 0).all():
                        suggestions['recommended'] = 'integer'
                    else:
                        suggestions['recommended'] = 'float'
                    suggestions['confidence'] = numeric_ratio
            elif date_ratio > 0.8:
                suggestions['recommended'] = 'datetime'
                suggestions['confidence'] = date_ratio
                suggestions['date_format'] = detected_pattern
            elif df[col].dtype == 'object':
                suggestions['recommended'] = 'string'
                suggestions['confidence'] = 1.0
            
            # Add issues
            if numeric_ratio > 0.5 and numeric_ratio < 0.9:
                suggestions['issues'].append(f"Mixed numeric/text ({numeric_ratio:.1%} numeric)")
            if len(series.unique()) == 1:
                suggestions['issues'].append("All values identical")
            if len(series.unique()) / len(series) > 0.95:
                suggestions['issues'].append("Mostly unique values")
                
            type_suggestions[col] = suggestions
        
        return type_suggestions
    
    def create_smart_ai_prompt(self, df: pd.DataFrame, type_analysis: Dict) -> str:
        """Creates an optimized prompt for AI validation."""
        sample_data = df.head(10).to_dict('records')
        
        return f"""
        Analyze this extracted table data for quality and suggest improvements:

        **STRUCTURE:** {len(df)} rows × {len(df.columns)} columns
        **COLUMNS:** {list(df.columns)}
        
        **TYPE ANALYSIS:**
        {json.dumps({k: {
            'current': v['current'],
            'recommended': v['recommended'], 
            'confidence': v['confidence'],
            'issues': v['issues']
        } for k, v in type_analysis.items()}, indent=2)}

        **SAMPLE DATA (first 10 rows):**
        {json.dumps(sample_data, indent=2, default=str)}

        Return JSON with:
        {{
            "quality_score": 0.0-1.0,
            "extraction_issues": [
                {{
                    "type": "ocr_error|alignment_issue|missing_header|wrong_datatype",
                    "severity": "critical|warning|info",
                    "column": "column_name",
                    "description": "brief issue description",
                    "auto_fix": "specific pandas code to fix" or null,
                    "manual_review": true/false
                }}
            ],
            "transformation_suggestions": [
                {{
                    "operation": "convert_types|clean_text|parse_dates|fill_missing",
                    "columns": ["col1", "col2"],
                    "method": "specific method",
                    "benefit": "why this helps"
                }}
            ],
            "data_insights": {{
                "potential_relationships": ["col1 correlates with col2"],
                "outliers_detected": ["unusual values in col3"],
                "patterns": ["dates follow YYYY-MM-DD format"]
            }}
        }}
        """

    def validate_with_ai(self, df: pd.DataFrame, model: str, type_analysis: Dict) -> Dict[str, Any]:
        """Optimized AI validation with better error handling."""
        prompt = self.create_smart_ai_prompt(df, type_analysis)
        
        try:
            if "claude" in model and self.anthropic_api_key:
                client = anthropic.Anthropic(api_key=self.anthropic_api_key)
                message = client.messages.create(
                    model=model,
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = message.content[0].text
                
            elif "gpt" in model and self.openai_api_key:
                client = openai.OpenAI(api_key=self.openai_api_key)
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=4000
                )
                response_text = response.choices[0].message.content
                
            elif "gemini" in model and self.gemini_api_key:
                payload = {
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generationConfig": {"responseMimeType": "application/json"}
                }
                api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.gemini_api_key}"
                response = requests.post(api_url, json=payload, timeout=60)
                response.raise_for_status()
                response_text = response.json()['candidates'][0]['content']['parts'][0]['text']
            else:
                return {"error": "No valid API key configured"}
            
            # Clean response
            if response_text.strip().startswith("```json"):
                response_text = response_text.strip()[7:-3].strip()
            
            return json.loads(response_text)
            
        except Exception as e:
            return {"error": f"AI validation failed: {str(e)}"}

class DataTransformer:
    """Handles all data transformation operations."""
    
    @staticmethod
    def auto_convert_types(df: pd.DataFrame, type_suggestions: Dict) -> pd.DataFrame:
        """Automatically converts columns to suggested types."""
        df_transformed = df.copy()
        conversion_log = []
        
        for col, suggestion in type_suggestions.items():
            if suggestion['confidence'] > 0.8 and suggestion['recommended']:
                try:
                    if suggestion['recommended'] == 'integer':
                        df_transformed[col] = pd.to_numeric(df_transformed[col], errors='coerce').astype('Int64')
                        conversion_log.append(f"✅ {col} → integer")
                    elif suggestion['recommended'] == 'float':
                        df_transformed[col] = pd.to_numeric(df_transformed[col], errors='coerce')
                        conversion_log.append(f"✅ {col} → float")
                    elif suggestion['recommended'] == 'datetime':
                        df_transformed[col] = pd.to_datetime(df_transformed[col], errors='coerce')
                        conversion_log.append(f"✅ {col} → datetime")
                except Exception as e:
                    conversion_log.append(f"❌ {col} conversion failed: {str(e)}")
        
        return df_transformed, conversion_log
    
    @staticmethod
    def smart_missing_value_handler(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Intelligently handles missing values based on column type and pattern."""
        df_filled = df.copy()
        fill_log = []
        
        for col in df_filled.columns:
            missing_count = df_filled[col].isnull().sum()
            if missing_count == 0:
                continue
                
            if df_filled[col].dtype in ['int64', 'float64', 'Int64']:
                # Use median for numeric columns
                fill_value = df_filled[col].median()
                df_filled[col].fillna(fill_value, inplace=True)
                fill_log.append(f"📊 {col}: filled {missing_count} missing with median ({fill_value})")
            elif df_filled[col].dtype == 'datetime64[ns]':
                # Forward fill for dates
                df_filled[col].fillna(method='ffill', inplace=True)
                fill_log.append(f"📅 {col}: forward filled {missing_count} missing dates")
            else:
                # Mode for categorical
                mode_value = df_filled[col].mode()
                if len(mode_value) > 0:
                    df_filled[col].fillna(mode_value[0], inplace=True)
                    fill_log.append(f"📝 {col}: filled {missing_count} missing with mode ('{mode_value[0]}')")
        
        return df_filled, fill_log
    
    @staticmethod
    def clean_text_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Cleans text columns by removing extra spaces, standardizing case."""
        df_clean = df.copy()
        clean_log = []
        
        text_cols = df_clean.select_dtypes(include=['object']).columns
        
        for col in text_cols:
            original_values = df_clean[col].notna().sum()
            
            # Remove extra whitespace
            df_clean[col] = df_clean[col].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)
            
            # Handle common OCR errors
            df_clean[col] = df_clean[col].str.replace('O', '0', regex=False).replace('l', '1', regex=False)
            
            clean_log.append(f"🧹 {col}: cleaned text formatting")
        
        return df_clean, clean_log

def main():
    st.set_page_config(page_title="Smart Table Validator", layout="wide")
    st.title("Step 3: 🤖 Smart Table Validator & Transformer")
    
    # Check prerequisites
    if 'extracted_df' not in st.session_state or st.session_state.extracted_df is None:
        st.error("⚠️ No table data found. Please extract a table first using the **🔎 AI-Powered Table Extractor**.")
        st.stop()
    
    df = st.session_state.extracted_df.copy()
    
    # Initialize validator
    validator = SmartTableValidator(
        anthropic_api_key=st.session_state.get('anthropic_api_key'),
        gemini_api_key=st.session_state.get('gemini_api_key'),
        openai_api_key=st.session_state.get('openai_api_key')
    )
    
    # === QUICK HEALTH CHECK ===
    with st.container():
        st.header("🏥 Quick Health Check")
        
        health_check = validator.quick_health_check(df)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            score_color = "green" if health_check['health_score'] > 0.8 else "orange" if health_check['health_score'] > 0.6 else "red"
            st.metric("Health Score", f"{health_check['health_score']:.1%}", 
                     delta=f"{score_color}")
        with col2:
            st.metric("Data Shape", f"{len(df)} × {len(df.columns)}")
        with col3:
            st.metric("Completeness", f"{health_check['completeness']:.1%}")
        with col4:
            st.metric("Critical Issues", len(health_check['critical_issues']))
        
        if health_check['critical_issues']:
            st.warning("⚠️ Critical Issues: " + ", ".join(health_check['critical_issues']))
    
    # === SMART TYPE ANALYSIS ===
    st.header("🔍 Smart Type Analysis")
    
    if st.button("🚀 Run Smart Analysis", type="primary"):
        with st.spinner("Analyzing data types and patterns..."):
            type_analysis = validator.smart_type_detection(df)
            st.session_state.type_analysis = type_analysis
    
    if 'type_analysis' in st.session_state:
        type_analysis = st.session_state.type_analysis
        
        # Display type recommendations
        st.subheader("📋 Type Recommendations")
        
        recommendations_data = []
        for col, analysis in type_analysis.items():
            recommendations_data.append({
                'Column': col,
                'Current Type': analysis['current'],
                'Recommended': analysis['recommended'] or 'No change',
                'Confidence': f"{analysis['confidence']:.1%}",
                'Issues': ', '.join(analysis['issues']) if analysis['issues'] else 'None',
                'Sample': ', '.join(map(str, analysis['sample_values'][:2]))
            })
        
        recommendations_df = pd.DataFrame(recommendations_data)
        st.dataframe(recommendations_df, use_container_width=True)
        
        # === AI VALIDATION ===
        st.header("🤖 AI-Enhanced Validation")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            ai_model = st.selectbox(
                "Select AI Model for Deep Analysis:",
                ["claude-3-5-sonnet-20240620", "gpt-4o", "gpt-4-turbo", "gemini-2.5-pro"]
            )
        
        with col2:
            if st.button("🔬 Deep AI Analysis"):
                with st.spinner(f"Running AI analysis with {ai_model}..."):
                    ai_results = validator.validate_with_ai(df, ai_model, type_analysis)
                    st.session_state.ai_results = ai_results
        
        # === TRANSFORMATION TOOLS ===
        st.header("⚡ Smart Transformations")
        
        transformer = DataTransformer()
        
        transform_options = st.columns(3)
        
        with transform_options[0]:
            if st.button("🔄 Auto-Convert Types", help="Automatically convert columns to optimal types"):
                with st.spinner("Converting data types..."):
                    transformed_df, log = transformer.auto_convert_types(df, type_analysis)
                    st.session_state.transformed_df = transformed_df
                    st.session_state.transform_log = log
                    st.success("✅ Types converted!")
                    for entry in log:
                        st.write(entry)
        
        with transform_options[1]:
            if st.button("🔧 Smart Fill Missing", help="Intelligently fill missing values"):
                current_df = st.session_state.get('transformed_df', df)
                with st.spinner("Filling missing values..."):
                    filled_df, log = transformer.smart_missing_value_handler(current_df)
                    st.session_state.transformed_df = filled_df
                    st.success("✅ Missing values handled!")
                    for entry in log:
                        st.write(entry)
        
        with transform_options[2]:
            if st.button("🧹 Clean Text", help="Clean and standardize text columns"):
                current_df = st.session_state.get('transformed_df', df)
                with st.spinner("Cleaning text..."):
                    cleaned_df, log = transformer.clean_text_columns(current_df)
                    st.session_state.transformed_df = cleaned_df
                    st.success("✅ Text cleaned!")
                    for entry in log:
                        st.write(entry)
    
    # === RESULTS COMPARISON ===
    if 'transformed_df' in st.session_state:
        st.header("📊 Before vs After Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔴 Original Data")
            st.dataframe(df.head(10), use_container_width=True)
            st.caption(f"Shape: {df.shape}, Missing: {df.isnull().sum().sum()}")
        
        with col2:
            st.subheader("🟢 Transformed Data")
            transformed_df = st.session_state.transformed_df
            st.dataframe(transformed_df.head(10), use_container_width=True)
            st.caption(f"Shape: {transformed_df.shape}, Missing: {transformed_df.isnull().sum().sum()}")
        
        # Quality improvement metrics
        original_missing = df.isnull().sum().sum()
        new_missing = transformed_df.isnull().sum().sum()
        improvement = max(0, (original_missing - new_missing) / max(original_missing, 1))
        
        st.metric("Quality Improvement", f"{improvement:.1%}", 
                 delta=f"Reduced {original_missing - new_missing} missing values")
        
        # === EXPORT OPTIONS ===
        st.header("📥 Export Transformed Data")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filename = st.text_input("Filename", value="validated_table")
        
        with col2:
            export_format = st.selectbox("Format", ["Excel (.xlsx)", "CSV (.csv)", "JSON (.json)"])
        
        with col3:
            if st.button("📄 Export Data", type="primary"):
                if export_format == "Excel (.xlsx)":
                    from io import BytesIO
                    output = BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        transformed_df.to_excel(writer, index=False, sheet_name='Validated_Data')
                    data = output.getvalue()
                    mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    file_ext = "xlsx"
                elif export_format == "CSV (.csv)":
                    data = transformed_df.to_csv(index=False).encode('utf-8')
                    mime_type = "text/csv"
                    file_ext = "csv"
                else:  # JSON
                    data = transformed_df.to_json(orient='records', indent=2).encode('utf-8')
                    mime_type = "application/json"
                    file_ext = "json"
                
                st.download_button(
                    label=f"⬇️ Download {export_format}",
                    data=data,
                    file_name=f"{filename}.{file_ext}",
                    mime=mime_type
                )
    
    # === AI RESULTS DISPLAY ===
    if 'ai_results' in st.session_state and 'error' not in st.session_state.ai_results:
        st.header("🧠 AI Analysis Results")
        
        ai_results = st.session_state.ai_results
        
        # Quality score with gauge
        if 'quality_score' in ai_results:
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=ai_results['quality_score'] * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "AI Quality Score"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 60], 'color': "lightgray"},
                        {'range': [60, 85], 'color': "yellow"},
                        {'range': [85, 100], 'color': "green"}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 90}
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Issues and suggestions
        if 'extraction_issues' in ai_results:
            st.subheader("🚨 Detected Issues")
            for issue in ai_results['extraction_issues']:
                severity = issue.get('severity', 'info')
                if severity == 'critical':
                    st.error(f"**{issue['type']}** in {issue.get('column', 'table')}: {issue['description']}")
                elif severity == 'warning':
                    st.warning(f"**{issue['type']}** in {issue.get('column', 'table')}: {issue['description']}")
                else:
                    st.info(f"**{issue['type']}** in {issue.get('column', 'table')}: {issue['description']}")
                
                if issue.get('auto_fix'):
                    st.code(f"Auto-fix: {issue['auto_fix']}")

if __name__ == "__main__":
    main()