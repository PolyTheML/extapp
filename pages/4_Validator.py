import streamlit as st
import pandas as pd
import numpy as np
import cv2
import base64
import json
import anthropic
import requests
from typing import Dict, List, Tuple, Any
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from enum import Enum

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

class TableValidator:
    def __init__(self, anthropic_api_key: str = None, gemini_api_key: str = None):
        self.anthropic_api_key = anthropic_api_key
        self.gemini_api_key = gemini_api_key
        self.validation_results = []
        
    def create_ai_validation_prompt(self, df: pd.DataFrame, original_image_b64: str = None) -> str:
        """Creates a comprehensive prompt for AI-based validation."""
        return f"""
        You are an expert data quality analyst. Analyze the extracted table data and provide a comprehensive validation report.

        **Table Structure:**
        - Rows: {len(df)}
        - Columns: {len(df.columns)}
        - Column Names: {list(df.columns)}

        **Sample Data (first 5 rows):**
        {df.head().to_string()}

        **Data Types:**
        {df.dtypes.to_string()}

        **Task: Perform comprehensive validation and return a JSON response with the following structure:**
        {{
            "overall_quality_score": 0.0-1.0,
            "issues_found": [
                {{
                    "type": "data_type_mismatch|missing_values|formatting_error|logical_inconsistency|structural_issue",
                    "severity": "critical|warning|info",
                    "column": "column_name_or_null",
                    "description": "detailed description",
                    "affected_rows": [row_indices],
                    "suggested_fix": "specific recommendation"
                }}
            ],
            "data_insights": {{
                "potential_data_types": {{"column_name": "suggested_type"}},
                "patterns_detected": ["list of patterns found"],
                "anomalies": ["list of anomalies"],
                "completeness_score": 0.0-1.0
            }},
            "extraction_quality": {{
                "table_structure_correct": true/false,
                "headers_properly_extracted": true/false,
                "data_alignment_issues": true/false,
                "ocr_quality_estimate": 0.0-1.0
            }},
            "recommendations": [
                {{
                    "action": "specific action to take",
                    "priority": "high|medium|low",
                    "impact": "description of impact"
                }}
            ]
        }}

        Focus on:
        1. Data type consistency and appropriateness
        2. Missing or null values and their patterns
        3. Formatting inconsistencies (dates, numbers, text)
        4. Logical relationships between columns
        5. Potential OCR extraction errors
        6. Structural integrity of the table
        """

    def validate_with_ai(self, df: pd.DataFrame, model: str = "claude-3-5-sonnet-20240620", 
                        original_image_b64: str = None) -> Dict[str, Any]:
        """Uses AI to perform intelligent validation of the extracted table."""
        prompt = self.create_ai_validation_prompt(df, original_image_b64)
        
        try:
            if "claude" in model and self.anthropic_api_key:
                client = anthropic.Anthropic(api_key=self.anthropic_api_key)
                message = client.messages.create(
                    model=model,
                    max_tokens=4096,
                    messages=[{
                        "role": "user", 
                        "content": [{"type": "text", "text": prompt}]
                    }]
                )
                response_text = message.content[0].text
                
            elif "gemini" in model and self.gemini_api_key:
                payload = {
                    "contents": [{
                        "role": "user",
                        "parts": [{"text": prompt}]
                    }],
                    "generationConfig": {"responseMimeType": "application/json"}
                }
                api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={self.gemini_api_key}"
                response = requests.post(api_url, headers={'Content-Type': 'application/json'}, 
                                       json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                response_text = result['candidates'][0]['content']['parts'][0]['text']
            else:
                return {"error": "No valid API key or model specified"}
            
            # Clean potential markdown formatting
            if response_text.strip().startswith("```json"):
                response_text = response_text.strip()[7:-3].strip()
            
            return json.loads(response_text)
            
        except Exception as e:
            return {"error": f"AI validation failed: {str(e)}"}

    def validate_data_types(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validates data types and suggests improvements."""
        results = []
        
        for col in df.columns:
            # Check for mixed data types
            if df[col].dtype == 'object':
                numeric_count = sum(pd.to_numeric(df[col], errors='coerce').notna())
                total_count = len(df[col].dropna())
                
                if total_count > 0:
                    numeric_ratio = numeric_count / total_count
                    
                    if numeric_ratio > 0.8:
                        results.append(ValidationResult(
                            check_name="Data Type Consistency",
                            severity=ValidationSeverity.WARNING,
                            message=f"Column '{col}' appears to be mostly numeric but stored as text",
                            details={"numeric_ratio": numeric_ratio},
                            suggested_fix=f"Convert column '{col}' to numeric type"
                        ))
        
        return results

    def validate_missing_values(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Analyzes missing value patterns."""
        results = []
        
        missing_summary = df.isnull().sum()
        total_rows = len(df)
        
        for col, missing_count in missing_summary.items():
            if missing_count > 0:
                missing_ratio = missing_count / total_rows
                
                if missing_ratio > 0.5:
                    severity = ValidationSeverity.CRITICAL
                elif missing_ratio > 0.2:
                    severity = ValidationSeverity.WARNING
                else:
                    severity = ValidationSeverity.INFO
                
                results.append(ValidationResult(
                    check_name="Missing Values",
                    severity=severity,
                    message=f"Column '{col}' has {missing_count} missing values ({missing_ratio:.1%})",
                    details={"missing_count": missing_count, "missing_ratio": missing_ratio},
                    suggested_fix="Review extraction quality or fill missing values appropriately"
                ))
        
        return results

    def validate_duplicates(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Checks for duplicate rows and unusual patterns."""
        results = []
        
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            results.append(ValidationResult(
                check_name="Duplicate Detection",
                severity=ValidationSeverity.WARNING,
                message=f"Found {duplicate_rows} duplicate rows",
                details={"duplicate_count": duplicate_rows},
                suggested_fix="Remove duplicates or verify if they are legitimate"
            ))
        
        return results

    def validate_format_consistency(self, df: pd.DataFrame) -> List[ValidationResult]:
        """Validates format consistency within columns."""
        results = []
        
        for col in df.columns:
            if df[col].dtype == 'object':
                values = df[col].dropna().astype(str)
                
                # Check for date-like patterns
                date_patterns = [
                    r'\d{1,2}/\d{1,2}/\d{4}',
                    r'\d{4}-\d{2}-\d{2}',
                    r'\d{1,2}-\d{1,2}-\d{4}'
                ]
                
                for pattern in date_patterns:
                    matches = values.str.match(pattern).sum()
                    if matches > len(values) * 0.7:  # 70% match threshold
                        results.append(ValidationResult(
                            check_name="Format Consistency",
                            severity=ValidationSeverity.INFO,
                            message=f"Column '{col}' appears to contain dates",
                            details={"pattern": pattern, "matches": matches},
                            suggested_fix=f"Consider converting column '{col}' to datetime"
                        ))
                        break
        
        return results

    def create_validation_report(self, df: pd.DataFrame, ai_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """Creates a comprehensive validation report."""
        
        # Traditional validation checks
        traditional_results = []
        traditional_results.extend(self.validate_data_types(df))
        traditional_results.extend(self.validate_missing_values(df))
        traditional_results.extend(self.validate_duplicates(df))
        traditional_results.extend(self.validate_format_consistency(df))
        
        # Combine with AI results
        report = {
            "basic_stats": {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "memory_usage": df.memory_usage(deep=True).sum(),
                "completeness": (1 - df.isnull().sum().sum() / (len(df) * len(df.columns)))
            },
            "traditional_validation": [
                {
                    "check_name": result.check_name,
                    "severity": result.severity.value,
                    "message": result.message,
                    "details": result.details or {},
                    "suggested_fix": result.suggested_fix
                } for result in traditional_results
            ],
            "ai_validation": ai_results or {}
        }
        
        return report

# Streamlit UI for the validator
def main():
    st.title("Step 3: 🤖 AI-Powered Table Validator")
    
    # Check if we have extracted data
    if 'extracted_df' not in st.session_state or st.session_state.extracted_df is None:
        st.warning("⚠️ Please extract a table first using the **🔎 AI-Powered Table Extractor**.")
        return
    
    df = st.session_state.extracted_df
    
    # Initialize validator
    validator = TableValidator(
        anthropic_api_key=st.session_state.get('anthropic_api_key'),
        gemini_api_key=st.session_state.get('gemini_api_key')
    )
    
    st.header("📊 Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", len(df))
    with col2:
        st.metric("Columns", len(df.columns))
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    with col4:
        completeness = (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        st.metric("Completeness", f"{completeness:.1f}%")
    
    # Display the data
    st.subheader("🔍 Current Data")
    st.dataframe(df, use_container_width=True)
    
    # Validation controls
    st.header("🛠️ Validation Controls")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        validation_mode = st.selectbox(
            "Choose Validation Mode:",
            ["Quick Traditional Validation", "AI-Enhanced Validation", "Comprehensive Validation"]
        )
    
    with col2:
        if st.button("🚀 Run Validation", type="primary"):
            with st.spinner("Running validation..."):
                
                if validation_mode == "Quick Traditional Validation":
                    report = validator.create_validation_report(df)
                    
                elif validation_mode == "AI-Enhanced Validation":
                    ai_results = validator.validate_with_ai(df)
                    report = validator.create_validation_report(df, ai_results)
                    
                else:  # Comprehensive
                    ai_results = validator.validate_with_ai(df)
                    report = validator.create_validation_report(df, ai_results)
                
                st.session_state.validation_report = report
                st.success("✅ Validation completed!")
    
    # Display validation results
    if 'validation_report' in st.session_state:
        report = st.session_state.validation_report
        
        st.header("📋 Validation Results")
        
        # Basic statistics
        st.subheader("📈 Basic Statistics")
        stats = report['basic_stats']
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", stats['total_rows'])
        with col2:
            st.metric("Total Columns", stats['total_columns'])
        with col3:
            st.metric("Memory Usage", f"{stats['memory_usage'] / 1024:.1f} KB")
        with col4:
            st.metric("Completeness", f"{stats['completeness']:.1%}")
        
        # Traditional validation results
        if report['traditional_validation']:
            st.subheader("🔧 Traditional Validation Issues")
            
            for issue in report['traditional_validation']:
                severity = issue['severity']
                if severity == 'critical':
                    st.error(f"**{issue['check_name']}**: {issue['message']}")
                elif severity == 'warning':
                    st.warning(f"**{issue['check_name']}**: {issue['message']}")
                else:
                    st.info(f"**{issue['check_name']}**: {issue['message']}")
                
                if issue['suggested_fix']:
                    st.caption(f"💡 Suggested fix: {issue['suggested_fix']}")
        
        # AI validation results
        if report.get('ai_validation') and 'overall_quality_score' in report['ai_validation']:
            ai_report = report['ai_validation']
            
            st.subheader("🤖 AI Validation Analysis")
            
            # Overall quality score
            quality_score = ai_report.get('overall_quality_score', 0)
            st.metric("Overall Quality Score", f"{quality_score:.1%}")
            
            # Quality gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=quality_score * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Data Quality Score"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Issues found by AI
            if ai_report.get('issues_found'):
                st.subheader("🚨 AI-Detected Issues")
                for issue in ai_report['issues_found']:
                    severity = issue.get('severity', 'info')
                    if severity == 'critical':
                        st.error(f"**{issue['type']}**: {issue['description']}")
                    elif severity == 'warning':
                        st.warning(f"**{issue['type']}**: {issue['description']}")
                    else:
                        st.info(f"**{issue['type']}**: {issue['description']}")
                    
                    if issue.get('suggested_fix'):
                        st.caption(f"💡 {issue['suggested_fix']}")
            
            # Recommendations
            if ai_report.get('recommendations'):
                st.subheader("💡 AI Recommendations")
                for rec in ai_report['recommendations']:
                    priority = rec.get('priority', 'medium')
                    priority_emoji = {'high': '🔥', 'medium': '⚠️', 'low': 'ℹ️'}.get(priority, 'ℹ️')
                    st.write(f"{priority_emoji} **{rec['action']}** ({rec.get('priority', 'medium')} priority)")
                    if rec.get('impact'):
                        st.caption(f"Impact: {rec['impact']}")
        
        # Data cleaning suggestions
        st.header("🧹 Data Cleaning Tools")
        
        cleaning_options = st.multiselect(
            "Select cleaning operations to apply:",
            [
                "Remove duplicate rows",
                "Fill missing values with forward fill",
                "Convert numeric columns",
                "Standardize text case",
                "Remove leading/trailing spaces"
            ]
        )
        
        if cleaning_options and st.button("Apply Cleaning Operations"):
            cleaned_df = df.copy()
            
            for operation in cleaning_options:
                if operation == "Remove duplicate rows":
                    cleaned_df = cleaned_df.drop_duplicates()
                elif operation == "Fill missing values with forward fill":
                    cleaned_df = cleaned_df.fillna(method='ffill')
                elif operation == "Convert numeric columns":
                    for col in cleaned_df.columns:
                        if cleaned_df[col].dtype == 'object':
                            numeric_converted = pd.to_numeric(cleaned_df[col], errors='coerce')
                            if not numeric_converted.isna().all():
                                cleaned_df[col] = numeric_converted
                elif operation == "Standardize text case":
                    for col in cleaned_df.select_dtypes(include=['object']).columns:
                        cleaned_df[col] = cleaned_df[col].astype(str).str.title()
                elif operation == "Remove leading/trailing spaces":
                    for col in cleaned_df.select_dtypes(include=['object']).columns:
                        cleaned_df[col] = cleaned_df[col].astype(str).str.strip()
            
            st.session_state.cleaned_df = cleaned_df
            st.success("✅ Cleaning operations applied!")
            st.subheader("🔄 Cleaned Data Preview")
            st.dataframe(cleaned_df, use_container_width=True)

if __name__ == "__main__":
    main()