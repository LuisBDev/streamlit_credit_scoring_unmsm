"""
=================================================================
SISTEMA DE CREDIT SCORING PARA MICROFINANZAS - PERÚ
Modelo basado en Random Forest
Tesis: Desarrollo de un Modelo de Credit Scoring para Evaluación
del Riesgo Crediticio en el Sector de Microfinanzas
=================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import boxcox
from datetime import datetime
import os

# =================================================================
# CONFIGURACIÓN DE LA PÁGINA
# =================================================================

st.set_page_config(
    page_title="Credit Scoring - Microfinanzas Perú",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =================================================================
# ESTILOS CSS PERSONALIZADOS
# =================================================================

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #e3f2fd 0%, #bbdefb 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .risk-low {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        font-weight: bold;
    }
    .risk-medium {
        background-color: #fff3cd;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        font-weight: bold;
    }
    .risk-high {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        font-weight: bold;
    }
    .info-box {
        background-color: #e7f3ff;
        padding: 1rem;
        border-radius: 5px;
        border-left: 3px solid #2196F3;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# =================================================================
# FUNCIONES DE CARGA Y CONFIGURACIÓN
# =================================================================

@st.cache_resource
def load_model():
    """Carga el modelo de Random Forest entrenado"""
    try:
        model_path = "results/models/credit_scoring_model_20250917_154932.pkl"
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            return model
        else:
            st.error(f"No se encontró el modelo en: {model_path}")
            return None
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

@st.cache_data
def load_model_metrics():
    """Carga las métricas del modelo"""
    return {
        "accuracy": 0.9389,
        "precision": 0.7705,
        "recall": 0.9108,
        "f1_score": 0.8348,
        "roc_auc": 0.9616,
        "specificity": 0.9446
    }

@st.cache_data
def load_feature_importance():
    """Carga la importancia de las características"""
    return pd.DataFrame({
        "feature": [
            "stability_score", "behavior_stability_score", "age",
            "employment_quality_score", "compound_risk_score", "payment_punctuality",
            "debt_service_ratio", "education_stability_interaction", "income_stability",
            "financial_capacity_index", "credit_utilization", "overextension_risk",
            "family_vulnerability", "employment_stability", "work_experience_years"
        ],
        "importance": [
            0.330816, 0.243019, 0.151957, 0.049906, 0.030878, 0.017413,
            0.009953, 0.009640, 0.009523, 0.009262, 0.009189, 0.008837,
            0.007914, 0.007760, 0.007431
        ]
    })

# =================================================================
# FUNCIÓN DE FEATURE ENGINEERING
# =================================================================

def perform_feature_engineering(data):
    """
    Replica el feature engineering del notebook de entrenamiento
    
    Args:
        data: DataFrame con las variables base
        
    Returns:
        DataFrame con todas las variables ingeniadas
    """
    df = data.copy()
    
    # 1. Variables de interacción
    if all(col in df.columns for col in ['payment_punctuality', 'income_stability']):
        df["behavior_stability_score"] = (
            df["payment_punctuality"] * df["income_stability"]
        )
    
    if all(col in df.columns for col in ['debt_service_ratio', 'credit_utilization', 'employment_stability']):
        df["compound_risk_score"] = (
            df["debt_service_ratio"]
            * df["credit_utilization"]
            * (1 - df["employment_stability"])
        )
    
    # 2. Categorización de variables
    if 'age' in df.columns:
        df["age_group"] = pd.cut(
            df["age"],
            bins=[0, 25, 35, 45, 100],
            labels=[0, 1, 2, 3]  # Codificado como números
        ).astype(float)
    
    if 'monthly_income' in df.columns:
        # Usar quantiles aproximados del dataset original
        df["income_tier"] = pd.cut(
            df["monthly_income"],
            bins=[0, 2000, 3000, 4500, 100000],
            labels=[0, 1, 2, 3]  # Codificado como números
        ).astype(float)
    
    # 3. Scores compuestos
    if all(col in df.columns for col in ['savings_to_income_ratio', 'liquidity_indicator', 'debt_service_ratio']):
        df["solvency_score"] = (
            (df["savings_to_income_ratio"] * 0.3)
            + (df["liquidity_indicator"] * 0.3)
            + ((1 - df["debt_service_ratio"]) * 0.4)
        ) * 100
    
    if all(col in df.columns for col in ['employment_stability', 'work_experience_years']):
        df["employment_quality_score"] = (
            (df["employment_stability"] * 0.6)
            + (df["work_experience_years"] / 50 * 0.4)
        ) * 100
    
    # 4. Indicadores específicos de microfinanzas
    if all(col in df.columns for col in ['dependents', 'income_per_dependent']):
        df["family_vulnerability"] = df["dependents"] / (
            df["income_per_dependent"] / 1000 + 1
        )
    
    if all(col in df.columns for col in ['overextension_risk', 'savings_to_income_ratio']):
        df["financial_pressure"] = (
            df["overextension_risk"]
            * (1 / (df["savings_to_income_ratio"] + 0.1))
        )
    
    # 5. Transformaciones logarítmicas
    if 'monthly_income' in df.columns:
        df["monthly_income_log_transformed"] = np.log1p(df["monthly_income"])
    
    if 'savings_balance' in df.columns:
        df["savings_balance_log_transformed"] = np.log1p(df["savings_balance"])
    
    # 6. Transformación Box-Cox para credit_history_score
    if 'credit_history_score' in df.columns:
        data_positive = df['credit_history_score'] + 1
        # Usar lambda fijo del entrenamiento (aproximado)
        lambda_val = 0.5
        df["credit_history_score_boxcox"] = (data_positive**lambda_val - 1) / lambda_val
    
    return df

# =================================================================
# FUNCIONES DE PREDICCIÓN
# =================================================================

def prepare_input_data(input_dict):
    """
    Prepara los datos de entrada para el modelo
    
    Args:
        input_dict: Diccionario con los valores de entrada
        
    Returns:
        DataFrame listo para predicción
    """
    # Crear DataFrame con las variables base
    df = pd.DataFrame([input_dict])
    
    # Aplicar feature engineering
    df_engineered = perform_feature_engineering(df)
    
    # Asegurar el orden correcto de las columnas (40 features)
    expected_features = [
        'monthly_income', 'savings_balance', 'credit_history_score',
        'debt_service_ratio', 'payment_punctuality', 'credit_utilization',
        'income_stability', 'credit_inquiries_6m', 'age', 'education_level',
        'dependents', 'gender', 'employment_stability', 'work_experience_years',
        'sector_risk', 'savings_to_income_ratio', 'income_per_dependent',
        'financial_capacity_index', 'stability_score', 'wealth_index',
        'income_log', 'savings_log', 'credit_history_squared',
        'age_income_interaction', 'education_stability_interaction',
        'liquidity_indicator', 'overextension_risk', 'experience_maturity',
        'financial_maturity', 'behavior_stability_score', 'compound_risk_score',
        'age_group', 'income_tier', 'solvency_score', 'employment_quality_score',
        'family_vulnerability', 'financial_pressure',
        'monthly_income_log_transformed', 'savings_balance_log_transformed',
        'credit_history_score_boxcox'
    ]
    
    # Rellenar valores faltantes con 0 si es necesario
    for col in expected_features:
        if col not in df_engineered.columns:
            df_engineered[col] = 0
    
    # Seleccionar solo las columnas esperadas en el orden correcto
    df_final = df_engineered[expected_features]
    
    return df_final

def make_prediction(model, input_data):
    """
    Realiza la predicción usando el modelo
    
    Args:
        model: Modelo de Random Forest cargado
        input_data: DataFrame con los datos preparados
        
    Returns:
        tuple: (predicción, probabilidad_no_default, probabilidad_default)
    """
    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    
    return prediction, probabilities[0], probabilities[1]

def get_risk_category(prob_default):
    """
    Determina la categoría de riesgo basado en la probabilidad de default
    
    Args:
        prob_default: Probabilidad de default (0-1)
        
    Returns:
        tuple: (categoría, color, mensaje)
    """
    if prob_default < 0.3:
        return "BAJO", "#28a745", "Cliente de bajo riesgo - Aprobación recomendada"
    elif prob_default < 0.6:
        return "MEDIO", "#ffc107", "Cliente de riesgo moderado - Evaluación adicional recomendada"
    else:
        return "ALTO", "#dc3545", "Cliente de alto riesgo - Rechazar o revisar condiciones"

# =================================================================
# ESCENARIOS DE PRUEBA PRE-CONFIGURADOS
# =================================================================

def get_test_scenarios():
    """Define escenarios de prueba realistas"""
    return {
        "Cliente Ideal - Riesgo Bajo": {
            "monthly_income": 5000.0,
            "savings_balance": 8000.0,
            "credit_history_score": 0.95,
            "debt_service_ratio": 0.15,
            "payment_punctuality": 0.98,
            "credit_utilization": 0.20,
            "income_stability": 0.95,
            "credit_inquiries_6m": 1,
            "age": 35,
            "education_level": 3,
            "dependents": 2,
            "gender": 1,
            "employment_stability": 0.95,
            "work_experience_years": 10,
            "sector_risk": 1,
            "savings_to_income_ratio": 1.6,
            "income_per_dependent": 2500.0,
            "financial_capacity_index": 0.90,
            "stability_score": 0.92,
            "wealth_index": 0.85,
            "income_log": 8.517,
            "savings_log": 8.987,
            "credit_history_squared": 0.9025,
            "age_income_interaction": 175000.0,
            "education_stability_interaction": 2.85,
            "liquidity_indicator": 0.88,
            "overextension_risk": 0.15,
            "experience_maturity": 0.70,
            "financial_maturity": 0.85
        },
        "Cliente Promedio - Riesgo Medio": {
            "monthly_income": 3000.0,
            "savings_balance": 2500.0,
            "credit_history_score": 0.70,
            "debt_service_ratio": 0.40,
            "payment_punctuality": 0.75,
            "credit_utilization": 0.55,
            "income_stability": 0.70,
            "credit_inquiries_6m": 3,
            "age": 28,
            "education_level": 2,
            "dependents": 3,
            "gender": 0,
            "employment_stability": 0.65,
            "work_experience_years": 5,
            "sector_risk": 2,
            "savings_to_income_ratio": 0.83,
            "income_per_dependent": 1000.0,
            "financial_capacity_index": 0.60,
            "stability_score": 0.65,
            "wealth_index": 0.55,
            "income_log": 8.006,
            "savings_log": 7.824,
            "credit_history_squared": 0.49,
            "age_income_interaction": 84000.0,
            "education_stability_interaction": 1.30,
            "liquidity_indicator": 0.58,
            "overextension_risk": 0.42,
            "experience_maturity": 0.35,
            "financial_maturity": 0.55
        },
        "Cliente Riesgoso - Riesgo Alto": {
            "monthly_income": 1800.0,
            "savings_balance": 500.0,
            "credit_history_score": 0.35,
            "debt_service_ratio": 0.75,
            "payment_punctuality": 0.45,
            "credit_utilization": 0.90,
            "income_stability": 0.40,
            "credit_inquiries_6m": 8,
            "age": 23,
            "education_level": 1,
            "dependents": 4,
            "gender": 1,
            "employment_stability": 0.35,
            "work_experience_years": 2,
            "sector_risk": 3,
            "savings_to_income_ratio": 0.28,
            "income_per_dependent": 450.0,
            "financial_capacity_index": 0.25,
            "stability_score": 0.30,
            "wealth_index": 0.20,
            "income_log": 7.495,
            "savings_log": 6.215,
            "credit_history_squared": 0.1225,
            "age_income_interaction": 41400.0,
            "education_stability_interaction": 0.35,
            "liquidity_indicator": 0.22,
            "overextension_risk": 0.78,
            "experience_maturity": 0.15,
            "financial_maturity": 0.25
        }
    }

# =================================================================
# INTERFAZ PRINCIPAL
# =================================================================

def main():
    """Función principal de la aplicación"""
    
    # Header principal
    st.markdown(
        '<div class="main-header">Sistema de Credit Scoring - Microfinanzas Perú</div>',
        unsafe_allow_html=True
    )
    
    # Cargar modelo
    model = load_model()
    
    if model is None:
        st.error("No se pudo cargar el modelo. Verifica que el archivo pkl esté en la ruta correcta.")
        return
    
    # Sidebar para navegación
    st.sidebar.title("Navegación")
    page = st.sidebar.radio(
        "Selecciona una opción:",
        ["Inicio", "Predicción Individual", "Escenarios de Prueba", 
         "Métricas del Modelo", "Información"]
    )
    
    # =================================================================
    # PÁGINA: INICIO
    # =================================================================
    
    if page == "Inicio":
        st.header("Bienvenido al Sistema de Credit Scoring")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### Objetivo del Sistema
            
            Este sistema evalúa el riesgo crediticio de clientes potenciales en el 
            sector de microfinanzas en Perú utilizando un modelo de **Random Forest** 
            optimizado.
            
            ### Características Principales
            
            - **Alta precisión**: 93.9% de accuracy
            - **Excelente discriminación**: 96.2% ROC-AUC
            - **Interpretable**: Identificación de factores clave
            - **Validado**: Validación cruzada rigurosa
            
            ### Capacidades
            
            - Predicción de probabilidad de default
            - Clasificación de riesgo (Bajo/Medio/Alto)
            - Análisis de factores contribuyentes
            - Escenarios de prueba pre-configurados
            """)
        
        with col2:
            st.markdown("### Métricas del Modelo")
            metrics = load_model_metrics()
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("Accuracy", f"{metrics['accuracy']:.1%}")
                st.metric("Precision", f"{metrics['precision']:.1%}")
                st.metric("Recall", f"{metrics['recall']:.1%}")
            
            with col_m2:
                st.metric("F1-Score", f"{metrics['f1_score']:.1%}")
                st.metric("ROC-AUC", f"{metrics['roc_auc']:.1%}")
                st.metric("Especificidad", f"{metrics['specificity']:.1%}")
            
            st.markdown("""
            <div class="info-box">
            <strong>Proyecto de Tesis</strong><br>
            Desarrollo de un Modelo de Credit Scoring basado en Random Forest 
            para la Evaluación del Riesgo Crediticio en el Sector de 
            Microfinanzas en Perú
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.info("Usa el menú lateral para navegar entre las diferentes funcionalidades del sistema")
    
    # =================================================================
    # PÁGINA: PREDICCIÓN INDIVIDUAL
    # =================================================================
    
    elif page == "Predicción Individual":
        st.header("Predicción de Riesgo Crediticio Individual")
        st.markdown("Ingresa los datos del cliente para obtener una evaluación de riesgo crediticio.")
        
        with st.form("prediction_form"):
            st.subheader("Información del Cliente")
            
            # Crear tabs para organizar los inputs
            tab1, tab2, tab3, tab4 = st.tabs([
                "Información Financiera", 
                "Información Personal",
                "Historial Crediticio",
                "Indicadores Calculados"
            ])
            
            with tab1:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    monthly_income = st.number_input(
                        "Ingreso Mensual (S/.)", 
                        min_value=0.0, 
                        value=3000.0, 
                        step=100.0,
                        help="Ingreso mensual del cliente en soles"
                    )
                    savings_balance = st.number_input(
                        "Saldo de Ahorros (S/.)", 
                        min_value=0.0, 
                        value=2000.0, 
                        step=100.0,
                        help="Saldo actual en cuenta de ahorros"
                    )
                    debt_service_ratio = st.slider(
                        "Ratio de Servicio de Deuda",
                        0.0, 1.0, 0.35, 0.01,
                        help="Proporción de ingresos destinados al pago de deudas (0-1)"
                    )
                
                with col2:
                    credit_utilization = st.slider(
                        "Utilización de Crédito",
                        0.0, 1.0, 0.50, 0.01,
                        help="Proporción de crédito utilizado vs crédito disponible"
                    )
                    savings_to_income_ratio = st.number_input(
                        "Ratio Ahorro/Ingreso",
                        0.0, 10.0, 0.67, 0.01,
                        help="Saldo de ahorros dividido por ingreso mensual"
                    )
                    liquidity_indicator = st.slider(
                        "Indicador de Liquidez",
                        0.0, 1.0, 0.60, 0.01,
                        help="Capacidad de cumplir obligaciones de corto plazo"
                    )
                
                with col3:
                    overextension_risk = st.slider(
                        "Riesgo de Sobreendeudamiento",
                        0.0, 1.0, 0.35, 0.01,
                        help="Nivel de riesgo de estar sobreendeudado"
                    )
                    income_per_dependent = st.number_input(
                        "Ingreso por Dependiente (S/.)",
                        0.0, 50000.0, 1500.0, 50.0,
                        help="Ingreso mensual dividido por número de dependientes"
                    )
            
            with tab2:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    age = st.number_input(
                        "Edad",
                        18, 100, 30,
                        help="Edad del cliente en años"
                    )
                    gender = st.selectbox(
                        "Género",
                        options=[0, 1],
                        format_func=lambda x: "Femenino" if x == 0 else "Masculino"
                    )
                    dependents = st.number_input(
                        "Número de Dependientes",
                        0, 20, 2,
                        help="Número de personas que dependen económicamente del cliente"
                    )
                
                with col2:
                    education_level = st.selectbox(
                        "Nivel Educativo",
                        options=[1, 2, 3, 4],
                        format_func=lambda x: {
                            1: "Primaria", 
                            2: "Secundaria", 
                            3: "Superior Técnica", 
                            4: "Universitaria"
                        }[x],
                        index=1
                    )
                    employment_stability = st.slider(
                        "Estabilidad Laboral",
                        0.0, 1.0, 0.70, 0.01,
                        help="Nivel de estabilidad en el empleo actual (0-1)"
                    )
                    work_experience_years = st.number_input(
                        "Años de Experiencia Laboral",
                        0, 50, 5,
                        help="Años totales de experiencia laboral"
                    )
                
                with col3:
                    sector_risk = st.selectbox(
                        "Riesgo del Sector Económico",
                        options=[1, 2, 3, 4],
                        format_func=lambda x: {
                            1: "Bajo Riesgo", 
                            2: "Riesgo Moderado", 
                            3: "Riesgo Alto", 
                            4: "Riesgo Muy Alto"
                        }[x],
                        index=1
                    )
                    income_stability = st.slider(
                        "Estabilidad de Ingresos",
                        0.0, 1.0, 0.70, 0.01,
                        help="Nivel de consistencia en los ingresos mensuales"
                    )
            
            with tab3:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    credit_history_score = st.slider(
                        "Puntaje de Historial Crediticio",
                        0.0, 1.0, 0.70, 0.01,
                        help="Score de historial crediticio (0=malo, 1=excelente)"
                    )
                    payment_punctuality = st.slider(
                        "Puntualidad de Pagos",
                        0.0, 1.0, 0.75, 0.01,
                        help="Proporción de pagos realizados a tiempo"
                    )
                
                with col2:
                    credit_inquiries_6m = st.number_input(
                        "Consultas Crediticias (6 meses)",
                        0, 50, 3,
                        help="Número de consultas crediticias en los últimos 6 meses"
                    )
                    credit_history_squared = st.number_input(
                        "Credit History Squared",
                        0.0, 1.0, credit_history_score**2, 0.01,
                        help="Transformación cuadrática del historial crediticio",
                        disabled=True
                    )
                
                with col3:
                    income_log = st.number_input(
                        "Log(Ingreso)",
                        0.0, 15.0, np.log1p(monthly_income), 0.01,
                        help="Transformación logarítmica del ingreso",
                        disabled=True
                    )
                    savings_log = st.number_input(
                        "Log(Ahorros)",
                        0.0, 15.0, np.log1p(savings_balance), 0.01,
                        help="Transformación logarítmica de ahorros",
                        disabled=True
                    )
            
            with tab4:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    financial_capacity_index = st.slider(
                        "Índice de Capacidad Financiera",
                        0.0, 1.0, 0.65, 0.01,
                        help="Índice compuesto de capacidad financiera"
                    )
                    stability_score = st.slider(
                        "Score de Estabilidad",
                        0.0, 1.0, 0.68, 0.01,
                        help="Puntuación de estabilidad general del cliente"
                    )
                
                with col2:
                    wealth_index = st.slider(
                        "Índice de Riqueza",
                        0.0, 1.0, 0.55, 0.01,
                        help="Índice de patrimonio y riqueza"
                    )
                    experience_maturity = st.slider(
                        "Madurez de Experiencia",
                        0.0, 1.0, 0.40, 0.01,
                        help="Indicador de madurez basado en experiencia"
                    )
                
                with col3:
                    financial_maturity = st.slider(
                        "Madurez Financiera",
                        0.0, 1.0, 0.58, 0.01,
                        help="Nivel de madurez en manejo financiero"
                    )
                    age_income_interaction = st.number_input(
                        "Interacción Edad-Ingreso",
                        0.0, 1000000.0, age * monthly_income, 100.0,
                        help="Interacción entre edad e ingreso",
                        disabled=True
                    )
                    education_stability_interaction = st.number_input(
                        "Interacción Educación-Estabilidad",
                        0.0, 10.0, education_level * employment_stability, 0.01,
                        help="Interacción entre educación y estabilidad laboral",
                        disabled=True
                    )
            
            submitted = st.form_submit_button("Realizar Predicción", use_container_width=True)
            
            if submitted:
                # Preparar datos de entrada
                input_data = {
                    "monthly_income": monthly_income,
                    "savings_balance": savings_balance,
                    "credit_history_score": credit_history_score,
                    "debt_service_ratio": debt_service_ratio,
                    "payment_punctuality": payment_punctuality,
                    "credit_utilization": credit_utilization,
                    "income_stability": income_stability,
                    "credit_inquiries_6m": credit_inquiries_6m,
                    "age": age,
                    "education_level": education_level,
                    "dependents": dependents,
                    "gender": gender,
                    "employment_stability": employment_stability,
                    "work_experience_years": work_experience_years,
                    "sector_risk": sector_risk,
                    "savings_to_income_ratio": savings_to_income_ratio,
                    "income_per_dependent": income_per_dependent,
                    "financial_capacity_index": financial_capacity_index,
                    "stability_score": stability_score,
                    "wealth_index": wealth_index,
                    "income_log": income_log,
                    "savings_log": savings_log,
                    "credit_history_squared": credit_history_squared,
                    "age_income_interaction": age_income_interaction,
                    "education_stability_interaction": education_stability_interaction,
                    "liquidity_indicator": liquidity_indicator,
                    "overextension_risk": overextension_risk,
                    "experience_maturity": experience_maturity,
                    "financial_maturity": financial_maturity
                }
                
                # Preparar datos para el modelo
                prepared_data = prepare_input_data(input_data)
                
                # Realizar predicción
                prediction, prob_no_default, prob_default = make_prediction(model, prepared_data)
                
                # Obtener categoría de riesgo
                risk_category, _, risk_message = get_risk_category(prob_default)
                
                # Mostrar resultados
                st.markdown("---")
                st.subheader("Resultados de la Evaluación")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Probabilidad de NO Default",
                        f"{prob_no_default*100:.2f}%",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        "Probabilidad de Default",
                        f"{prob_default*100:.2f}%",
                        delta=None,
                        delta_color="inverse"
                    )
                
                with col3:
                    st.metric(
                        "Decisión del Modelo",
                        "NO DEFAULT" if prediction == 0 else "DEFAULT",
                        delta=None
                    )
                
                # Visualización de riesgo
                st.markdown("### Clasificación de Riesgo")
                
                if risk_category == "BAJO":
                    st.markdown(f'<div class="risk-low">🟢 RIESGO {risk_category}<br>{risk_message}</div>', 
                               unsafe_allow_html=True)
                elif risk_category == "MEDIO":
                    st.markdown(f'<div class="risk-medium">🟡 RIESGO {risk_category}<br>{risk_message}</div>', 
                               unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="risk-high">🔴 RIESGO {risk_category}<br>{risk_message}</div>', 
                               unsafe_allow_html=True)
                
                # Gráfico de probabilidades
                fig = go.Figure(data=[
                    go.Bar(
                        x=['No Default', 'Default'],
                        y=[prob_no_default*100, prob_default*100],
                        marker_color=['#28a745', '#dc3545'],
                        text=[f'{prob_no_default*100:.2f}%', f'{prob_default*100:.2f}%'],
                        textposition='auto',
                    )
                ])
                
                fig.update_layout(
                    title="Distribución de Probabilidades",
                    xaxis_title="Clase",
                    yaxis_title="Probabilidad (%)",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Factores principales
                st.markdown("### Factores Más Relevantes para Esta Evaluación")
                
                feature_importance = load_feature_importance()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Top 5 Factores Más Importantes:**")
                    for i, row in feature_importance.head(5).iterrows():
                        st.markdown(f"{i+1}. **{row['feature']}**: {row['importance']:.2%}")
                
                with col2:
                    # Gráfico de barras
                    fig_imp = px.bar(
                        feature_importance.head(10),
                        x='importance',
                        y='feature',
                        orientation='h',
                        title="Top 10 Variables Más Importantes"
                    )
                    fig_imp.update_layout(height=350)
                    st.plotly_chart(fig_imp, use_container_width=True)
    
    # =================================================================
    # PÁGINA: ESCENARIOS DE PRUEBA
    # =================================================================
    
    elif page == "Escenarios de Prueba":
        st.header("Escenarios de Prueba Pre-configurados")
        st.markdown("Selecciona un escenario de prueba para ver cómo funciona el modelo con diferentes perfiles de cliente.")
        
        scenarios = get_test_scenarios()
        
        scenario_name = st.selectbox(
            "Selecciona un escenario:",
            list(scenarios.keys())
        )
        
        if st.button("Ejecutar Escenario", use_container_width=True):
            scenario_data = scenarios[scenario_name]
            
            # Preparar datos
            prepared_data = prepare_input_data(scenario_data)
            
            # Realizar predicción
            prediction, prob_no_default, prob_default = make_prediction(model, prepared_data)
            
            # Obtener categoría de riesgo
            risk_category, _, risk_message = get_risk_category(prob_default)
            
            # Mostrar resultados
            st.markdown("---")
            st.subheader(f"Resultados: {scenario_name}")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Probabilidad No Default", f"{prob_no_default*100:.2f}%")
            
            with col2:
                st.metric("Probabilidad Default", f"{prob_default*100:.2f}%")
            
            with col3:
                st.metric("Decisión", "NO DEFAULT" if prediction == 0 else "DEFAULT")
            
            with col4:
                st.metric("Categoría Riesgo", risk_category)
            
            # Clasificación de riesgo
            if risk_category == "BAJO":
                st.markdown(f'<div class="risk-low">RIESGO {risk_category}<br>{risk_message}</div>', 
                           unsafe_allow_html=True)
            elif risk_category == "MEDIO":
                st.markdown(f'<div class="risk-medium">RIESGO {risk_category}<br>{risk_message}</div>', 
                           unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-high">RIESGO {risk_category}<br>{risk_message}</div>', 
                           unsafe_allow_html=True)
            
            # Mostrar datos del escenario
            st.markdown("### Datos del Cliente (Escenario)")
            
            scenario_df = pd.DataFrame([scenario_data]).T
            scenario_df.columns = ['Valor']
            scenario_df.index.name = 'Variable'
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.dataframe(scenario_df, height=400)
            
            with col2:
                # Gráfico comparativo
                fig = go.Figure()
                
                fig.add_trace(go.Indicator(
                    mode="gauge+number+delta",
                    value=prob_default*100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Probabilidad de Default (%)"},
                    delta={'reference': 50, 'increasing': {'color': "red"}},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 60], 'color': "yellow"},
                            {'range': [60, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        # Comparación de escenarios
        st.markdown("---")
        st.subheader("Comparación de Todos los Escenarios")
        
        if st.button("Comparar Todos los Escenarios", use_container_width=True):
            results = []
            
            for name, data in scenarios.items():
                prepared = prepare_input_data(data)
                pred, prob_no, prob_yes = make_prediction(model, prepared)
                risk, _, _ = get_risk_category(prob_yes)
                
                results.append({
                    'Escenario': name,
                    'Prob. No Default': f"{prob_no*100:.2f}%",
                    'Prob. Default': f"{prob_yes*100:.2f}%",
                    'Decisión': 'NO DEFAULT' if pred == 0 else 'DEFAULT',
                    'Riesgo': risk
                })
            
            results_df = pd.DataFrame(results)
            st.dataframe(results_df, use_container_width=True)
            
            # Gráfico comparativo
            prob_defaults = [float(r['Prob. Default'].strip('%')) for r in results]
            scenario_names = [r['Escenario'] for r in results]
            
            fig = go.Figure(data=[
                go.Bar(
                    x=scenario_names,
                    y=prob_defaults,
                    marker_color=['#28a745', '#ffc107', '#dc3545'],
                    text=[f"{p:.2f}%" for p in prob_defaults],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title="Comparación de Probabilidad de Default por Escenario",
                xaxis_title="Escenario",
                yaxis_title="Probabilidad de Default (%)",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    # =================================================================
    # PÁGINA: MÉTRICAS DEL MODELO
    # =================================================================
    
    elif page == "Métricas del Modelo":
        st.header("Métricas de Rendimiento del Modelo")
        st.markdown("Resultados obtenidos durante el entrenamiento y validación del modelo Random Forest.")
        
        metrics = load_model_metrics()
        
        # Métricas principales
        st.subheader("Métricas Principales (Conjunto de Prueba)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}", 
                     f"{metrics['accuracy']*100:.2f}%")
            st.metric("Precision", f"{metrics['precision']:.4f}",
                     f"{metrics['precision']*100:.2f}%")
        
        with col2:
            st.metric("Recall", f"{metrics['recall']:.4f}",
                     f"{metrics['recall']*100:.2f}%")
            st.metric("F1-Score", f"{metrics['f1_score']:.4f}",
                     f"{metrics['f1_score']*100:.2f}%")
        
        with col3:
            st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}",
                     f"{metrics['roc_auc']*100:.2f}%")
            st.metric("Especificidad", f"{metrics['specificity']:.4f}",
                     f"{metrics['specificity']*100:.2f}%")
        
        # Gráfico de métricas
        fig = go.Figure(data=[
            go.Bar(
                x=list(metrics.keys()),
                y=list(metrics.values()),
                marker_color='#1f77b4',
                text=[f"{v:.4f}" for v in metrics.values()],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title="Métricas de Rendimiento del Modelo",
            xaxis_title="Métrica",
            yaxis_title="Valor",
            height=400,
            yaxis_range=[0, 1]
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Validación cruzada
        st.subheader("Resultados de Validación Cruzada (10-Fold)")
        
        cv_metrics = {
            "Accuracy": "0.9385 ± 0.0014",
            "Precision": "0.7705 ± 0.0050",
            "Recall": "0.9074 ± 0.0047",
            "F1-Score": "0.8334 ± 0.0035",
            "ROC-AUC": "0.9618 ± 0.0015"
        }
        
        cv_df = pd.DataFrame(list(cv_metrics.items()), columns=['Métrica', 'Valor (Media ± Std)'])
        st.dataframe(cv_df, use_container_width=True)
        
        st.info("El modelo muestra alta estabilidad con coeficientes de variación < 0.01 en todas las métricas")
        
        # Cumplimiento de objetivos
        st.subheader("Cumplimiento de Objetivos de la Tesis")
        
        objectives = {
            "Accuracy > 90%": (metrics['accuracy'] > 0.90, metrics['accuracy']),
            "Precision > 85%": (metrics['precision'] > 0.85, metrics['precision']),
            "Recall > 80%": (metrics['recall'] > 0.80, metrics['recall']),
            "ROC-AUC > 95%": (metrics['roc_auc'] > 0.95, metrics['roc_auc'])
        }
        
        for objective, (achieved, value) in objectives.items():
            col1, col2, col3 = st.columns([2, 1, 1])
            with col1:
                st.markdown(f"**{objective}**")
            with col2:
                st.markdown(f"{value:.4f} ({value*100:.2f}%)")
            with col3:
                if achieved:
                    st.success("CUMPLIDO")
                else:
                    st.warning("PARCIAL")
        
        cumplidos = sum([a for a, _ in objectives.values()])
        total = len(objectives)
        
        st.markdown(f"### Resultado Final: {cumplidos}/{total} objetivos cumplidos ({cumplidos/total*100:.0f}%)")
        
        # Feature Importance
        st.subheader("Importancia de Variables (Top 15)")
        
        feature_importance = load_feature_importance()
        
        fig = px.bar(
            feature_importance,
            x='importance',
            y='feature',
            orientation='h',
            title="Variables Más Importantes del Modelo",
            labels={'importance': 'Importancia', 'feature': 'Variable'}
        )
        
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Interpretación
        st.markdown("""
        ### Interpretación de Variables Clave
        
        **1. Stability Score (33.08%)**: El factor más importante. Mide la estabilidad general del cliente 
        en términos financieros y laborales.
        
        **2. Behavior Stability Score (24.30%)**: Combina puntualidad de pagos y estabilidad de ingresos, 
        indicando el comportamiento financiero histórico.
        
        **3. Age (15.20%)**: La edad del cliente influye significativamente, reflejando madurez financiera 
        y experiencia.
        
        **4. Employment Quality Score (4.99%)**: Evalúa la calidad del empleo basándose en estabilidad 
        laboral y años de experiencia.
        
        **5. Compound Risk Score (3.09%)**: Indicador de riesgo compuesto que combina endeudamiento, 
        utilización de crédito y estabilidad laboral.
        """)
    
    # =================================================================
    # PÁGINA: INFORMACIÓN
    # =================================================================
    
    elif page == "Información":
        st.header("Información del Sistema")
        
        st.markdown("""
        ### Sobre el Proyecto
        
        Este sistema es parte de la tesis titulada **"Desarrollo de un Modelo de Credit Scoring 
        basado en Random Forest para la Evaluación del Riesgo Crediticio en el Sector de 
        Microfinanzas en Perú"**.
        
        ### Objetivos del Modelo
        
        El modelo tiene como objetivo:
        
        1. **Predecir el riesgo de default**: Evaluar la probabilidad de que un cliente no cumpla 
           con sus obligaciones crediticias.
        
        2. **Mejorar la toma de decisiones**: Proporcionar una herramienta objetiva y basada en 
           datos para la aprobación de créditos.
        
        3. **Aumentar la inclusión financiera**: Utilizar variables alternativas que permitan 
           evaluar clientes con historial crediticio limitado.
        
        4. **Reducir la morosidad**: Identificar con precisión clientes riesgosos para minimizar 
           pérdidas.
        
        ### Metodología
        
        El desarrollo del modelo siguió la metodología **CRISP-DM**:
        
        1. **Business Understanding**: Comprensión del problema de negocio en microfinanzas
        2. **Data Understanding**: Análisis exploratorio de 300,000 registros
        3. **Data Preparation**: Feature engineering con 40 variables predictoras
        4. **Modeling**: Optimización de Random Forest con GridSearchCV
        5. **Evaluation**: Validación rigurosa con múltiples métricas
        6. **Deployment**: Implementación de este sistema web
        
        ### Características del Dataset
        
        - **Tamaño**: 300,000 clientes
        - **Variables**: 30 originales + 11 ingeniadas = 41 features
        - **Target**: Default (0: No, 1: Sí)
        - **Balance**: 83.1% No Default, 16.9% Default
        - **División**: 60% entrenamiento, 20% validación, 20% prueba
        
        ### Hiperparámetros del Modelo
        
        El modelo Random Forest optimizado utiliza:
        
        ```
        - n_estimators: 100
        - max_depth: 30
        - min_samples_split: 10
        - min_samples_leaf: 2
        - max_features: 0.5
        - bootstrap: True
        - class_weight: None
        - random_state: 42
        ```
        
        ### Variables del Modelo (40 features)
        
        **Variables Financieras:**
        - Ingreso mensual, saldo de ahorros, ratio ahorro/ingreso
        - Ratio de servicio de deuda, utilización de crédito
        - Indicador de liquidez, riesgo de sobreendeudamiento
        
        **Variables de Historial Crediticio:**
        - Puntaje de historial crediticio, puntualidad de pagos
        - Consultas crediticias (6 meses), utilización de crédito
        
        **Variables Personales:**
        - Edad, género, nivel educativo, dependientes
        - Experiencia laboral, estabilidad laboral
        
        **Variables Ingeniadas:**
        - Scores de estabilidad y comportamiento
        - Indicadores de riesgo compuesto
        - Índices de madurez financiera
        - Transformaciones logarítmicas y Box-Cox
        
        ### Rendimiento del Modelo
        
        - **Accuracy**: 93.9% (Objetivo: >90%) - CUMPLIDO
        - **Precision**: 77.0% (Objetivo: >85%) - PARCIAL
        - **Recall**: 91.1% (Objetivo: >80%) - CUMPLIDO
        - **ROC-AUC**: 96.2% (Objetivo: >95%) - CUMPLIDO
        
        **Score Final: 3/4 objetivos cumplidos**
        
        ### Aplicaciones Prácticas
        
        Este sistema puede ser utilizado por:
        
        - **Instituciones microfinancieras**: Para evaluar solicitudes de crédito
        - **Analistas de riesgo**: Para identificar clientes de alto riesgo
        - **Gerentes de crédito**: Para tomar decisiones informadas
        - **Auditores**: Para validar procesos de aprobación crediticia
        
        ### Consideraciones Éticas
        
        El modelo debe usarse como:
        - Herramienta de apoyo a la decisión
        - Complemento al juicio humano experto
        - Sistema transparente e interpretable
        
        No debe usarse como:
        - Único criterio de decisión
        - Sistema discriminatorio
        - Caja negra sin supervisión
        
        ### Tecnologías Utilizadas
        
        - **Python 3.x**: Lenguaje de programación
        - **Scikit-learn**: Modelo Random Forest
        - **Streamlit**: Framework web
        - **Plotly**: Visualizaciones interactivas
        - **Pandas/NumPy**: Procesamiento de datos
        
        ### Información de Contacto
        
        Para más información sobre este proyecto de tesis, contacta al autor.
        
        ---
        
        **Versión del Sistema**: 1.0.0  
        **Fecha de Última Actualización**: Noviembre 2025  
        **Modelo Entrenado**: 03/12/2025 15:49:32
        """)
        
        st.info("Nota: Este sistema es un prototipo de investigación desarrollado con fines académicos.")

# =================================================================
# EJECUTAR APLICACIÓN
# =================================================================

if __name__ == "__main__":
    main()

