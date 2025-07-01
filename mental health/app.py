import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

# Enhanced page config
st.set_page_config(
    page_title="Mental Health Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling with WHITE TEXT for info boxes
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        text-align: center;
        color: #333;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #000 !important;
    }
    
    .warning-box h3, .warning-box h2, .warning-box p, .warning-box strong {
        color: #000 !important;
    }
    
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #000 !important;
    }
    
    .success-box h3, .success-box h2, .success-box p, .success-box strong {
        color: #000 !important;
    }
    
    .info-box {
        background-color: #e3f2fd;
        border: 1px solid #bbdefb;
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
        color: #000 !important;
    }
    
    .info-box h4, .info-box p, .info-box strong {
        color: #000 !important;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .sidebar .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 10px;
    }
    
    .sidebar .stSlider > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
    
    /* WHITE TEXT for Streamlit info/alert boxes */
    .stAlert > div {
        color: #fff !important;
    }
    
    .stAlert > div > div {
        color: #fff !important;
    }
    
    .stAlert p {
        color: #fff !important;
    }
    
    .stAlert div[data-testid="stMarkdownContainer"] {
        color: #fff !important;
    }
    
    .stAlert div[data-testid="stMarkdownContainer"] p {
        color: #fff !important;
    }
    
    .stSuccess > div {
        color: #fff !important;
    }
    
    .stSuccess > div > div {
        color: #fff !important;
    }
    
    .stSuccess p {
        color: #fff !important;
    }
    
    .stSuccess div[data-testid="stMarkdownContainer"] {
        color: #fff !important;
    }
    
    .stSuccess div[data-testid="stMarkdownContainer"] p {
        color: #fff !important;
    }
    
    .stInfo > div {
        color: #fff !important;
    }
    
    .stInfo > div > div {
        color: #fff !important;
    }
    
    .stInfo p {
        color: #fff !important;
    }
    
    .stInfo div[data-testid="stMarkdownContainer"] {
        color: #fff !important;
    }
    
    .stInfo div[data-testid="stMarkdownContainer"] p {
        color: #fff !important;
    }
    
    .stWarning > div {
        color: #fff !important;
    }
    
    .stWarning > div > div {
        color: #fff !important;
    }
    
    .stWarning p {
        color: #fff !important;
    }
    
    .stWarning div[data-testid="stMarkdownContainer"] {
        color: #fff !important;
    }
    
    .stWarning div[data-testid="stMarkdownContainer"] p {
        color: #fff !important;
    }
    
    /* Additional targeting for nested elements */
    [data-testid="stAlert"] {
        color: #fff !important;
    }
    
    [data-testid="stAlert"] p {
        color: #fff !important;
    }
    
    [data-testid="stAlert"] div {
        color: #fff !important;
    }
    
    /* Force white text on all alert components */
    div[data-baseweb="notification"] {
        color: #fff !important;
    }
    
    div[data-baseweb="notification"] p {
        color: #fff !important;
    }
    
    div[data-baseweb="notification"] div {
        color: #fff !important;
    }
</style>
""", unsafe_allow_html=True)

# Load and preprocess dataset
@st.cache_data
def load_data():
    df = pd.read_csv("student_depression_dataset.csv")
    df = df.drop(columns=["id", "City", "Profession"])
    
    label_encoders = {}
    for col in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
    return df, label_encoders

df, encoders = load_data()

# Train model
@st.cache_resource
def train_model(df):
    X = df.drop("Depression", axis=1)
    y = df["Depression"]
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    return model

model = train_model(df)

# Header
st.markdown('<h1 class="main-header">🧠 Mental Health Detector</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Mental Health Assessment Tool</p>', unsafe_allow_html=True)

# Info box with BLACK TEXT
st.markdown("""
<div class="info-box">
    <h4 style="color: #000 !important;">📋 About This Tool</h4>
    <p style="color: #000 !important;">This application uses machine learning to assess depression likelihood based on lifestyle and personal factors. 
    Please answer all questions honestly for the most accurate assessment.</p>
    <p style="color: #000 !important;"><strong style="color: #000 !important;">⚠️ Disclaimer:</strong> This is a screening tool, not a medical diagnosis. Please consult healthcare professionals for proper evaluation.</p>
</div>
""", unsafe_allow_html=True)

# Create two columns for better layout
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Assessment Form")
    
    # Personal Information Section
    with st.expander("👤 Personal Information", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            gender = st.selectbox("Gender", encoders['Gender'].classes_, help="Select your gender")
            age = st.slider("Age", 15, 60, 25, help="Your current age")
        with col_b:
            degree = st.selectbox("Education Level", encoders['Degree'].classes_, help="Your highest degree")
            cgpa = st.slider("CGPA/GPA", 0.0, 10.0, 7.0, step=0.1, help="Your academic performance")

    # Academic & Work Pressure Section
    with st.expander("📚 Academic & Work Pressure", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            academic_pressure = st.slider("Academic Pressure", 0, 5, 3, help="Rate your academic stress level")
            study_satisfaction = st.slider("Study Satisfaction", 0, 5, 3, help="How satisfied are you with your studies?")
        with col_b:
            work_pressure = st.slider("Work Pressure", 0, 5, 2, help="Rate your work-related stress")
            job_satisfaction = st.slider("Job Satisfaction", 0, 5, 3, help="How satisfied are you with your job?")
        
        work_hours = st.slider("Work/Study Hours per day", 0, 16, 6, help="Average hours spent on work/study daily")

    # Lifestyle Section
    with st.expander("🏃‍♂️ Lifestyle Factors", expanded=True):
        col_a, col_b = st.columns(2)
        with col_a:
            sleep_duration = st.selectbox("Sleep Duration", encoders['Sleep Duration'].classes_, 
                                        help="How many hours do you sleep per night?")
        with col_b:
            dietary = st.selectbox("Dietary Habits", encoders['Dietary Habits'].classes_,
                                 help="Rate your overall dietary habits")

    # Risk Factors Section
    with st.expander("⚠️ Risk Assessment", expanded=True):
        financial_stress = st.selectbox("Financial Stress Level", encoders['Financial Stress'].classes_,
                                      help="Rate your financial stress (1=Low, 5=High)")
        
        col_a, col_b = st.columns(2)
        with col_a:
            suicidal_thoughts = st.selectbox("Ever had Suicidal Thoughts?", 
                                           encoders['Have you ever had suicidal thoughts ?'].classes_,
                                           help="Have you ever experienced suicidal thoughts?")
        with col_b:
            family_history = st.selectbox("Family History of Mental Illness", 
                                        encoders['Family History of Mental Illness'].classes_,
                                        help="Any family history of mental health issues?")

with col2:
    st.markdown("### 📊 Quick Stats")
    
    # Display some quick stats
    total_responses = len(df)
    depression_rate = (df['Depression'].sum() / len(df)) * 100
    
    st.markdown(f"""
    <div class="metric-card">
        <h3>{total_responses:,}</h3>
        <p>Total Assessments</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="metric-card">
        <h3>{depression_rate:.1f}%</h3>
        <p>Depression Rate in Dataset</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature importance chart
    st.markdown("### 🎯 Key Risk Factors")
    feature_importance = model.feature_importances_
    feature_names = [col for col in df.columns if col != 'Depression']
    
    # Create a simple bar chart
    fig_importance = go.Figure(data=[
        go.Bar(x=feature_importance[:5], 
               y=feature_names[:5],
               orientation='h',
               marker_color='rgba(102, 126, 234, 0.8)')
    ])
    fig_importance.update_layout(
        title="Top 5 Important Factors",
        height=300,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    st.plotly_chart(fig_importance, use_container_width=True)

# Prediction Section
st.markdown("---")
st.markdown("### 🔮 Get Your Assessment")

# Encode inputs
input_dict = {
    "Gender": encoders['Gender'].transform([gender])[0],
    "Age": age,
    "Academic Pressure": academic_pressure,
    "Work Pressure": work_pressure,
    "CGPA": cgpa,
    "Study Satisfaction": study_satisfaction,
    "Job Satisfaction": job_satisfaction,
    "Sleep Duration": encoders['Sleep Duration'].transform([sleep_duration])[0],
    "Dietary Habits": encoders['Dietary Habits'].transform([dietary])[0],
    "Degree": encoders['Degree'].transform([degree])[0],
    "Have you ever had suicidal thoughts ?": encoders['Have you ever had suicidal thoughts ?'].transform([suicidal_thoughts])[0],
    "Work/Study Hours": work_hours,
    "Financial Stress": encoders['Financial Stress'].transform([financial_stress])[0],
    "Family History of Mental Illness": encoders['Family History of Mental Illness'].transform([family_history])[0]
}

input_df = pd.DataFrame([input_dict])

# Predict button with better styling
col_center = st.columns([1, 2, 1])[1]
with col_center:
    if st.button("🧠 Analyze Mental Health Status", use_container_width=True):
        with st.spinner("Analyzing your responses..."):
            prediction = model.predict(input_df)[0]
            proba = model.predict_proba(input_df)[0]
            
            # Results section with BLACK TEXT
            st.markdown("---")
            st.markdown("### 📋 Your Assessment Results")
            
            col_result1, col_result2 = st.columns(2)
            
            with col_result1:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="warning-box">
                        <h3 style="color: #000 !important;">⚠️ High Risk Detected</h3>
                        <h2 style="color: #d63031 !important;">{proba[1]*100:.1f}%</h2>
                        <p style="color: #000 !important;"><strong style="color: #000 !important;">Likelihood of Depression</strong></p>
                        <p style="color: #000 !important;">Your responses indicate a higher risk for depression. We strongly recommend consulting with a mental health professional.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="success-box">
                        <h3 style="color: #000 !important;">✅ Lower Risk</h3>
                        <h2 style="color: #00b894 !important;">{proba[0]*100:.1f}%</h2>
                        <p style="color: #000 !important;"><strong style="color: #000 !important;">No Depression Indicated</strong></p>
                        <p style="color: #000 !important;">Your responses suggest a lower risk for depression. Continue maintaining good mental health practices!</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col_result2:
                # Enhanced pie chart with Plotly
                fig = go.Figure(data=[go.Pie(
                    labels=['No Depression', 'Depression'],
                    values=[proba[0]*100, proba[1]*100],
                    hole=.3,
                    marker_colors=['#00b894', '#d63031']
                )])
                
                fig.update_traces(
                    textposition='inside', 
                    textinfo='percent+label',
                    textfont_size=12
                )
                
                fig.update_layout(
                    title="Risk Assessment Breakdown",
                    showlegend=True,
                    height=300,
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Additional insights with WHITE TEXT
            st.markdown("### 💡 Personalized Insights")
            
            insights = []
            if academic_pressure >= 4:
                insights.append("📚 Consider stress management techniques for academic pressure")
            if work_pressure >= 4:
                insights.append("💼 Work-life balance might need attention")
            if study_satisfaction <= 2:
                insights.append("🎯 Low study satisfaction may be affecting your well-being")
            if work_hours >= 12:
                insights.append("⏰ Consider reducing work hours for better mental health")
            if sleep_duration in ["Less than 5 hours", "More than 10 hours"]:
                insights.append("😴 Improving sleep schedule could benefit your mental health")
            
            if insights:
                for insight in insights:
                    st.info(insight)  # This will now have WHITE text
            else:
                st.success("🌟 Your lifestyle factors look well-balanced!")  # This will also have WHITE text

# Resources section
st.markdown("---")
st.markdown("### 🆘 Mental Health Resources")

col_res1, col_res2, col_res3 = st.columns(3)

with col_res1:
    st.markdown("""
    **🏥 Emergency Help**
    - National Suicide Prevention Lifeline: 988
    - Crisis Text Line: Text HOME to 741741
    - Emergency Services: 911
    """)

with col_res2:
    st.markdown("""
    **💬 Online Support**
    - BetterHelp: Online therapy
    - 7 Cups: Free emotional support
    - NAMI: Mental health information
    """)

with col_res3:
    st.markdown("""
    **📱 Mental Health Apps**
    - Headspace: Meditation
    - Calm: Sleep & relaxation
    - Mood Meter: Emotion tracking
    """)

# Footer
st.markdown("""
<div class="footer">
    <p>Developed with ❤️ using Streamlit | Mental Health Awareness Project</p>
    <p><small>This tool is for educational purposes only and should not replace professional medical advice.</small></p>
</div>
""", unsafe_allow_html=True)