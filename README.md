# Mental_Health_Detector_Webapp
# 🧠 Mental Health Detector

Screen your mental well-being using AI. This app predicts the likelihood of **depression** based on lifestyle, academic, and personal indicators using machine learning.

---

## Features

* 🧠 AI-powered mental health assessment form
* 📊 Depression likelihood score with explanation
* 🎯 Top 5 contributing risk factors visualized
* 📝 Personalized recommendations based on input
* 📈 Dataset insights and depression statistics
* 🆘 Mental health resource links and helplines

---

## Installation

Follow these steps to get the Mental Health Detector up and running on your local machine:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/RajdeepSutradhar5105/Mental_Health_Detector_Webapp.git
    cd Mental_Health_Detector
    ```

2. **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    > Includes: `streamlit`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `plotly`

---

## How to Run

1. **Start the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

2. **Open in your browser:**

    The app will open automatically at `http://localhost:8501`.

---

## Usage

1. Navigate through form sections:
   - 👤 Personal Information
   - 📚 Academic & Work Pressure
   - 🏃‍♂️ Lifestyle Habits
   - ⚠️ Risk Assessment

2. Fill in the assessment form honestly.

3. Click **"🧠 Analyze Mental Health Status"** to view:
   - Depression likelihood score
   - Pie chart of results
   - Key insights and recommendations

4. Scroll down for educational resources and emergency contact info.

---
 ## Video Presentation Of Project

https://github.com/user-attachments/assets/b7e67fe6-2906-402d-b60c-9924e44b8469

---
## Workflow
```mermaid
flowchart TD
    A[Start: User Opens App] --> B[Page Configuration & Styling]
    B --> C[Load Dataset from CSV]
    C --> D[Preprocess Data with Label Encoding]
    D --> E[Train Random Forest Model]
    E --> F[Display App Header and Info Box]

    F --> G[Show Assessment Form]
    
    G --> G1[Personal Information]
    G --> G2[Academic & Work Pressure]
    G --> G3[Lifestyle Factors]
    G --> G4[Risk Assessment]

    G --> H[Show Quick Stats & Feature Importance]

    H --> I[User Clicks 'Analyze Mental Health Status']

    I --> J[Encode Inputs from User]
    J --> K[Predict Depression Risk]
    K --> L{Prediction Result?}
    
    L -->|High Risk| M1[Show Warning Box with Probability]
    L -->|Low Risk| M2[Show Success Box with Probability]
    
    M1 --> N[Show Risk Breakdown Pie Chart]
    M2 --> N[Show Risk Breakdown Pie Chart]
    
    N --> O[Generate Personalized Insights]
    O --> P[Display Mental Health Resources]
    P --> Q[Footer with Credits]
    Q --> R[End]
```


