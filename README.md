# 🚁 AgriDrone - AI-Powered Crop Disease Monitoring System

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red.svg)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


# Group Members

| S. No. | Name             | Roll No.        |
|--------:|------------------|-----------------|
| 1 | Aakash Kumar      | 2K23/CSME/01 |
| 2 | Pawan Kumar       | 2K23/CSME/39 |
| 3 | M. Haneef Memon   | 2K23/CSME/29 |
---
## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Features](#-features)
- [Technology Stack](#-technology-stack)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [How It Works](#-how-it-works)
- [Algorithms Used](#-algorithms-used)
- [Usage Guide](#-usage-guide)
- [API Configuration](#-api-configuration)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🌱 Project Overview

**AgriDrone** is an intelligent agricultural monitoring system that simulates a drone scanning a 25×25 farm grid to detect and monitor crop diseases. The system combines:

- 🤖 **Machine Learning** (Random Forest) for disease classification
- 🗺️ **Path Planning Algorithms** (Boustrophedon + A*) for complete field coverage
- 🧬 **Cellular Automata** for realistic disease spread simulation
- 🧠 **AI Integration** (OpenRouter/Gemini) for intelligent agronomist reports
- 📊 **Interactive Visualizations** for real-time data analysis

This project was developed as a comprehensive AI application for agricultural monitoring, demonstrating the integration of multiple AI techniques in a user-friendly web interface.

---

## ✨ Features

### Core Functionality
| Feature | Description |
|---------|-------------|
| 🗺️ **Coverage Path Planning** | Boustrophedon (lawnmower) pattern with A* obstacle avoidance |
| 🔬 **Disease Detection** | Random Forest classifier identifies Healthy, Early, or Severe disease |
| 🧬 **Disease Spread Simulation** | Cellular automaton with configurable spread steps (0-10) |
| 📊 **Interactive Heatmap** | Plotly-based 25×25 grid with hover details and zoom |
| 📈 **Analytics Dashboard** | Pie charts, bar charts, and detailed metrics |
| 🤖 **AI Field Reports** | LLM-generated agronomist advice and spray recommendations |
| 📤 **Data Export** | CSV and JSON download capabilities |

### Field Configurations
- **Flat Farm** - Open field with minimal obstacles
- **Pond Farm** - Contains water obstacles requiring detours
- **Dense Field** - High-density disease distribution for stress testing

---

## 🛠️ Technology Stack

### Framework & Libraries
```python
├── Frontend Framework: Streamlit 1.32.0
├── Visualization: Plotly 5.19.0
├── Machine Learning: scikit-learn 1.4.0
├── Data Processing: NumPy 1.26.0, Pandas 2.2.0
├── AI Integration: OpenRouter API / Gemini API
├── Model Persistence: Joblib 1.3.0
└── HTTP Client: Requests 2.31.0

## 🚀 Installation & Setup

### 1. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Train the Machine Learning Model

```bash
python disease_model.py
```

### 3. Run the Application

**Using Python**

```bash
python -m streamlit run app.py
```

**Or (if Streamlit is added to PATH)**

```bash
streamlit run app.py
```

---

# 🔧 How It Works

## System Architecture

<p align="center">
  <img src="https://github.com/user-attachments/assets/d589c101-dda8-419d-b832-fb99ebf5029e"
       alt="System Architecture"
       width="1000">
</p>

## Screenshots

### 1. Home Screen
![Home Screen](screenshots/1_home.jpg)

### 2. After Drone Scan
![Scan Result](screenshots/2_scan_result.jpg)

### 3. AI Field Report
![AI Report](screenshots/3_AI_report.jpg)
