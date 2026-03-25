
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


st.set_page_config(
    page_title="DBil Trajectory Prediction Model",
    layout="wide"
)


n_groups = 5  
param_matrix = np.array([
    [174.8771,  -77.7399,  9.4778],   # Group 1
    [178.2627,  -74.1496,  15.4246],  # Group 2
    [149.7439,  -11.6684,  -4.2360],  # Group 3
    [310.8862,  -234.4296,  55.2103], # Group 4
    [159.157,    53.316,   -18.028]   # Group 5
])

# Define trajectory prediction function
def predict_gbtm_trajectory(time_points, group_params):
    intercept, linear, quadratic = group_params
    predictions = (
        intercept + 
        linear * time_points + 
        quadratic * (time_points ** 2)
    )
    return predictions

def calculate_group_probabilities(tba_values, time_points, param_matrix):
    n_groups = param_matrix.shape[0]
    
    # Calculate distance between predicted and actual values for each group
    distances = []
    for g in range(n_groups):
        params = param_matrix[g]
        predicted = predict_gbtm_trajectory(time_points, params)
        
        # Calculate mean squared error
        mse = np.mean((np.array(tba_values) - predicted) ** 2)
        distances.append(mse)
    
    # Convert distance to probability
    distances = np.array(distances)
    if np.min(distances) == 0:
        probabilities = np.zeros(n_groups)
        probabilities[np.argmin(distances)] = 1.0
    else:
        weights = 1 / (distances + 1e-10)  # Add small value to avoid division by zero
        probabilities = weights / np.sum(weights)
    
    return probabilities

# Streamlit user interface
st.title("BA-NLS Predictor: An Online Tool for Prognostic Stratification Based on Postoperative DBil Trajectories in Biliary Atresia")

# ====================== Key Modification 1: Globally unified time point definition ======================
# Time point configuration (globally unified)
time_labels = ["Baseline (Preoperative)", "2 Weeks Postoperative", "1 Month Postoperative", "3 Months Postoperative"]  # Unified medical terminology
time_points_original = np.array([1, 2, 3, 4])  # Time points for modeling (1-4, consistent with R language)
time_points_smooth = np.linspace(1, 4, 100)    # Smooth time points for plotting
n_time_points = len(time_labels)

# Sidebar - Input parameters
st.sidebar.header("Input of Patient's DBil Measurement Data")
st.sidebar.subheader("Please enter DBil values at each time point (μmol/L)")

# Create input boxes (using unified time_labels)
tba_values = []
for i, label in enumerate(time_labels):
    value = st.sidebar.number_input(
        label,
        min_value=0.0,
        max_value=900.0,
        value=float(i * 20 + 10),
        step=1.0,
        format="%.1f",
        key=f"tba_{i}"
    )
    tba_values.append(value)

# Main interface
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("DBil Trajectory Visualization")
    # Create canvas
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot [Patient's actual DBil trajectory] - 4 discrete points + blue line
    ax.plot(time_points_original, tba_values, 'bo-', linewidth=2, markersize=8, label='Patient\'s Actual DBil Values', zorder=5)

    # Plot [5 groups of GBTM background curves]
    colors = ['#1F77B4', '#2CA02C', '#9467BD', '#E377C2', '#FF7F0E']  # R language color scheme
    all_predictions = []

    for g in range(n_groups):
        params = param_matrix[g]
        predicted_smooth = predict_gbtm_trajectory(time_points_smooth, params)
        all_predictions.append(predicted_smooth)
        ax.plot(time_points_smooth, predicted_smooth,
                color=colors[g],
                linestyle='--',
                linewidth=1.5,
                alpha=0.7,
                label=f'Trajectory {g+1}')

    # Axis configuration (using unified time_labels)
    ax.set_xlabel('Follow-up Time', fontsize=12)
    ax.set_ylabel('DBil Value (μmol/L)', fontsize=12)
    ax.set_title('DBil Trajectory Comparison (GBTM 5-Group Model)', fontsize=14)
    ax.set_xticks(time_points_original)
    ax.set_xticklabels(time_labels, rotation=0)  # Use unified medical terminology
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

with col2:
    st.subheader("Prediction Results")
    
    # Calculate probability of each group
    if st.sidebar.button("Start Prediction"):
        with st.spinner("Calculating..."):
            # ====================== Modification 2: Pass correct time point parameters ======================
            probabilities = calculate_group_probabilities(tba_values, time_points_original, param_matrix)
            
            # Find the most likely group
            most_likely_group = np.argmax(probabilities) + 1
            
            st.write(f"**Prediction Completed!**")
            st.write(f"**Most Likely Trajectory Group:** Group {most_likely_group}")
            
            # Display probability of all groups
            st.subheader("Probability of Each Trajectory Group")
            for g in range(n_groups):
                prob_percent = probabilities[g] * 100
                st.write(f"**Trajectory {g+1}**: {prob_percent:.1f}%") 
            
            # Clinical recommendations
            st.subheader("Clinical Recommendations")
            advice_dict = {
                1: "Trajectory 1 (Low Baseline Decreasing Group): A standard follow-up protocol may be maintained, with focused monitoring of liver function and growth parameters.",
                2: "Trajectory 2 (Low Baseline Sharp Increasing Group): Intensified management is indicated with increased follow-up frequency. The timing of liver transplantation should be continuously evaluated during follow-up, with proactive preparation for subsequent liver transplantation assessment when necessary.",
                3: "Trajectory 3 (Low Baseline Increasing Group): Follow-up intervals should be appropriately shortened and the frequency of endoscopic surveillance enhanced, so as to enable the early detection and management of portal hypertension-related complications.",
                4: "Trajectory 4 (High Baseline Decreasing Group): A standard follow-up protocol may be maintained, with focused monitoring of liver function and growth parameters.",
                5: "Trajectory 5 (High Baseline Increasing Group): Close follow-up must be implemented with increased endoscopic surveillance for the proactive mitigation of risks such as variceal bleeding. For this subgroup, priority should be given to the monitoring of liver function, and evaluation for liver transplantation should be promptly initiated in the event of liver function deterioration."
            }
            # Display main recommendations
            if most_likely_group in advice_dict:
                st.write(advice_dict[most_likely_group])
            else:
                st.write("Specific recommendations cannot be provided; please make judgments based on actual clinical conditions.")

# Add data summary
st.subheader("Data Summary")
# Create data summary table (using unified time_labels)
summary_data = {
    "Time Point": time_labels,
    "DBil Value (μmol/L)": tba_values
}
summary_df = pd.DataFrame(summary_data)
st.dataframe(summary_df, use_container_width=True)

# Run check
import sys
if "streamlit" not in sys.modules:
    st.warning("Please run this application with the command 'streamlit run app.py'")
    st.stop()
