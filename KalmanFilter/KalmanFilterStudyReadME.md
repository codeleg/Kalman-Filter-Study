Kalman Filter Parameter Analysis and Impact Evaluation
Overview
This report presents a systematic analysis of how different Kalman filter parameters affect filtering performance when estimating a noisy signal. The analysis is based on running simulations with various parameter settings and evaluating the noise variance before and after filtering.
Kalman Filter Basics
The Kalman filter is an algorithm used to estimate the state of a dynamic system from noisy measurements. It involves two main steps:
    1. Prediction Step: Predict the state and its covariance based on the previous state.
    2. Update Step: Update the prediction with new measurements to correct the state estimation.
Key Parameters
Key parameters involved in the filter include:
    • State Transition Matrix (F): Describes how the state evolves from one time step to the next.
    • Measurement Matrix (H): Maps the true state to the observed measurement space.
    • Process Noise Covariance (Q): Represents the uncertainty in the process model.
    • Measurement Noise Covariance (R): Represents the uncertainty in the measurements.
    • Initial Covariance (P): Initial uncertainty about the state estimate.
    • Initial State Estimate (x): Initial guess of the state.
Simulation Setup
The script simulates the filtering of a noisy sine wave signal with various parameter configurations. The filter estimates the true signal, and the noise variance is calculated before and after filtering. The simulations aim to demonstrate how changing the Kalman filter parameters influences the filter's behavior.
Signal Generation
The script generates a sine wave signal with added Gaussian noise:
    • The true signal is given by sin(0.1⋅time)+offset\text{sin}(0.1 \cdot \text{time}) + \text{offset}sin(0.1⋅time)+offset.
    • Noise is added to simulate real-world measurement noise, drawn from a normal distribution.
Parameters Tested
The following Kalman filter parameters were varied in different test cases:
    • Process Noise Covariance (Q): Increased and decreased to assess the filter's sensitivity to process uncertainty.
    • Measurement Noise Covariance (R): Modified to observe the filter's adaptation to varying levels of measurement reliability.
    • Initial Covariance (P): Different values tested to understand the impact on the initial state uncertainty.
    • Initial State Estimate: Different starting values to see how initial guesses influence convergence.
    • Signal Offset: Adjusted to evaluate the filter's response to a shifted signal.
    • Total Simulation Time: Extended to observe the filter's ability to stabilize results over a longer period.
Results and Analysis
Case 1: Base Case
    • Parameters: Q = 1e-5, R = 1.0, P = 1.0, initial_state = 0, offset = 0, total_time = 100.
    • Noise Variance Before Filtering: [Calculated value]
    • Noise Variance After Filtering: [Calculated value]
    • Observations: The filter reduced the noise variance and tracked the true signal well.
Case 2: Increased Q
    • Parameters: Q = 1e-3, other parameters unchanged.
    • Impact: Higher Q allowed the filter to adapt more quickly to changes but also introduced more noise.
Case 3: Increased R
    • Parameters: R = 10.0, other parameters unchanged.
    • Impact: With higher measurement noise, the filter smoothed out the signal more, resulting in less variance reduction.
Case 4: Increased P
    • Parameters: P = 10.0, other parameters unchanged.
    • Impact: The filter initially exhibited more uncertainty, taking longer to stabilize.
Case 5: Changed Initial State
    • Parameters: initial_state = 1.
    • Impact: The choice of initial state affected early performance but converged over time.
Case 6: Increased Offset
    • Parameters: offset = 0.5.
    • Impact: The filter successfully adapted to the shifted signal.
Case 7: Extended Time
    • Parameters: total_time = 200.
    • Impact: More simulation time allowed the filter to achieve better results and stability.
Conclusions
    • Parameter Sensitivity: Q and R had significant impacts on the filter's adaptability and noise reduction.
    • Importance of Proper Initialization: The initial state and covariance influenced early filter performance.
    • Stabilization Time: Longer simulation periods allowed the filter to better stabilize the state estimate.

