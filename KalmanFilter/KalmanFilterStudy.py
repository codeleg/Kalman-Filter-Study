import numpy as np
import matplotlib.pyplot as plt
import os

class KalmanFilter:
    def __init__(self, F, H, Q, R, P, x):
        self.F = F  # State transition matrix
        self.H = H  # Measurement matrix
        self.Q = Q  # Process noise covariance
        self.R = R  # Measurement noise covariance
        self.P = P  # Initial estimation error covariance
        self.x = x  # Initial state

    def predict(self):
        # Predict the state and error covariance
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(self.F, np.dot(self.P, self.F.T)) + self.Q
        return self.x

    def update(self, z):
        # Calculate Kalman Gain
        K = np.dot(self.P, self.H.T) / (np.dot(self.H, np.dot(self.P, self.H.T)) + self.R)
        
        # Update the estimate via measurement z
        self.x = self.x + K * (z - np.dot(self.H, self.x))
        
        # Update the error covariance
        self.P = (np.eye(len(self.P)) - K * self.H) @ self.P
        return self.x

def simulate_kalman_filter(F, H, Q, R, P, x, frequency, amplitude, offset, sampling_interval, total_time, noise_variance, title_suffix=""):
    # Create Kalman filter instance
    kf = KalmanFilter(F, H, Q, R, P, x)

    # === Signal Generation ===
    time_steps = np.arange(0, total_time, sampling_interval)
    true_signal = offset + amplitude * np.sin(2 * np.pi * frequency * time_steps)
    noise_std_dev = np.sqrt(noise_variance)
    noisy_signal = [val + np.random.normal(0, noise_std_dev) for val in true_signal]

    # === Apply Kalman Filter ===
    kalman_estimates = []
    for measurement in noisy_signal:
        kf.predict()
        estimate = kf.update(measurement)
        kalman_estimates.append(estimate[0][0])

    # === Calculate Variance Before and After Filtering ===
    noise_variance_before = np.var(noisy_signal - true_signal)
    noise_variance_after = np.var(kalman_estimates - true_signal)

    # === Plot the Results ===
    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, noisy_signal, label='Noisy Signal', color='orange', linestyle='-', alpha=0.6)
    plt.plot(time_steps, true_signal, label='True Signal (Sine Wave)', linestyle='--', color='blue')
    plt.plot(time_steps, kalman_estimates, label='Kalman Filter Estimate', color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Value')
    plt.title(f'Kalman Filter Applied to a Noisy Sinusoidal Wave {title_suffix}\nNoise Variance Before: {noise_variance_before:.2f}, After: {noise_variance_after:.2f}')
    plt.legend()
    plt.grid()
    
    # Save the plot
    filename = f"KalmanFilter_{title_suffix.replace(' ', '_')}.png"
    plt.savefig(filename)
    plt.show()

    # Return results for reporting
    return noise_variance_before, noise_variance_after

def run_analysis():
    # === Initial Parameters ===
    frequency = 1  # Hz
    amplitude = 5  # Amplitude of sine wave
    offset = 10  # Offset of sine wave
    sampling_interval = 0.001  # Sampling interval in seconds
    total_time = 1  # Total duration in seconds
    noise_variance = 16  # Variance of noise

    # Kalman Filter Parameters
    F = np.array([[1]])  # State transition matrix
    H = np.array([[1]])  # Measurement matrix
    Q = np.array([[1]])  # Process noise covariance
    R = np.array([[10]])  # Measurement noise covariance
    P = np.array([[1]])  # Initial estimation error covariance
    x = np.array([[0]])  # Initial state estimate

    # === Run the Kalman Filter Simulation for Initial Settings ===
    initial_results = simulate_kalman_filter(F, H, Q, R, P, x, frequency, amplitude, offset, sampling_interval, total_time, noise_variance, "Initial Settings")

    # === Analysis of Different Parameter Configurations ===
    parameter_variations = [
        ("High Process Noise", np.array([[10]]), R, P, x),
        ("Low Process Noise", np.array([[0.1]]), R, P, x),
        ("High Measurement Noise", Q, np.array([[50]]), P, x),
        ("Low Measurement Noise", Q, np.array([[1]]), P, x),
        ("High Initial Uncertainty", Q, R, np.array([[10]]), x),
        ("Low Initial Uncertainty", Q, R, np.array([[0.1]]), x),
        ("Different Initial State", Q, R, P, np.array([[5]])),
        ("High Offset", Q, R, P, x),  # Changing offset
        ("Longer Duration", Q, R, P, x)  # Changing total time
    ]

    # Store results
    results_summary = []
    for description, Q_var, R_var, P_var, x_var in parameter_variations:
        if description == "High Offset":
            offset_var = 15  # Changing offset
            total_time_var = 1
        else:
            offset_var = 10
            total_time_var = 1
        
        if description == "Longer Duration":
            total_time_var = 2  # Changing total time

        print(f"\n=== {description} ===")
        noise_before, noise_after = simulate_kalman_filter(F, H, Q_var, R_var, P_var, x_var, frequency, amplitude, offset_var, sampling_interval, total_time_var, noise_variance, description)
        results_summary.append(f"{description}: Noise Variance Before: {noise_before:.2f}, After: {noise_after:.2f}")

    # Save results summary to a text file
    with open("Kalman_Filter_Results_Summary.txt", "w") as file:
        file.write("\n".join(results_summary))

    print("\nAnalysis complete. Results saved in 'Kalman_Filter_Results_Summary.txt'.")

# === Run the Analysis ===
run_analysis()
