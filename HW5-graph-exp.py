# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the save path
save_path = r'C:\Users\wbalt\Desktop\Duke\ECE555-ProbStat\HW5'

# Ensure the save path exists
if not os.path.exists(save_path):
    os.makedirs(save_path)

# Function to simulate and analyze the system
def simulate_system(m_values, lam=1.0, alpha=1.0, beta=1.0, distribution='exponential', save_path=''):
    for m in m_values:
        print(f"\nSimulating for m = {m} ({distribution.capitalize()} Distribution)")
        # Generate random samples
        if distribution == 'exponential':
            # Exponential distribution
            U1 = np.random.uniform(0, 1, m)
            U2 = np.random.uniform(0, 1, m)
            U3 = np.random.uniform(0, 1, m)
            X1 = -np.log(U1) / lam
            X2 = -np.log(U2) / lam
            X3 = -np.log(U3) / lam
        elif distribution == 'weibull':
            # Weibull distribution
            U = np.random.uniform(0, 1, (m, 3))
            X = alpha * (-np.log(U)) ** (1 / beta)
            X1, X2, X3 = X[:, 0], X[:, 1], X[:, 2]
        else:
            raise ValueError("Invalid distribution type. Choose 'exponential' or 'weibull'.")

        # Stack the samples for easy computation
        samples = np.vstack((X1, X2, X3)).T  # Shape: (m, 3)

        # Compute y1j (minimum), y2j (second minimum), y3j (maximum)
        y1j = np.min(samples, axis=1)
        y3j = np.max(samples, axis=1)
        # To find the second minimum, we sort each row and take the second element
        sorted_samples = np.sort(samples, axis=1)
        y2j = sorted_samples[:, 1]

        # Plot histograms and theoretical PDFs
        plot_histograms(y1j, y2j, y3j, m, lam, alpha, beta, distribution, save_path)

        # Estimate and plot reliabilities
        estimate_reliabilities(y1j, y2j, y3j, m, lam, alpha, beta, distribution, save_path)

# Function to plot histograms and theoretical PDFs
def plot_histograms(y1j, y2j, y3j, m, lam, alpha, beta, distribution, save_path):
    import matplotlib.pyplot as plt

    # Define a range of t-values
    t_values = np.linspace(0, np.max(y3j), 1000)
    if distribution == 'exponential':
        # Exponential distribution PDFs
        # f_Y1(y)
        f_Y1 = 3 * lam * np.exp(-3 * lam * t_values)
        # f_Y2(y)
        f_Y2 = 6 * lam * np.exp(-2 * lam * t_values) * (1 - np.exp(-lam * t_values))
        # f_Y3(y)
        f_Y3 = 3 * (1 - np.exp(-lam * t_values)) ** 2 * lam * np.exp(-lam * t_values)
    elif distribution == 'weibull':
        # Weibull distribution PDFs
        # f_X(y)
        f_X = (beta / alpha) * (t_values / alpha) ** (beta - 1) * np.exp(-(t_values / alpha) ** beta)
        # F_X(y)
        F_X = 1 - np.exp(-(t_values / alpha) ** beta)
        # f_Y1(y)
        f_Y1 = 3 * f_X * (1 - F_X) ** 2
        # f_Y2(y)
        f_Y2 = 6 * f_X * F_X * (1 - F_X)
        # f_Y3(y)
        f_Y3 = 3 * f_X * F_X ** 2
    else:
        raise ValueError("Invalid distribution type. Choose 'exponential' or 'weibull'.")

    # Plot histograms with theoretical PDFs
    plt.figure(figsize=(15, 5))
    # Y1
    plt.subplot(1, 3, 1)
    plt.hist(y1j, bins=30, density=True, alpha=0.7, color='blue', label='Simulated Data')
    plt.plot(t_values, f_Y1, 'k-', label='Theoretical PDF')
    # Update the title to include the name
    plt.title(f'Y1 Min, m={m}, {distribution.capitalize()}')
    plt.xlabel('Time to Failure')
    plt.ylabel('Probability Density')
    plt.legend()

    # Y2
    plt.subplot(1, 3, 2)
    plt.hist(y2j, bins=30, density=True, alpha=0.7, color='green', label='Simulated Data')
    plt.plot(t_values, f_Y2, 'k-', label='Theoretical PDF')
    plt.title(f'Y2 Second Min, m={m}, {distribution.capitalize()}')
    plt.xlabel('Time to Failure')
    plt.legend()

    # Y3
    plt.subplot(1, 3, 3)
    plt.hist(y3j, bins=30, density=True, alpha=0.7, color='red', label='Simulated Data')
    plt.plot(t_values, f_Y3, 'k-', label='Theoretical PDF')
    plt.title(f'Y3 Max, m={m}, {distribution.capitalize()}')
    plt.xlabel('Time to Failure')
    plt.legend()

    plt.tight_layout()

    # Construct the filename
    if distribution == 'exponential':
        dist_name = ''
    else:
        dist_name = '_weibull'

    figure_name = f'y1y2_min_y3_max_{m}{dist_name}.png'
    # Save the figure
    save_file = os.path.join(save_path, figure_name)
    plt.savefig(save_file)
    print(f"Saved histogram figure: {save_file}")
    plt.close()

# Function to estimate and plot reliabilities
def estimate_reliabilities(y1j, y2j, y3j, m, lam, alpha, beta, distribution, save_path):
    import matplotlib.pyplot as plt

    # Define time values for which to estimate reliability
    t_values = np.linspace(0, np.max(y3j), 1000)
    # Initialize arrays to store estimated reliabilities
    R_series_est = np.zeros_like(t_values)
    R_TMR_est = np.zeros_like(t_values)
    R_parallel_est = np.zeros_like(t_values)

    # Estimate reliabilities
    for i, t in enumerate(t_values):
        # Series system (operational if y1j > t)
        R_series_est[i] = np.sum(y1j > t) / m
        # TMR system (operational if y2j > t)
        R_TMR_est[i] = np.sum(y2j > t) / m
        # Parallel system (operational if y3j > t)
        R_parallel_est[i] = np.sum(y3j > t) / m

    # Calculate theoretical reliabilities
    if distribution == 'exponential':
        R_series_theoretical = np.exp(-3 * lam * t_values)
        R_parallel_theoretical = 1 - (1 - np.exp(-lam * t_values)) ** 3  # Corrected line
        R_TMR_theoretical = np.exp(-3 * lam * t_values) + \
                            3 * np.exp(-2 * lam * t_values) * (1 - np.exp(-lam * t_values))
    elif distribution == 'weibull':
        R_series_theoretical = np.exp(-3 * (t_values / alpha) ** beta)
        R_parallel_theoretical = 1 - (1 - np.exp(-(t_values / alpha) ** beta)) ** 3  # Corrected line
        R_TMR_theoretical = R_series_theoretical + \
                            3 * np.exp(-2 * (t_values / alpha) ** beta) * \
                            (1 - np.exp(- (t_values / alpha) ** beta))
    else:
        raise ValueError("Invalid distribution type. Choose 'exponential' or 'weibull'.")

    # Plot estimated and theoretical reliabilities
    plt.figure(figsize=(10, 7))

    # Series system reliability
    plt.plot(t_values, R_series_est, 'b-', label='Estimated Series Reliability')
    plt.plot(t_values, R_series_theoretical, 'b--', label='Theoretical Series Reliability')

    # TMR system reliability
    plt.plot(t_values, R_TMR_est, 'g-', label='Estimated TMR Reliability')
    plt.plot(t_values, R_TMR_theoretical, 'g--', label='Theoretical TMR Reliability')

    # Parallel system reliability
    plt.plot(t_values, R_parallel_est, 'r-', label='Estimated Parallel Reliability')
    plt.plot(t_values, R_parallel_theoretical, 'r--', label='Theoretical Parallel Reliability')

    # Update the title to include the name
    plt.title(f'Estimated vs. Theoretical Reliabilities, m={m}, {distribution.capitalize()}')
    plt.xlabel('Time (t)')
    plt.ylabel('Reliability R(t)')
    plt.legend()
    plt.grid(True)

    # Construct the filename
    if distribution == 'exponential':
        dist_name = ''
    else:
        dist_name = '_weibull'

    figure_name = f'estimated_actual_{m}{dist_name}.png'
    # Save the figure
    save_file = os.path.join(save_path, figure_name)
    plt.savefig(save_file)
    print(f"Saved reliability figure: {save_file}")
    plt.close()

# Main code execution
if __name__ == "__main__":
    # Parameters
    lam = 1.0  # Rate parameter for exponential distribution
    alpha = 1.0  # Scale parameter for Weibull distribution
    beta = 2.0   # Shape parameter for Weibull distribution

    # Values of m to simulate
    m_values = [10, 100, 1000]

    # Simulate and analyze for Exponential distribution
    simulate_system(m_values, lam=lam, distribution='exponential', save_path=save_path)

    # Extra credit: Simulate and analyze for Weibull distribution
    simulate_system(m_values, alpha=alpha, beta=beta, distribution='weibull', save_path=save_path)
