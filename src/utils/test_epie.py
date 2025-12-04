import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2


def create_concentric_circle_object(size, inner_radius, outer_radius):
    """Create a concentric double circle as the object."""
    y, x = np.ogrid[:size, :size]
    center = size // 2
    distance = np.sqrt((x - center) ** 2 + (y - center) ** 2)
    object_ = np.zeros((size, size), dtype=complex)
    object_[(distance >= inner_radius) & (distance <= outer_radius)] = 1.0 + 1j * 0.5
    return object_


def epie_reconstruction_with_new_plots(
    diffraction_patterns, scan_positions, probe, object_guess, true_probe, true_object, beta=0.9, max_iter=50
):
    """
    EPIE Algorithm with New Plots Every 10 Iterations
    
    Parameters:
        diffraction_patterns (numpy array): Measured diffraction patterns (Nx, Ny, Npos)
        scan_positions (list of tuples): List of (x, y) positions for each scan
        probe (numpy array): Initial guess for the probe function
        object_guess (numpy array): Initial guess for the object
        true_probe (numpy array): True probe function (for visualization)
        true_object (numpy array): True object function (for visualization)
        beta (float): Feedback parameter (0 < beta <= 1)
        max_iter (int): Maximum number of iterations
    """
    # Get dimensions
    Ny, Nx, Npos = diffraction_patterns.shape
    object_reconstruction = object_guess
    probe_reconstruction = probe

    # Iterate
    for iteration in range(max_iter):
        for i, (x, y) in enumerate(scan_positions):
            # Extract the object patch corresponding to the probe
            obj_patch = object_reconstruction[y:y+Ny, x:x+Nx]

            # Forward propagation (Fourier transform)
            exit_wave = probe_reconstruction * obj_patch
            diffracted_wave = fft2(exit_wave)

            # Apply amplitude constraint
            measured_amplitudes = np.sqrt(diffraction_patterns[:, :, i])
            updated_wave = measured_amplitudes * np.exp(1j * np.angle(diffracted_wave))

            # Inverse propagation (Inverse Fourier transform)
            corrected_exit_wave = ifft2(updated_wave)

            # Calculate updates for object and probe
            update = beta * (corrected_exit_wave - exit_wave)
            object_update = update * np.conj(probe_reconstruction) / np.max(np.abs(probe_reconstruction)**2)
            probe_update = update * np.conj(obj_patch) / np.max(np.abs(obj_patch)**2)

            # Update object and probe
            object_reconstruction[y:y+Ny, x:x+Nx] += object_update
            probe_reconstruction += probe_update

        # Normalize the probe
        probe_reconstruction /= np.linalg.norm(probe_reconstruction)

        # Create new plots every 10 iterations
        if iteration % 10 == 0 or iteration == max_iter - 1:
            plt.close("all")
            fig, axs = plt.subplots(2, 4, figsize=(16, 8))
            titles = [
                "True Object", "Reconstructed Object",
                "True Probe (Magnitude)", "Reconstructed Probe (Magnitude)",
                "True Probe (Phase)", "Reconstructed Probe (Phase)"
            ]
            for ax, title in zip(axs.flatten(), titles):
                ax.set_title(title)
                ax.axis("off")
            axs[0, 0].imshow(np.abs(true_object), cmap="gray", vmin=0, vmax=1)
            axs[0, 1].imshow(np.abs(object_reconstruction), cmap="gray", vmin=0, vmax=1)
            axs[1, 0].imshow(np.abs(true_probe), cmap="gray")
            axs[1, 1].imshow(np.abs(probe_reconstruction), cmap="gray")
            axs[1, 2].imshow(np.angle(true_probe), cmap="twilight")
            axs[1, 3].imshow(np.angle(probe_reconstruction), cmap="twilight")
            plt.tight_layout()
            plt.show()

        print(f"Iteration {iteration+1}/{max_iter} complete.")

    return object_reconstruction, probe_reconstruction


# Example usage
Nx, Ny = 128, 128  # Patch size
Npos = 100  # Number of scan positions
object_size = 512  # Size of the object
probe_size = (Ny, Nx)

# Create true object as a concentric double circle
true_object = create_concentric_circle_object(object_size, inner_radius=100, outer_radius=150)

# Simulated true probe
true_probe = np.random.rand(*probe_size) + 1j * np.random.rand(*probe_size)

# Simulated data (replace with actual diffraction patterns and positions)
diffraction_patterns = np.abs(np.random.rand(Ny, Nx, Npos))**2
scan_positions = [(np.random.randint(0, object_size - Nx), np.random.randint(0, object_size - Ny)) for _ in range(Npos)]
object_guess = np.ones((object_size, object_size), dtype=complex)
probe = np.ones(probe_size, dtype=complex)

reconstructed_object, reconstructed_probe = epie_reconstruction_with_new_plots(
    diffraction_patterns, scan_positions, probe, object_guess, true_probe, true_object, beta=0.9, max_iter=50
)

