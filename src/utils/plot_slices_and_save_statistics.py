# Import packages with their abbreviations
import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from tqdm import tqdm
import pandas as pd
import os

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Load the TIFF file
tiff_file = os.getcwd() + "/01/tomogram_alignment_recon_FBP_14nm.tif"
image_array = tiff.imread(tiff_file)  # Shape: (431, 592, 592)

# Calculate global statistics across all slices
global_stats = {
    "mean": np.mean(image_array),
    "std": np.std(image_array),
    "min": np.min(image_array),
    "max": np.max(image_array),
    "25%": np.percentile(image_array, 25),
    "50% (median)": np.median(image_array),
    "75%": np.percentile(image_array, 75),
}
print("Global Statistics Across All Slices:")
for stat, value in global_stats.items():
    print(f"{stat}: {value}")

# Create a list to store slice data and per-slice statistics
slices_data = []

for i, slice_ in tqdm(enumerate(image_array), total=image_array.shape[0], desc="Processing slices"):
    stats = {
        "mean": np.mean(slice_),
        "std": np.std(slice_),
        "min": np.min(slice_),
        "max": np.max(slice_),
        "25%": np.percentile(slice_, 25),
        "50% (median)": np.median(slice_),
        "75%": np.percentile(slice_, 75),
    }
    slice_data = {
        "slice_index": i,
        "slice_array": slice_,
        "statistics": stats,
    }
    slices_data.append(slice_data)

# Save all slice statistics to a CSV file
def save_slices_to_csv():
    stats_list = [
        {"Slice": data["slice_index"], **data["statistics"]} for data in slices_data
    ]
    df = pd.DataFrame(stats_list)
    output_file = f"{tiff_file.split('.')[0]}_slice_statistics.csv"
    df.to_csv(output_file, index=False)
    print(f"All slice statistics saved to {output_file}")

# Interactive slider to display slices and their statistics
fig, ax = plt.subplots(figsize=(8, 8))  # Make the plot square for centering
plt.subplots_adjust(bottom=0.4)  # Adjust for slider and statistics space

# Display the first slice initially
current_slice = 0
img = ax.imshow(slices_data[current_slice]["slice_array"], cmap='gray')
ax.set_title(f"Slice {current_slice}")
plt.colorbar(img, ax=ax)

# Center the image
ax.set_aspect("equal")

# Format the statistics as a multi-line string
def format_statistics(statistics):
    formatted = "\n".join([f"{key}: {value:.2f}" for key, value in statistics.items()])
    return formatted

# Add statistics below the plot
stats_text = ax.text(
    0.5, -0.25,  # Position below the image
    f"Slice {current_slice} Statistics:\n{format_statistics(slices_data[current_slice]['statistics'])}",
    transform=ax.transAxes,
    fontsize=10,
    ha="center",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
)

# Create slider
ax_slider = plt.axes([0.2, 0.15, 0.6, 0.03])  # Position of the slider
slider = Slider(ax_slider, "Slice", 0, len(slices_data) - 1, valinit=current_slice, valstep=1)

# Update function for the slider
def update(val):
    slice_idx = int(slider.val)
    img.set_data(slices_data[slice_idx]["slice_array"])
    ax.set_title(f"Slice {slice_idx}")
    stats_text.set_text(
        f"Slice {slice_idx} Statistics:\n{format_statistics(slices_data[slice_idx]['statistics'])}"
    )
    fig.canvas.draw_idle()

# Connect the slider to the update function
slider.on_changed(update)

# Add a save button
ax_button = plt.axes([0.8, 0.05, 0.1, 0.04])  # Position of the button
button = Button(ax_button, "Save CSV")

# Connect the button to save function
button.on_clicked(lambda event: save_slices_to_csv())

plt.show()
