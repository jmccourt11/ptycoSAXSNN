#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

# Constants
R = 20.0  # Radius of the spherical shell
D = 0.5  # Radius of the cylinder
shell_thickness = 2  # Thickness of the shell (increased for better visualization)
cylinder_height = R/2  # Example height

# Functions for lateral movement simulation



def calculate_lateral_overlap_volume(x_center, y_center, R_inner, R_outer, cylinder_radius, cylinder_height):
    """
    Calculate the volume of a spherical shell that overlaps with a cylinder
    when the sphere moves laterally (perpendicular to cylinder axis).
    
    Parameters:
    x_center, y_center: lateral position of sphere center
    R_inner: inner radius of spherical shell
    R_outer: outer radius of spherical shell  
    cylinder_radius: radius of cylinder
    cylinder_height: height of cylinder
    
    Returns:
    overlap_volume: volume of intersection
    """
    
    # Distance from sphere center to cylinder axis
    lateral_distance = np.sqrt(x_center**2 + y_center**2)
    
    # Check if sphere is too far from cylinder axis
    if lateral_distance > R_outer + cylinder_radius:
        return 0.0
    
    # Define cylinder bounds in z-direction
    z_min = -cylinder_height / 2
    z_max = cylinder_height / 2
    
    # Calculate effective z bounds for integration (sphere bounds)
    z_start = max(z_min, -R_outer)
    z_end = min(z_max, R_outer)
    
    # Numerical integration along z-axis
    n_points = 500
    z_values = np.linspace(z_start, z_end, n_points)
    dz = (z_end - z_start) / (n_points - 1) if n_points > 1 else 0
    
    total_volume = 0.0
    
    for z in z_values:
        # Check if this z-plane intersects the spherical shell
        if abs(z) <= R_outer:
            # Calculate radii of sphere at this z-plane
            r_outer_at_z = np.sqrt(R_outer**2 - z**2)
            
            if abs(z) <= R_inner:
                r_inner_at_z = np.sqrt(R_inner**2 - z**2)
    else:
        r_inner_at_z = 0
        
        # Calculate intersection area with cylinder at this z-plane
        # considering the lateral offset
        area_outer = circle_circle_intersection_area(r_outer_at_z, cylinder_radius, lateral_distance)
        area_inner = circle_circle_intersection_area(r_inner_at_z, cylinder_radius, lateral_distance)
        
        # Shell area at this z-plane
        shell_area = area_outer - area_inner
        
        total_volume += shell_area * dz
    
    return total_volume

def circle_circle_intersection_area(r1, r2, d):
    """
    Calculate the area of intersection between two circles.
    r1: radius of first circle (sphere cross-section)
    r2: radius of second circle (cylinder cross-section)
    d: distance between circle centers
    """
    if r1 <= 0 or r2 <= 0:
        return 0.0
    
    # No intersection if circles are too far apart
    if d >= r1 + r2:
        return 0.0
    
    # One circle completely inside the other
    if d <= abs(r1 - r2):
        return np.pi * min(r1, r2)**2
    
    # Partial intersection - use standard formula
    # Area = r1²⋅arccos((d²+r1²-r2²)/(2⋅d⋅r1)) + r2²⋅arccos((d²+r2²-r1²)/(2⋅d⋅r2)) - 0.5⋅√((-d+r1+r2)⋅(d+r1-r2)⋅(d-r1+r2)⋅(d+r1+r2))
    
    term1 = r1**2 * np.arccos((d**2 + r1**2 - r2**2) / (2 * d * r1))
    term2 = r2**2 * np.arccos((d**2 + r2**2 - r1**2) / (2 * d * r2))
    term3 = 0.5 * np.sqrt((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2))
    
    return term1 + term2 - term3

def calculate_lateral_overlap_wrapper(x_pos, R, shell_thickness, D, height):
    """
    Wrapper function for lateral movement (y_center = 0).
    """
    R_inner = R
    R_outer = R + shell_thickness
    
    return calculate_lateral_overlap_volume(x_pos, 0, R_inner, R_outer, D, height)



def plot_lateral_cross_section(x_pos, z_pos, R, shell_thickness, D, height):
    """
    Plot a 2D cross-section showing lateral movement.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot cylinder cross-section
    cylinder_circle = plt.Circle((0, 0), D, color='blue', alpha=0.5, label='Cylinder')
    ax.add_patch(cylinder_circle)
    
    # Check if sphere intersects this z-plane
    R_outer = R + shell_thickness
    if abs(z_pos) <= R_outer:
        # Calculate sphere radii at this z-position
        r_outer_at_z = np.sqrt(R_outer**2 - z_pos**2)
        
        # Plot outer sphere cross-section at lateral position
        outer_circle = plt.Circle((x_pos, 0), r_outer_at_z, fill=False, 
                                color='red', linewidth=3, alpha=0.8, label='Outer Sphere')
        ax.add_patch(outer_circle)
        
        # Plot inner sphere cross-section if it exists at this z
        if abs(z_pos) <= R:
            r_inner_at_z = np.sqrt(R**2 - z_pos**2)
            inner_circle = plt.Circle((x_pos, 0), r_inner_at_z, fill=False, 
                                    color='blue', linewidth=2, linestyle='--', alpha=0.9, label='Inner Sphere')
            ax.add_patch(inner_circle)
    
    # Set limits and aspect
    limit = max(R_outer + abs(x_pos), D) * 1.2
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect('equal', 'box')
    ax.set_title(f'Lateral Cross-section: x = {x_pos:.3f}, z = {z_pos:.3f}')
    ax.legend()
    ax.grid(True)
    
    plt.show()

def create_lateral_movement_animation(R, shell_thickness, D, height, n_frames=50):
    """
    Create an animation showing the spherical shell moving laterally through the cylinder.
    """
    # Set up the figure and axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    plt.tight_layout()
    
    # Animation parameters
    R_outer = R + shell_thickness
    x_range = R_outer + D + 1.0  # Range of lateral movement
    x_positions = np.linspace(-x_range, x_range, n_frames)
    
    # Calculate overlap volumes for all positions
    print(f"Calculating overlap volumes for {n_frames} positions...")
    overlap_volumes = [calculate_lateral_overlap_wrapper(x, R, shell_thickness, D, height) for x in x_positions]
    print("Volume calculations complete.")
    
    def animate(frame):
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        
        x_pos = x_positions[frame]
        current_volume = overlap_volumes[frame]
        
        # Top view (XY plane, z=0)
        ax1.set_xlim(-x_range * 1.2, x_range * 1.2)
        ax1.set_ylim(-max(R_outer, D) * 1.2, max(R_outer, D) * 1.2)
        ax1.set_aspect('equal')
        ax1.set_title(f'Top View (z=0) - Volume: {current_volume:.6f}')
        ax1.grid(True)
        
        # Add shell thickness info
        ax1.text(0.02, 0.98, f'Shell thickness: {shell_thickness:.3f}\nInner R: {R:.1f}, Outer R: {R_outer:.1f}', 
                transform=ax1.transAxes, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Plot cylinder cross-section
        cylinder_circle = plt.Circle((0, 0), D, color='blue', alpha=0.5, label='Cylinder')
        ax1.add_patch(cylinder_circle)
        
        # Plot sphere cross-sections at z=0
        # Make outer sphere semi-transparent so inner sphere is visible
        outer_circle = plt.Circle((x_pos, 0), R_outer, fill=False, 
                                color='red', linewidth=3, alpha=0.8, label='Outer Sphere')
        ax1.add_patch(outer_circle)
        
        # Make inner sphere more visible with different style
        inner_circle = plt.Circle((x_pos, 0), R, fill=False, 
                                color='blue', linewidth=2, linestyle='--', alpha=0.9, label='Inner Sphere')
        ax1.add_patch(inner_circle)
        
        ax1.legend(loc='upper right')
        
        # Side view (XZ plane, y=0)
        ax2.set_xlim(-x_range * 1.2, x_range * 1.2)
        ax2.set_ylim(-height/2 * 1.2, height/2 * 1.2)
        ax2.set_aspect('equal')
        ax2.set_title(f'Side View (y=0) - x = {x_pos:.3f}')
        ax2.grid(True)
        
        # Plot cylinder side view
        cylinder_rect = plt.Rectangle((-D, -height/2), 2*D, height, 
                                    color='blue', alpha=0.5, label='Cylinder')
        ax2.add_patch(cylinder_rect)
        
        # Plot sphere side view with consistent styling
        outer_circle_side = plt.Circle((x_pos, 0), R_outer, fill=False, 
                                     color='red', linewidth=3, alpha=0.8, label='Outer Sphere')
        ax2.add_patch(outer_circle_side)
        
        inner_circle_side = plt.Circle((x_pos, 0), R, fill=False, 
                                     color='blue', linewidth=2, linestyle='--', alpha=0.9, label='Inner Sphere')
        ax2.add_patch(inner_circle_side)
        
        ax2.legend(loc='upper right')
        
        # Add position indicator
        ax2.axvline(x=x_pos, color='red', linestyle='--', alpha=0.7)
        
        # Progress indicator
        if frame % 10 == 0:
            print(f"Animation frame {frame+1}/{n_frames}")
    
    # Create animation
    print("Creating animation object...")
    anim = FuncAnimation(fig, animate, frames=n_frames, interval=200, repeat=True, blit=False)
    
    return anim, overlap_volumes, x_positions

def create_static_frames(R, shell_thickness, D, height, n_frames=10):
    """
    Create static frames showing key positions of lateral movement.
    """
    R_outer = R + shell_thickness
    x_range = R_outer + D + 1.0
    x_positions = np.linspace(-x_range, x_range, n_frames)
    
    # Calculate overlap volumes
    overlap_volumes = [calculate_lateral_overlap_wrapper(x, R, shell_thickness, D, height) for x in x_positions]
    
    # Create subplots
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    fig.suptitle('Lateral Movement - Key Positions', fontsize=16)
    
    for i, (x_pos, volume) in enumerate(zip(x_positions, overlap_volumes)):
        row = i // 5
        col = i % 5
        ax = axes[row, col]
        
        # Plot cylinder
        cylinder_circle = plt.Circle((0, 0), D, color='blue', alpha=0.5)
        ax.add_patch(cylinder_circle)
        
        # Plot sphere with better visibility
        outer_circle = plt.Circle((x_pos, 0), R_outer, fill=False, 
                                color='red', linewidth=2, alpha=0.8)
        ax.add_patch(outer_circle)
        
        inner_circle = plt.Circle((x_pos, 0), R, fill=False, 
                                color='blue', linewidth=1.5, linestyle='--', alpha=0.9)
        ax.add_patch(inner_circle)
        
        # Set properties
        limit = max(R_outer + abs(x_pos), D) * 1.2
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_aspect('equal')
        ax.set_title(f'x={x_pos:.2f}\nV={volume:.4f}')
        ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    return x_positions, overlap_volumes

def plot_3d_lateral_movement(R, shell_thickness, D, height, x_position=0):
    """
    Create a 3D visualization of the cylinder and spherical shell for lateral movement.
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create cylinder
    theta = np.linspace(0, 2*np.pi, 50)
    z_cyl = np.linspace(-height/2, height/2, 50)
    Theta, Z = np.meshgrid(theta, z_cyl)
    X_cyl = D * np.cos(Theta)
    Y_cyl = D * np.sin(Theta)
    
    # Plot cylinder surface
    ax.plot_surface(X_cyl, Y_cyl, Z, alpha=0.3, color='blue', label='Cylinder')
    
    # Create spherical shell
    phi = np.linspace(0, np.pi, 30)
    theta_sphere = np.linspace(0, 2*np.pi, 50)
    Phi, Theta_sphere = np.meshgrid(phi, theta_sphere)
    
    # Outer sphere at lateral position
    R_outer = R + shell_thickness
    X_outer = R_outer * np.sin(Phi) * np.cos(Theta_sphere) + x_position
    Y_outer = R_outer * np.sin(Phi) * np.sin(Theta_sphere)
    Z_outer = R_outer * np.cos(Phi)
    
    # Inner sphere at lateral position
    X_inner = R * np.sin(Phi) * np.cos(Theta_sphere) + x_position
    Y_inner = R * np.sin(Phi) * np.sin(Theta_sphere)
    Z_inner = R * np.cos(Phi)
    
    # Plot spherical shell
    ax.plot_surface(X_outer, Y_outer, Z_outer, alpha=0.3, color='red', label='Outer Sphere')
    ax.plot_surface(X_inner, Y_inner, Z_inner, alpha=0.2, color='pink', label='Inner Sphere')
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'3D View: Lateral Movement (x_pos = {x_position:.2f})')
    
    # Set equal aspect ratio
    max_range = max(R_outer + abs(x_position), height/2, D) * 1.2
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    plt.show()

# Simulation parameters for LATERAL movement
print("Starting LATERAL movement simulation...")
print(f"Cylinder: radius = {D:.3f}, height = {cylinder_height:.3f}")
print(f"Spherical shell: inner radius = {R:.3f}, outer radius = {R + shell_thickness:.3f}")
print(f"Shell thickness = {shell_thickness:.3f}")

# Define lateral movement range
R_outer = R + shell_thickness
x_range = R_outer + D + 0.5  # Range of lateral movement
x_positions = np.linspace(-x_range, x_range, 200)  # Lateral positions

print(f"Lateral movement range: {-x_range:.3f} to {x_range:.3f}")

# Calculate overlap volumes for lateral movement
print("Calculating overlap volumes for lateral movement...")
lateral_overlap_volumes = [calculate_lateral_overlap_wrapper(x, R, shell_thickness, D, cylinder_height) for x in x_positions]

# Plot lateral movement results
fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(x_positions, lateral_overlap_volumes, 'g-', linewidth=3, label='Lateral Overlap Volume')
ax.set_xlabel('Lateral Position (x)')
ax.set_ylabel('Overlap Volume')
ax.set_title('Volume Overlap vs Lateral Position (Perpendicular to Cylinder Axis)')
ax.legend()
ax.grid(True)

# Mark maximum overlap position
max_lateral_idx = np.argmax(lateral_overlap_volumes)
max_lateral_volume = lateral_overlap_volumes[max_lateral_idx]
max_lateral_position = x_positions[max_lateral_idx]

ax.axvline(x=max_lateral_position, color='red', linestyle='--', alpha=0.7, 
           label=f'Max overlap at x={max_lateral_position:.3f}')
ax.legend()

plt.tight_layout()
plt.show()

print(f"\nMaximum lateral overlap volume: {max_lateral_volume:.6f}")
print(f"Position of maximum lateral overlap: x = {max_lateral_position:.3f}")

# Create 3D visualization at maximum lateral overlap
print("\nGenerating 3D visualization at maximum overlap...")
plot_3d_lateral_movement(R, shell_thickness, D, cylinder_height, max_lateral_position)

# Create and display animation
print("\nCreating animation of lateral movement...")
print("Note: Animation may take a moment to calculate...")

try:
    anim, anim_volumes, anim_positions = create_lateral_movement_animation(
        R, shell_thickness, D, cylinder_height, n_frames=30)
    
    # Try different approaches to display/save animation
    animation_saved = False
    animation_displayed = False
    
    # First try to save as GIF
    try:
        print("Saving animation as 'lateral_movement.gif'...")
        anim.save('lateral_movement.gif', writer='pillow', fps=5)
        print("Animation saved as GIF successfully!")
        animation_saved = True
    except Exception as save_e:
        print(f"Failed to save as GIF: {save_e}")
        
        # Try saving as MP4 instead
        try:
            print("Trying to save as MP4...")
            anim.save('lateral_movement.mp4', writer='ffmpeg', fps=5)
            print("Animation saved as MP4 successfully!")
            animation_saved = True
        except Exception as mp4_e:
            print(f"Failed to save as MP4: {mp4_e}")
    
    # Try to display the animation
    try:
        # For Jupyter notebooks
        from IPython.display import HTML, display
        print("Attempting to display animation in Jupyter...")
        display(HTML(anim.to_jshtml()))
        animation_displayed = True
    except ImportError:
        # Not in Jupyter, try regular matplotlib display
        try:
            print("Attempting to display animation with matplotlib...")
            plt.show()
            animation_displayed = True
        except Exception as display_e:
            print(f"Failed to display animation: {display_e}")
    
    if animation_saved or animation_displayed:
        print("Animation created successfully!")
        print("The animation shows:")
        print("- Left panel: Top view (XY plane at z=0)")
        print("- Right panel: Side view (XZ plane at y=0)")
        print("- The sphere moves laterally (left-right) through the cylinder")
        if animation_saved:
            print("- Check the saved animation file in your directory")
    else:
        raise Exception("Could not save or display animation")
    
except Exception as e:
    print(f"Animation creation failed: {e}")
    print("Error details:", str(e))
    print("Showing static views instead...")
    
    # Show static frames instead
    print("Creating static frames...")
    create_static_frames(R, shell_thickness, D, cylinder_height, n_frames=10)

# Show key cross-sections
print("\nGenerating key cross-sectional views...")
plot_lateral_cross_section(max_lateral_position, 0, R, shell_thickness, D, cylinder_height)
plot_lateral_cross_section(0, 0, R, shell_thickness, D, cylinder_height)



print("\nSimulation complete!")

# %%
