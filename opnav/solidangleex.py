import matplotlib.pyplot as plt
import numpy as np

# Create a 3D hemisphere and a detector area to illustrate solid angle and irradiance

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

# Sphere parameters
r = 1
u = np.linspace(0, 2 * np.pi, 100)
v = np.linspace(0, np.pi / 2, 100)
x = r * np.outer(np.cos(u), np.sin(v))
y = r * np.outer(np.sin(u), np.sin(v))
z = r * np.outer(np.ones(np.size(u)), np.cos(v))

# Plot the hemisphere (upper half)
ax.plot_surface(x, y, z, color='skyblue', alpha=0.3, edgecolor='none')

# Mark a solid angle patch on the hemisphere
theta_patch = np.linspace(np.pi/6, np.pi / 3, 30)
phi_patch = np.linspace(0, np.pi / 6, 30)
theta_patch, phi_patch = np.meshgrid(theta_patch, phi_patch)
x_patch = r * np.sin(theta_patch) * np.cos(phi_patch)
y_patch = r * np.sin(theta_patch) * np.sin(phi_patch)
z_patch = r * np.cos(theta_patch)
ax.plot_surface(x_patch, y_patch, z_patch, color='orange', alpha=0.6, edgecolor='none')

# Detector plane (flat surface)
detector_size = 0.5
x_det = np.array([[-detector_size, detector_size], [-detector_size, detector_size]])
y_det = np.array([[-detector_size, -detector_size], [detector_size, detector_size]])
z_det = np.array([[0, 0], [0, 0]])
ax.plot_surface(x_det, y_det, z_det, color='green', alpha=0.5)

# Labels and view
ax.set_title("Solid Angle and Irradiance", fontsize=14)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.view_init(elev=30, azim=135)
ax.text(0.1, 0.1, 1.1, "Hemisphere", color='skyblue')
ax.text(-0.3, -0.3, 0.05, "Detector", color='green')
ax.text(0.5, 0.2, 0.7, "Solid Angle (Ω)", color='orange')

plt.tight_layout()
plt.show()
