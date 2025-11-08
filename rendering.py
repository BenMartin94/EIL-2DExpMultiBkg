import numpy as np
import pyvista as pv


def render_volume(
    volume: np.ndarray,
    output_path: str = "volume_render.png",
    cmap: str = "coolwarm",
    vmin: float = None,
    vmax: float = None,
    camera_position: tuple = (-50, 50, 175),
    focal_point: tuple = None,
    image_size: tuple = (800, 800),
    colormap: str = None,  # Alias for cmap
    alpha: list = None,  # Opacity values
    show_axes: bool = False,
    background: str = "white",
    zoom: float = 1.0,
) -> np.ndarray:
    """
    Render a 3D voxel volume using PyVista headless and save as PNG.
    
    Parameters:
    -----------
    volume : np.ndarray
        3D array representing the voxel data.
    output_path : str
        Path to save the rendered PNG image.
    cmap : str
        Colormap name (default: "coolwarm").
    vmin : float, optional
        Minimum value for colormap scaling. If None, uses data minimum.
    vmax : float, optional
        Maximum value for colormap scaling. If None, uses data maximum.
    camera_position : tuple, optional
        Camera position (x, y, z). If None, uses default view.
    focal_point : tuple, optional
        Point the camera looks at (x, y, z). If None, uses volume center.
    image_size : tuple
        Output image size (width, height). Default: (800, 800).
    colormap : str, optional
        Alias for cmap parameter (for backward compatibility).
    alpha : list, optional
        Opacity transfer function values (5 values from 0-1).
        If None, uses PyVista's default opacity.
    show_axes : bool
        Whether to show coordinate axes. Default: False.
    background : str
        Background color. Default: "white".
    zoom : float
        Camera zoom factor. Default: 1.0.
    
    Returns:
    --------
    np.ndarray
        Rendered image as numpy array.
    
    Example:
    --------
    >>> volume = np.random.rand(64, 64, 64)
    >>> render_volume(volume, "output.png", vmin=0.2, vmax=0.8)
    """
    # Handle colormap alias
    if colormap is not None:
        cmap = colormap
    
    # Create PyVista ImageData from numpy array
    grid = pv.ImageData(dimensions=volume.shape)
    grid[""] = volume.flatten(order="F")  # Empty string to avoid label on colorbar
    
    # Determine clim for colormap
    if vmin is None:
        vmin = volume.min()
    if vmax is None:
        vmax = volume.max()
    
    # Create plotter for headless rendering
    pl = pv.Plotter(off_screen=True, window_size=image_size)
    pl.set_background(background)
    
    # Add volume with optional opacity
    if alpha is not None:
        _ = pl.add_volume(grid, cmap=cmap, clim=[vmin, vmax], opacity=alpha, scalar_bar_args={'title': ''})
    else:
        _ = pl.add_volume(grid, cmap=cmap, clim=[vmin, vmax], scalar_bar_args={'title': ''})
    
    # Set camera position if provided
    if camera_position is not None:
        if focal_point is None:
            # Default to volume center
            focal_point = tuple(np.array(volume.shape) / 2.0)
        pl.camera_position = [camera_position, focal_point, (0, 0, 1)]
    
    # Apply zoom
    if zoom != 1.0:
        pl.camera.zoom(zoom)
    
    # Show axes if requested
    if show_axes:
        pl.show_axes()
    
    # Render and save to PNG
    pl.show(screenshot=output_path)
    
    print(f"Saved rendered volume to {output_path}")
    
    return pl.screenshot(output_path, return_img=True)


# --- Example usage ---
if __name__ == "__main__":
    # Create a simple sphere volume
    size = 64
    x, y, z = np.meshgrid(
        np.linspace(-1, 1, size),
        np.linspace(-1, 1, size),
        np.linspace(-1, 1, size),
        indexing='ij'
    )
    
    # Create a solid sphere with gradient
    distance = np.sqrt(x**2 + y**2 + z**2)
    volume = np.maximum(0, 1.0 - distance / 0.8)
    
    print(f"Volume stats: min={volume.min():.3f}, max={volume.max():.3f}, mean={volume.mean():.3f}")
    
    # Render using PyVista with all the extra arguments
    render_volume(
        volume=volume,
        output_path="sphere_render.png",
        camera_position=(10, 10, 150),
        focal_point=(size//2, size//2, size//2),
        image_size=(800, 800),
        colormap="turbo",
        #alpha=[0, 0.1, 0.3, 0.6, 1.0],
        show_axes=True,
        background="white",
        zoom=1
    )
    
    print("Sphere rendering complete!")
