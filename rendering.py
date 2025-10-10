import numpy as np
from typing import Tuple, Optional, Union, List
from vedo import Volume, Plotter, settings, addons


# Configure vedo for offscreen rendering
settings.default_backend = "vtk"
settings.use_depth_peeling = True


def render_volume(
    volume: np.ndarray,
    output_path: str = "volume_render.png",
    camera_position: Optional[Union[Tuple, List, np.ndarray]] = None,
    focal_point: Optional[Union[Tuple, List, np.ndarray]] = None,
    image_size: Tuple[int, int] = (800, 800),
    colormap: str = "turbo",
    alpha: Optional[Union[List[float], str]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    threshold: Optional[float] = 1.1,
    threshold_mode: str = "below",
    show_axes: bool = False,
    show_colorbar: bool = True,
    background: str = "white",
    zoom: float = 1.0
) -> np.ndarray:
    """
    Render a 3D voxel volume using vedo and save as PNG.
    
    Parameters:
    -----------
    volume : np.ndarray
        3D array representing the voxel data.
        Shape: (depth, height, width) or any (D, H, W) configuration.
    output_path : str
        Path to save the rendered PNG image.
    camera_position : tuple/list/array, optional
        3D position of the camera [x, y, z].
        If None, vedo will choose automatically.
    focal_point : tuple/list/array, optional
        3D point the camera looks at [x, y, z].
        If None, looks at the center of the volume.
    image_size : Tuple[int, int]
        Output image dimensions (width, height).
    colormap : str
        Colormap name (e.g., "turbo", "viridis", "jet", "bone").
    alpha : list of floats or str, optional
        Opacity values. Can be:
        - List of 5 values [0-1] for low to high density
        - "auto" for automatic opacity
        - None for default [0, 0.1, 0.3, 0.6, 1.0]
    vmin : float, optional
        Minimum value for colormap normalization.
    vmax : float, optional
        Maximum value for colormap normalization.
    threshold : float, optional
        Threshold value for hiding voxels. If provided, voxels will be
        hidden based on threshold_mode.
    threshold_mode : str
        How to apply threshold:
        - "below": hide voxels below threshold (default)
        - "above": hide voxels above threshold
    show_axes : bool
        Whether to show coordinate axes.
    show_colorbar : bool
        Whether to show a colorbar for the volume.
    background : str
        Background color (e.g., "white", "black", "gray").
    zoom : float
        Zoom factor (larger = closer).
    
    Returns:
    --------
    np.ndarray
        Rendered image as numpy array (height, width, 3 or 4).
    
    Example:
    --------
    >>> volume = np.random.rand(64, 64, 64)
    >>> render_volume(
    ...     volume, 
    ...     output_path="result.png",
    ...     camera_position=(100, 100, 100),
    ...     focal_point=(32, 32, 32)
    ... )
    """
    
    # Default alpha values if not provided
    if alpha is None:
        alpha = [0, 0.1, 0.3, 0.6, 1.0]
    elif alpha == "auto":
        alpha = "auto"
    
    # Apply thresholding if requested
    volume_processed = volume.copy()
    if threshold is not None:
        if threshold_mode == "below":
            volume_processed[volume_processed < threshold] = 0
        elif threshold_mode == "above":
            volume_processed[volume_processed > threshold] = 0
        else:
            raise ValueError(f"Invalid threshold_mode: {threshold_mode}. Use 'below' or 'above'.")
    
    # Create vedo Volume object
    vol = Volume(volume_processed)
    
    # Apply colormap and alpha
    vol.cmap(colormap, vmin=vmin, vmax=vmax)
    if isinstance(alpha, list):
        vol.alpha(alpha)
    
    # Create plotter for offscreen rendering
    plt = Plotter(offscreen=True, size=image_size, bg=background)
    plt.add(vol)
    
    # Add colorbar if requested
    if show_colorbar:
        # Create scalar bar with explicit label configuration
        sb = addons.ScalarBar(
            vol,
            title="",
            nlabels=5,
            c='black' if background == 'white' else 'white',  # Contrast with background
            horizontal=False,
        )
        plt.add(sb)
    
    # Set up camera if provided
    if camera_position is not None:
        cam_pos = np.array(camera_position)
        
        if focal_point is None:
            # Default focal point is the center of the volume
            focal_point = np.array(volume.shape) / 2.0
        else:
            focal_point = np.array(focal_point)
        
        plt.camera.SetPosition(cam_pos)
        plt.camera.SetFocalPoint(focal_point)
        plt.camera.SetViewUp(0, 0, 1)  # Z-up by default
    
    # Apply zoom
    if zoom != 1.0:
        plt.camera.Zoom(zoom)
    
    # Show axes if requested
    if show_axes:
        plt.show(axes=1)
    
    # Render and save
    plt.show()
    plt.screenshot(output_path)
    
    # Read back the screenshot as numpy array
    from PIL import Image
    img = Image.open(output_path)
    screenshot = np.array(img)
    
    print(f"Saved rendered volume to {output_path}")
    
    plt.close()
    
    return screenshot


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
    
    # Render from different angles using vedo
    render_volume(
        volume=volume,
        output_path="sphere_render.png",
        camera_position=(10, 10, 100),
        focal_point=(32, 32, 32),
        image_size=(800, 800),
        colormap="turbo",
        alpha="auto",
        threshold=0.3,  # Hide voxels below 0.3
        threshold_mode="below",
        show_axes=True,
        show_colorbar=True,
        background="black",
        zoom=1
    )
    
    print("Sphere rendering complete!")
