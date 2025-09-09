import numpy as np
from scipy.spatial.distance import cdist

def calculate_thickness_epicardium_endocardium(seg_slice, pixel_spacing, endo_label=1, epi_label=2):
    """
    Calculate myocardial thickness from the distance between epicardium and endocardium.

    Args:
        seg_slice (numpy.ndarray): 2D segmentation slice with labeled regions.
        pixel_spacing (float): Spacing between pixels in mm (assumes isotropic spacing).
        endo_label (int): Label for endocardium.
        epi_label (int): Label for epicardium.
        
    Returns:
        avg_thickness_mm (float): Average myocardial thickness in mm.
    """
    # Identify boundary pixels for endocardium and epicardium
    endo_boundary = np.argwhere(seg_slice == endo_label)
    epi_boundary = np.argwhere(seg_slice == epi_label)
    
    if len(endo_boundary) == 0 or len(epi_boundary) == 0:
        # If one of the boundaries is missing, return 0
        return 0.0

    # Compute pairwise distances between boundaries
    distances = cdist(endo_boundary, epi_boundary)  # Shape: (num_endo_points, num_epi_points)

    # Find the minimum distance for each endocardium point
    min_distances = distances.min(axis=1)

    # Convert to mm using pixel spacing
    min_distances_mm = min_distances * pixel_spacing

    # Calculate the average thickness
    avg_thickness_mm = min_distances_mm.mean()

    return avg_thickness_mm

# Example usage
if __name__ == "__main__":
    # Example segmented slice with endocardium and epicardium
    seg_slice = (np.abs(normGT[dispind,vol_slice,:,:]))
    
    # Define pixel spacing in mm
    pixel_spacing = 0.5  # Example: 0.5 mm per pixel

    # Calculate myocardial thickness
    avg_thickness_mm = calculate_thickness_epicardium_endocardium(
        seg_slice, pixel_spacing, endo_label=1, epi_label=2
    )
    
    # Print the result
    print(f"Average Myocardial Thickness: {avg_thickness_mm:.2f} mm")


