import numpy as np


def perimeter_point(lat, lon, angle, radius):
    # Convert angle from degrees to radians
    angle = np.radians(angle)
    
    # Convert meters to degrees approximately, assuming spherical Earth
    radius /= (6371000 * np.pi / 180)  # Approximation of 1 degree in meters
    
    # Calculate relative length of the circle of longitude compared to equator
    scale = np.cos(np.radians(lat))
    
    # Calculate offsets to longitude and latitude
    # (Assuming angle = 0 means due east)
    lat += radius * np.sin(angle)
    lon += radius * np.cos(angle) / scale
    
    return lat, lon


def square_perimeter(lat, lon, radius):
    # Initialize an empty list to store the square's corner points
    square_corners = []
    
    # Calculate the corner points of the square
    p1 = perimeter_point(lat, lon, 90, radius)
    top_left = perimeter_point(p1[0], p1[1], 180, radius)

    p2 = perimeter_point(lat, lon, 270, radius)
    bottom_right = perimeter_point(p2[0], p2[1], 0, radius)

    square_corners.append(top_left)
    square_corners.append(bottom_right)

    return square_corners


def create_grid(lat, lon, grid_size, resolution):
    """
    Args:
    lat (float): Latitude of the center of the grid
    lon (float): Longitude of the center of the grid
    grid_size (float): Size of the grid in meters (side length)
    resolution (int): Distance between each point in the grid in meters
    """
    # Calculate the corner points of the square
    square_corners = square_perimeter(lat, lon, grid_size / 2)
    top_left, bottom_right = square_corners

    # Resolution means how many meters between each point
    num_points = int(grid_size / resolution) + 1

    grid_points = []

    for i in range(num_points):
        row_point = perimeter_point(top_left[0], top_left[1], 270, i * resolution)
        for j in range(num_points):
            point = perimeter_point(row_point[0], row_point[1], 0, j * resolution)
            grid_points.append(point)

    return top_left, bottom_right, grid_points


def visualize_points(points):
    import folium
    from tqdm import tqdm
    import webbrowser

    if not isinstance(points, np.ndarray):
        points = np.array(points)

    # Calculate the center of the points
    center = points.mean(axis=0)

    # Create map
    M = folium.Map(location=center, zoom_start=16)

    # Show lat/lon popups
    M.add_child(folium.LatLngPopup())

    folium.Circle(location=center, radius=2, opacity=1, fill=True, fill_color="blue").add_to(M)

    # Plot the grid
    for point in tqdm(points):
        folium.Circle(location=point, radius=1, color='green').add_to(M)

    file = f'pano_search.html'

    ## Save map and open it
    M.save(file)
    webbrowser.open(file)

