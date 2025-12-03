from scipy.io import loadmat

def load_cellinfo_data(file_path):
    """
    Load and extract arrays from the 'cellinfo' structure in the given .mat file.
    
    Args:
        file_path (str): Path to the .mat file.
        
    Returns:
        dict: A dictionary where keys are field names and values are the corresponding arrays.
    """
    
    # Load the .mat file
    mat_data = loadmat(file_path)
    
    # Extract the 'cellinfo' data
    cellinfo_data = mat_data.get('cellinfo')
    
    if cellinfo_data is None:
        raise ValueError("'cellinfo' key not found in the .mat file.")
    
    # Initialize a dictionary to store the extracted data
    data_dict = {}
    
    # Iterate through each field and extract its content
    for field_name in cellinfo_data.dtype.names:
        data_dict[field_name] = cellinfo_data[field_name][0, 0]
    
    return data_dict
