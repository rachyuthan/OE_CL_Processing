def flagging(new_buildings, removed_buildings, total_buildings, criteria=0.3):
    
    """
    Flags images based on the ratio of new and removed buildings to total buildings.
    
    Parameters:
    - new_buildings: List of new building features
    - removed_buildings: List of removed building features
    - total_buildings: Total number of buildings in the baseline
    - criteria: Threshold ratio to flag an image (default is 0.3)
    
    Returns:
    - flag: Boolean indicating if the image is flagged (True) or not (False)
    """
    new_count = len(new_buildings)
    removed_count = len(removed_buildings)
    
    if total_buildings > 0:
        ratio = new_count + removed_count / total_buildings
    else:
        ratio = 0

    flag = (ratio >= criteria)

    return flag