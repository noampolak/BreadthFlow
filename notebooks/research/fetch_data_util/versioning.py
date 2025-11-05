"""
File versioning utilities for managing data downloads.

Provides functions to handle versioning when files already exist,
backing up existing files, and maintaining latest copies.
"""

import os
import shutil
from datetime import datetime
import re


def get_versioned_filename(base_path, prefix, start_year, end_year, download_date_str):
    """
    Generate a versioned filename.
    
    Parameters:
    -----------
    base_path : str
        Base directory path
    prefix : str
        File prefix (e.g., 'stocks_prices')
    start_year : int or str
        Start year for the data
    end_year : int or str
        End year for the data
    download_date_str : str
        Download date in DD_mm_yyyy format
        
    Returns:
    --------
    str
        Full path to versioned filename
    """
    filename = f"{prefix}_{start_year}-{end_year}_download_date_{download_date_str}.pkl"
    return os.path.join(base_path, filename)


def get_version_number(filepath):
    """
    Extract version number from filename if it exists.
    
    Looks for pattern: filename_v{number}.{ext}
    
    Parameters:
    -----------
    filepath : str
        Path to file
        
    Returns:
    --------
    int
        Version number (0 if no version found)
    """
    base_name = os.path.basename(filepath)
    # Look for _v{number} pattern before file extension
    match = re.search(r'_v(\d+)\.(pkl|csv)$', base_name)
    if match:
        return int(match.group(1))
    return 0


def backup_existing_file(filepath):
    """
    Backup an existing file by adding version number.
    
    If file exists, renames it to filename_v{version}.{ext}
    where version is incremented from existing versions.
    
    Parameters:
    -----------
    filepath : str
        Path to file that may exist
        
    Returns:
    --------
    str or None
        Path to backup file if backup was created, None otherwise
    """
    if not os.path.exists(filepath):
        return None
    
    # Find all existing versions
    base_dir = os.path.dirname(filepath)
    base_name = os.path.basename(filepath)
    file_ext = os.path.splitext(base_name)[1]
    base_name_no_ext = os.path.splitext(base_name)[0]
    
    # Find all versioned files with same base name
    existing_versions = []
    for filename in os.listdir(base_dir):
        if filename.startswith(base_name_no_ext):
            version = get_version_number(os.path.join(base_dir, filename))
            existing_versions.append(version)
    
    # Get next version number
    next_version = max(existing_versions) + 1 if existing_versions else 1
    
    # Create backup filename
    backup_path = filepath.replace(file_ext, f'_v{next_version}{file_ext}')
    
    # Rename existing file
    shutil.move(filepath, backup_path)
    
    return backup_path


def save_with_versioning(data, filepath, prefix, start_year, end_year, download_date_str, save_csv=True):
    """
    Save data with automatic versioning.
    
    If target file exists, backs it up with version number first.
    Then saves new data to the target file.
    Optionally saves CSV version as well.
    
    Parameters:
    -----------
    data : object
        Data to save (DataFrame, dict, etc.)
    filepath : str
        Target file path (should end in .pkl)
    prefix : str
        File prefix for logging
    start_year : int or str
        Start year for filename
    end_year : int or str
        End year for filename
    download_date_str : str
        Download date string
    save_csv : bool
        Whether to also save as CSV (for DataFrames)
        
    Returns:
    --------
    tuple (str, str or None)
        (saved_filepath, backup_filepath)
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Backup existing file if it exists
    backup_path = backup_existing_file(filepath)
    
    if backup_path:
        print(f"   📦 Backed up existing file to: {os.path.basename(backup_path)}")
    
    # Save new data
    if hasattr(data, 'to_pickle'):
        # DataFrame - save as pickle
        data.to_pickle(filepath)
        
        # Also save as CSV if requested
        if save_csv:
            csv_path = filepath.replace('.pkl', '.csv')
            csv_backup = backup_existing_file(csv_path)
            data.to_csv(csv_path, index=False)
            if csv_backup:
                print(f"   📦 Backed up CSV to: {os.path.basename(csv_backup)}")
    elif isinstance(data, dict):
        # Dictionary - save as JSON
        import json
        json_path = filepath.replace('.pkl', '.json')
        json_backup = backup_existing_file(json_path)
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        if json_backup:
            print(f"   📦 Backed up JSON to: {os.path.basename(json_backup)}")
        print(f"   ✅ Saved JSON to: {os.path.basename(json_path)}")
    else:
        # Generic pickle
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    print(f"   ✅ Saved to: {os.path.basename(filepath)}")
    
    return filepath, backup_path


def copy_to_latest(filepath, latest_dir, latest_filename):
    """
    Copy file to latest directory with standardized name.
    
    Parameters:
    -----------
    filepath : str
        Source file path
    latest_dir : str
        Directory for latest files
    latest_filename : str
        Name for latest file (without extension)
        
    Returns:
    --------
    str
        Path to latest file
    """
    os.makedirs(latest_dir, exist_ok=True)
    
    # Determine extension
    ext = os.path.splitext(filepath)[1]
    latest_path = os.path.join(latest_dir, f"{latest_filename}{ext}")
    
    # Copy file
    shutil.copy2(filepath, latest_path)
    
    print(f"   📋 Copied to latest: {os.path.basename(latest_path)}")
    
    return latest_path

