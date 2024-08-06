import os
from pymediainfo import MediaInfo
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata
from tqdm import tqdm

def is_mp4_corrupted(file_path):
    """Use hachoir to check if an MP4 file is corrupted."""
    parser = createParser(file_path)
    if not parser:
        return True

    try:
        metadata = extractMetadata(parser)
        if metadata is None:
            return True
    except Exception:
        return True

    return False

def check_directory_for_corrupted_mp4s(directory):
    corrupted_files = []
    for root, dirs, files in os.walk(directory):
        for file in tqdm(files):
            if file.lower().endswith('.mp4'):
                file_path = os.path.join(root, file)
                if is_mp4_corrupted(file_path):
                    corrupted_files.append(file_path)
    return corrupted_files


directory_to_check = '/data/hvw5451/data/videochat2_conversations/videos'  # 替换为你要检查的目录路径
corrupted_files = check_directory_for_corrupted_mp4s(directory_to_check)

if corrupted_files:
    print(f"Found {len(corrupted_files)} corrupted MP4 files:")
    for file in corrupted_files:
        print(file)
else:
    print("No corrupted MP4 files found.")
















# directory_to_check = '/home/haibo/data/InternVid-G/videos'  # 替换为你要检查的目录路径

# files = os.listdir(directory_to_check)

# files_1 = files[850:900]


# for file in files_1:
#     print(file)
#     is_mp4_corrupted(os.path.join(directory_to_check, file))


# """
# QHxzKtysxc0.mp4
# O_kvVWhEY3M.mp4
# Mw0jKTAue40.mp4
# pkZC46LCgKE.mp4
# VJ8vqatmn0M.mp4
# JiGShiWw_Ic.mp4
# """

# moment_10m_delete = ['FOk358JthPQ.mp4', 'LhFHBud42a0.mp4']
# for file in moment_10m_delete:
#     os.remove(os.path.join(directory_to_check, file)) 
