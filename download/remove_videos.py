import os
def find_and_delete_empty_files(directory):
    empty_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.getsize(file_path) == 0:
                empty_files.append(file_path)
                os.remove(file_path)
                print(f"Deleted: {file_path}")
    return empty_files


"""
'/data3/whb/data/querYD/videos/'
'/data3/whb/data/coin/videos/'
'/data3/whb/data/HiREST/videos/'
'/data3/whb/data/Moment-10M/videos/'
'/data3/whb/data/vitt/videos/'
'/data3/whb/data/VTG-IT/videos/'
'/data3/whb/data/panda70m_2m/clips/'
'/data3/whb/data/DiDeMo/videos'
"""
# 示例使用
directory_path = '/data3/whb/data/DiDeMo/videos'

empty_files = find_and_delete_empty_files(directory_path)


