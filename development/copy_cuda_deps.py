import os
import shutil


def copy_include_files(source_dir, dest_dir, target='include'):
    for root, dirs, files in os.walk(source_dir):
        for directory in dirs:
            if directory == target:
                source_include_dir = os.path.join(root, directory)
                for file in os.listdir(source_include_dir):
                    source_file_path = os.path.join(source_include_dir, file)
                    if not os.path.isfile(source_file_path):
                        continue
                    dest_file_path = os.path.join(dest_dir, file)
                    try:
                        shutil.copy(source_file_path, dest_file_path)
                        print(f"Coped {source_file_path} to {dest_file_path}")
                    except FileNotFoundError:
                        print(f"File {source_file_path} not found")
                    except shutil.Error as e:
                        print(f"Error: {e}")

if __name__ == "__main__":
    source_dir = "/home/randxie/anaconda3/envs/vllm/lib/python3.9/site-packages/nvidia"
    dest_dir = "/home/randxie/anaconda3/envs/vllm/include"
    copy_include_files(source_dir, dest_dir, target='include')


    source_dir = "/home/randxie/anaconda3/envs/vllm/lib/python3.9/site-packages/nvidia"
    dest_dir = "/home/randxie/anaconda3/envs/vllm/lib"
    copy_include_files(source_dir, dest_dir, target='lib')
