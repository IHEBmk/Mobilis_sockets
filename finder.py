import os

def find_non_utf8_files(root_dir):
    for foldername, subfolders, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.py'):
                filepath = os.path.join(foldername, filename)
                try:
                    with open(filepath, encoding='utf-8') as f:
                        f.read()
                except UnicodeDecodeError:
                    print(f"⚠️ Encoding issue in: {filepath}")

# Change the path to your Django project's root if needed
find_non_utf8_files(".")

