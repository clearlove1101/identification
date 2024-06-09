import zipfile


def unzip_file(zip_path, extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        print(zip_ref.namelist())
        zip_ref.extractall(extract_path)



unzip_file(r"D:\reid\PA100K\data.zip", r"D:\reid\PA100K\data")
