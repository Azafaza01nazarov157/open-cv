import urllib.request
import bz2
import os

def download_file(url, filename):
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print("Download completed!")

def extract_bz2(filename):
    print(f"Extracting {filename}...")
    with bz2.open(filename, 'rb') as source, open(filename[:-4], 'wb') as dest:
        dest.write(source.read())
    print("Extraction completed!")
    # Удаляем сжатый файл
    os.remove(filename)

def main():
    url = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    filename = "shape_predictor_68_face_landmarks.dat.bz2"
    
    # Скачиваем файл
    download_file(url, filename)
    
    # Распаковываем файл
    extract_bz2(filename)
    
    print("Model is ready to use!")

if __name__ == "__main__":
    main() 