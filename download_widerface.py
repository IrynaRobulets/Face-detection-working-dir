#!/usr/bin/env python3

import requests
import os
import shutil

"""

The script has been taken from https://github.com/qdraw/tensorflow-face-object-detector-tutorial and modified
Thanks a lot, qdraw! 
TODO: When I have a time I will make a PayPal account and make a donation.

Script to download
Wider Face Training Images
Wider Face Validation Images
from Google Drive using Python3.6

http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/

Wider_face_split is included in repo

"""

# credits: https://stackoverflow.com/a/16664766
# https://stackoverflow.com/questions/16664526/howto-download-file-from-drive-api-using-python-script

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)


# The script
curr_path = os.getcwd()
models_path = os.path.join(curr_path,"downloads")

# make dir => wider_data in folder
try:
    os.makedirs(models_path)
except Exception as e:
    pass

if os.path.exists(os.path.join(models_path,"train.zip")) == False:
    print("downloading.. train.zip -- 1.47GB")
    download_file_from_google_drive("0B6eKvaijfFUDQUUwd21EckhUbWs", os.path.join(models_path,"train.zip"))

if os.path.exists(os.path.join(models_path,"val.zip")) == False:
    print("downloading.. val.zip -- 362.8MB")
    download_file_from_google_drive("0B6eKvaijfFUDd3dIRmpvSk8tLUk", os.path.join(models_path,"val.zip"))

print("files downloaded")

# unzip the files
import zipfile

if os.path.exists(os.path.join(models_path,"WIDER_train")) == False:
    with zipfile.ZipFile(os.path.join(models_path,"train.zip"),"r") as zip_ref:
        zip_ref.extractall(models_path)

if os.path.exists(os.path.join(models_path,"WIDER_val")) == False:
    with zipfile.ZipFile(os.path.join(models_path,"val.zip"),"r") as zip_ref:
        zip_ref.extractall(models_path)

print("files unziped")

os.chdir(models_path)

print("done")
