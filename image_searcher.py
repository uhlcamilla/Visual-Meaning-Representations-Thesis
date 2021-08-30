"""
Author: Camilla Uhlbäck
Date:   2021-06-01

Derived from Moritz Jakob's code for downloading images. 
With the addition of error handling to skip corrupted images etc. 
Pool from multiprocessing is used to search for and download many images concurrently.
requires the 'google_images_search_modified' library, which has had error handling added to the resize method resize() at line 330 of google_images_search_modified/fetch_resize_save.py:
    remove:    
        -   img = Image.open(fd_img)
    add:
        +   try:
        +       img = Image.open(fd_img)
        +   except UnidentifiedImageError:
        +       print("UnidentifiedImageError resizing:", path_to_image)
        +       return

Instructions:
    - create folders for images_dir, failed_words_dir and working_words_dir. create blank .txt for failed_words_list_dir. add directory for master_word_list_dir file
    - add API_KEY and cx for your own custom search engine
    - change the working directory to where the google_images_search_modified folder library is stored (likely where this file is)
    - set the number of images to search for and number to exit on (num_imgs and target_images inside downloader_process)
    - set the number of processes to launch and number of words per process

Note: 
    - Text files are used here to handle word lists as they persist even if the script crashes or is aborted. 
        Killing the script should not lose much more than the images for the words currently being worked on.
    - Rerunning the script (even with n_words_per_worker = 0) will still collect the previously failed words and add them to the master list.
    - Words on the failed_words_list will never be searched for. It could be that the list ends up containing a large number of words, in which case it can be purged and num_imgs increased. If this is not successful, a different approach must be found.
    - Keep in mind that each search, downloaded image (cropped or not) will contribute to the number of API calls, and therefore cost (it is unclear _exactly_ how much they contribute to the number).
"""

import pandas as pd
import os, sys
from PIL import Image, UnidentifiedImageError           # mostly to resize images
from io import BytesIO                                  # to compare images
from sklearn.metrics.pairwise import cosine_similarity  # to compare images
import numpy as np                                      # to compare images
from multiprocessing import Pool                        # allow multiple processes at once

os.chdir("C:/Users/path/to/parent/of/...") #folder containing google_images_search_modified
from google_images_search_modified import GoogleImagesSearch     # modified to skip image resizing when the image is corrupted

master_word_list_dir = 'C:/Users/path/to/your/all_no_dupes.txt'   # a .txt file of words to be downloaded (no spaces, no quotes, no seperators apart from newline)
failed_words_list_dir = 'C:/Users/path/to/your/failed_words.txt'  # a .txt file for a list of words which are the google search api has failed to download

images_dir = "C:/Users/path/to/your/imagesearchnotebook/images"                               # parent directory to save images in. for each word a folder will be created
failed_words_dir = "C:/Users/path/to/your/failed_words_bucket/"   # a folder for temporarily holding lists of failed words
working_words_dir = "C:/Users/path/to/your/working_words_bucket"  # a folder for temporarily holding lists of words being worked on

API_KEY = ''  # the credentials for the Google API
cx = ''

def no_dups(height, width, num_imgs, query, direc, target_images = None):     # checks if an image is already in the folder before downloading it as a resized version
    """ Searches for query and downloads target_images, or if target_images is None, downloads num_imgs. 
        First saves a cropped version of each result with the name 'temp_cropped' (occasionally with a different name due to some unknown error). 
        The cropped image is compared to all other images, if it has a cosine similarity < 0.999, a full version is saved with the name <query>.jpg
        
        Many images fail to download / are corrupted. It is unclear why this occurs. 
        To work around it, num_imgs should be set above the actual required number, and target_images can be used to set the required number

        Parameters: 
            height::int         - pixel height of the cropepd image
            width::int          - pixel width of the cropped image
            num_imgs::int       - the number of images to request when searching google.
            query::str          - search querry
            direc::str          - parent path to save. images are saved in a folder inside the parent path: C:/<direc>/<query>
            target_images::int  - number of images required. After target_images are successfully downloaded the function exits. 
    """

    gis = GoogleImagesSearch(API_KEY, cx) 
    
    _search_params = {'q': query, 'num': num_imgs}

    gis.search(search_params = _search_params, path_to_dir = os.path.join(direc, query), width = width, height = height, custom_image_name = "temp_cropped")    
    # a cropped version of all images are saved at this point
    
    
    vectors_dict = {}
    for i in os.listdir(os.path.join(direc, query)):     
        if os.path.isdir(i):
            continue
        try:
            pic = Image.open(os.path.join(direc, query, i))  
            pic_rgb = pic.convert("RGB")                     # converting to rgb
            pic_rgb_rszd = pic_rgb.resize((height, width))   # resizing
            vec = np.array(pic_rgb_rszd)                     # vectorizing the image
            vectors_dict[i] = vec                            # storing the image vectors into a dict
        except UnidentifiedImageError:
            print("UnidentifiedImageError for vector_dict", i)    

    # check if its a duplicate with cosine similarity and download if we don't have it yet
    print("num images found:", len(gis.results()))
    n_downloads = 0
    gis._custom_image_name = query
    for img in gis.results():
        
        # initiate a BytesIO object
        my_bytes_io = BytesIO()

        # set BytesIO object back to 0
        my_bytes_io.seek(0)

        # take raw img data
        raw_image_data = img.get_raw_data()

        # write raw data to BytesIO object
        img.copy_to(my_bytes_io, raw_image_data)

       # set BytesIO object back to 0 again so PIL can read it
        my_bytes_io.seek(0)

        # create temporary image object
        try:
            temp_img = Image.open(my_bytes_io)
        except UnidentifiedImageError:          # the image was corrupted - skip it 
            print("UnidentifiedImageError opening temp_img with url:", img.url)
            continue
        temp_img_rgb = temp_img.convert("RGB")

        # resize it AND REDOWNLOAD IT! this ends up downloading the resized image into the main folder -> path above fixed!
        rszd = temp_img_rgb.resize((height, width)) # must be a tuple, otherwise causes a filter error

        # vectorise it
        arr = np.array(rszd)      
        
        # calculate cosine similarities for each file in vectors_dict with the array of the temp_img
        cosine_sim_list = []
        for word, number in vectors_dict.items():
            curr_arr = np.array(vectors_dict[word])
            cosine_sim = cosine_similarity(curr_arr.reshape(1,-1), arr.reshape(1,-1))
            cosine_sim_list.append(cosine_sim)

        # if none of the cosine similarities exceeds 0.999, download the image
        if all(i < 0.999 for i in cosine_sim_list):
            img.download(os.path.join(direc, query))
            print("downloaded", img.url)   
            n_downloads += 1
        
        if target_images and n_downloads == target_images:
            print("####", n_downloads, "images downloaded - target reached ")
            break

    print(f"\ndownloaded {n_downloads} images")

def downloader_process(working_file_dir):
    """ This function is called multiple times by seperate processes to download images concurrently.
        no_dupes() is called for each work in the working_file_dir file. If an unhandled error occured inside no_dupes(), the word is added to a temporary failed word list.
        text files are used rather than lists in memory to avoid losing progress in the case the run crashes or is terminated.

        working_file_dir::str - path to a .txt file containing a list of words to download.        
    """
    failed = []
    while True:
        with open(working_file_dir) as f:
            words = f.read().splitlines()
        words = list(set(words) - set(failed))  # we shouldn't need this line, but we do
        if len(words) > 0:
            word = words.pop()
            print(f"\n ID: {os.path.basename(working_file_dir)} | {word} | {len(words)} remaining\n")
            
            try:
                no_dups(155, 155, 35, word, images_dir, 25)     # This is where to modify num_imgs and target_images

            except Exception as e:
                print("ERROR", e)
                print(f"####\n{word} failed\n####")
                failed.append(word)
                with open(os.path.join(failed_words_dir, os.path.basename(working_file_dir) + ".txt"), "a") as f:
                    f.write("%s\n" % word)
                continue

            with open(working_file_dir, "w") as f:
                for w in words:
                    f.write("%s\n" % w)
        else:
            break

    os.remove(working_file_dir)
    print(f"ID: {os.path.basename(working_file_dir)} COMPLETE | failed: {failed}")

if __name__ == '__main__':      # required for p.map() to work properly

    # Setup: collect and organise list of words
    with open(master_word_list_dir) as f:
        nouns_no_dups = f.read().splitlines()

    with open(failed_words_list_dir) as f:
        failed_words = f.read().splitlines()
        
    def chunks(lst, n):
        """ splits lst into len(lst)//n lists, each containing n items. Plus 1 shorter list of the remainder if necessary """
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def save_chunks(lists, dir):
        """ saves a list of lists as many .txt files in dir. Returns a list of directories of files created """
        files = [os.path.join(dir, f"words_{i}.txt") for i in range(len(lists))]
        for file, word_list in zip(files, lists):
            with open(file, "w") as f:
                for word in word_list:
                    f.write("%s\n" % word)
        return files

    downloaded_images = {dir:[file for file in files  if "temp_cropped" not in file] for  dir, _, files in [*os.walk(images_dir)][1:]}

    satisfied_words  = [dir.split("\\")[-1] for dir, files in downloaded_images.items() if len(files) >= 25]

    remaining_words = list(set(nouns_no_dups) - set(satisfied_words) - set(failed_words))
    
    print(f"----- {len(remaining_words)} words remaining -----")
    
    import time
    start_time = time.time()
    
    # total words >= n_words * n_workers
    n_words_per_worker = 1      # modify this! this should be larger than the n_workers
    n_workers = 4               # by making this larger your number of virtual cores won't speed up the process

    working_files = save_chunks([*chunks(remaining_words, n_words_per_worker)][:n_workers], working_words_dir)

    # Run donloader_process for each list, to download images for each word
    # Pool and p.map() allow several processes to run concurrently
    with Pool(10) as p:
        print("starting pool...")
        p.map(downloader_process, working_files)

    # Cleanup: collect the new failed words, save them to the master list and then delete the temp files. 
    for file in [f for f in os.listdir(failed_words_dir) if "words_" in f]:
        with open(os.path.join(failed_words_dir, file)) as f:
            failed_words = failed_words + f.read().splitlines()
    
    print("failed_words:", failed_words)
    
    with open(failed_words_list_dir, 'w') as f:
        for item in failed_words:
            f.write("%s\n" % item)
    
    for file in [f for f in os.listdir(failed_words_dir) if "words_" in f]:
        os.remove(os.path.join(failed_words_dir, file))

    end_time = time.time()
    print(f"--- {(end_time - start_time)} seconds total | {(end_time - start_time)/(n_words_per_worker*n_workers)} seconds per word ---" )
