#!/usr/bin/python3
# -*- coding: utf-8 -*-


### Script available from https://github.com/adbar/german-reddit
### Copyright Adrien Barbaresi, 2015.
### MIT license


from __future__ import print_function

import argparse
import atexit
import io
from multiprocessing import Pool, Value, Lock
import re
import time
import ujson
import pathlib
import zstandard
import os
import urllib.request

# language specific imports
#from ftlangdetect import detect
import fasttext


lock = Lock()
c = 0



parser = argparse.ArgumentParser()
parser.add_argument('-m', '--month', dest='month', help='month of the input file', required=True)
parser.add_argument('-y', '--year', dest='year', help='year of the input file', required=True)
parser.add_argument('-p', '--processes', dest='processes', help='number of processes (has to be an integer)', required=True)
args = parser.parse_args()


def get_de_subr():
    l = set()
    with open("/../data/subreddits.txt", "r") as subr:
        for sub in subr:
            l.add(sub.strip())
    return l

subr = get_de_subr()
ld_model = fasttext.load_model("/tmp/lid.176.bin")  # TODO: change to path of your Fasttext model


def check_if_german(text):
    langid_response = ld_model.predict(text, k=1)
    print(langid_response, type(langid_response))
    return langid_response["lang"] == 'de'


def sanitize_body(parsed_json):
    sanitized_body = parsed_json['body'].replace('\r', '')
    sanitized_body = sanitized_body.replace('\n', ' ')
    sanitized_body = re.sub(r'\(?http[^ ]+\)?', '', sanitized_body)
    return sanitized_body


# line-by-line filtering
def process_line(line):
    global c
    c += 1
    if (c % 1000000) == 0:
        print(f"{c} comments analyzed.")

    line = line.strip()
    parsed_json = ujson.loads(line)
    if parsed_json["subreddit"] not in subr:
        return

    sanitized_body = sanitize_body(parsed_json)
    if len(sanitized_body) > 30 and sanitized_body != '[deleted]':
        # fastText language identification
        if check_if_german(sanitized_body):
            return parsed_json['id'], sanitized_body, line


# store result in file
def handle_result(result):
    if result:
        # lock necessary because of concurrency / race conditions
        with lock:
            with io.open(outputfile, 'a', encoding='utf-8') as outputfh:
                outputfh.write(str(result[2]) + '\n')

# shut down processes nicely
@atexit.register
def the_end():
    # pool.close()
    # pool.terminate()
    pass

# launch multiprocessing and collect results
if __name__ == "__main__":
    start_time = time.time()

    print(f"Year {args.year}, month {args.month}, type {type(args.month)}")
    month = str(args.month) if int(args.month) > 9 else f"0{args.month}"
    f_name = f"RC_{args.year}-{month}"
    inputfile = f"../reddit_comments/{f_name}.zst"
    processes = args.processes
    outputfile = f"../german_reddit_comments/{f_name}_ft.txt"

    print(f"Start time: {start_time}")
    print ('### starting:', inputfile)
    print ('### pool size:', processes)
    input_file = pathlib.Path(inputfile)
    with open(input_file, 'rb') as compressed:
        decomp = zstandard.ZstdDecompressor(max_window_size=2147483648)
        decomp_file = pathlib.Path(r"../tmp") / f_name
        with open(decomp_file, 'wb') as destination:
            decomp.copy_stream(compressed, destination)

    pool = Pool(processes = int(processes), maxtasksperchild=10000)
    with open(decomp_file, 'r', encoding="utf-8") as inputfh:
        print(f"Opened the decompressed file and starting to process the lines.")
        results = pool.imap_unordered(process_line, inputfh, 50000)
        for r in results:
            handle_result(r)



    pool.close()
    pool.join()
    os.remove(decomp_file)
    #os.remove(input_file)

    end_time = time.time()
    print(f"Ending session. This took {(end_time - start_time)/60} minutes.")


