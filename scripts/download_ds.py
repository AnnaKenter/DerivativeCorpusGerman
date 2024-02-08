import urllib.request

"""
This file makes downloading all the data from Reddit a little easier :)
"""

year = "2007"  # TODO: change year-string to the year you want to download from Reddit
for i in range(1, 12):
    if i < 10:
        month = f"0{i}"
    else:
        month = i
    download_file = f"../reddit_comments/RC_{year}-{month}.zst"  # args.inputfile
    print(f"Starting to download {year}, {month}.")
    x = urllib.request.urlretrieve(f"https://files.pushshift.io/reddit/comments/RC_{year}-{month}.zst", download_file)