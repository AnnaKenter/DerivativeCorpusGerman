# Derivative Corpus in German
Code and data for a corpus of German derivatives in context from Reddit.

The dataset was created to finetune a BERT model to a derivative prediction task. If you do not need the data for this 
purpose read the section "Derivatives ONLY"

## Setup requirements

1. Install Python 3.9 and install dependencies from *requirements.txt*
2. Download the DeReKo-2014-II-MainArchive-STT.100000.freq file from [here](https://www.ids-mannheim.de/digspra/kl/projekte/methoden/derewo/) into the *data* directory

## Create German derivative dataset
### Download Reddit data

Download all the Reddit comments from [here](https://github.com/pushshift/api) to the *reddit* folder.
The files are named like RC_year_month.zst

You can use the *scripts/download_ds.py* script

### Filter Reddit data for German content

Once the Reddit data is downloaded we have to filter for German comments. This returns a file with German comments for 
month/year

1. Download the Fasttext model lid.176.bin and change in *scripts/get_german_comments.py* the path to where your model is
2. Run *scripts/get_german_comments.py*


### Search for derivatives
Now we can search for derivatives! This search will give you one excel table for each month/year that includes the 
derivatives of this month.

1. Run the script *scripts/get_derivatives.py*

The file is am excel file that contains for each derivative:

1. affix:	affix the derivative matches with
2. base:	    base of the derivative
3. count:    frequency of the derivative
4. in_lexica:   True/False if the derivative is in the DeReKo lexica	
5. stem: stem of the derivative
6. mode: prefixated, suffixated or both suffix and prefix
7. context: list of contexts the derivative appears in 


### Join data

We now have one table of derivatives for each month. To join the affixes 

1. run the script *scripts/join_data.py*



### Finetuning prep: Split data in training/test/dev set
This step splits the data into the train/test/dev set in the conditions SHARED and SPLIT.
In SHARED, the data is split by context and in SPLIT by derivative. This means that in the SPLIT condition, a derivative 
is either in the train or test (or dev) set.

1. Run the script *scripts/finetuning_prep.py*


## Derivatives ONLY

This dataset was developed to finetune a BERT model for derivative prediction.
If you do not care about BERT but about derivatives, you have to deactivate that all derivatives are checked for being
included into BERTs vocabulary. You can do this, for example, by always returning **True** in the method
*check_token_stem_for_bert* and *token_in_bert* in *scripts/utils_reddit.py*

You also do not need the "finetuning prep" step.

