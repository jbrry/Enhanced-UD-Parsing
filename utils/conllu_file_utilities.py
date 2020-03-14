import argparse
import os
import codecs

"""
Reads a regular CoNLLU file and performs various utilities.

Example usage:
      prep new data dir: cp -r data/train-dev/ data/train-dev-filtered/
      replace original file with filtered version: python utils/conllu_file_utilities.py -i data/train-dev/UD_Czech-CAC/cs_cac-ud-train.conllu -o data/train-dev-filtered/UD_Czech-CAC/ -m max-len -c 197
"""

parser = argparse.ArgumentParser(description='File utils')
parser.add_argument('--input', '-i', type=str, help='Input CoNLLU file.')
parser.add_argument('--outdir','-o', type=str, help='Directory to write out files to.')
parser.add_argument('--mode', '-m', type=str, default='utf-8', help='The behaviour to filter sentences by: elided, max-len.')
parser.add_argument('--cutoff', '-c', type=int, default='utf-8', help='The cutoff value for the maximum length of senences if using mode max-len.')
parser.add_argument('--encoding', '-e', type=str, default='utf-8', help='Type of encoding.')
args = parser.parse_args()

if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)


def conllu_reader(infile):
    """Simple CoNLL-U reader to return a list of sentences from a file 
       as well as token/sentence counts."""
    
    print("Reading sentences from {}".format(infile))

    file = codecs.open(infile, 'r', encoding="utf-8")
    
    sentences = []
    current_sentence = []
    current_tokens=[]
    sent_count=0
    token_counts=0
    max_sentence_len = 0

    while True:
        line = file.readline()
        if not line:
            break
        # new conllu sentence
        if line.isspace():
            # append the current sentence to sentences
            sentences.append(current_sentence)
            # update token/sentence counts
            tokens_per_sent = len(current_tokens)
            if tokens_per_sent > max_sentence_len:
                max_sentence_len = tokens_per_sent              
            token_counts += tokens_per_sent
            sent_count += 1
            # clear the lists for the next conllu sentence
            current_sentence = [] 
            current_tokens = []
        else:
            # add text and conllu items
            current_sentence.append(line)
            # normal conllu line
            if line.count('\t') == 9:
                rows = line.split('\t')
                word = rows[1]
                current_tokens.append(word)
    
    file.close()
    assert len(sentences) == sent_count
    print("Found {} sentences and {} tokens".format(sent_count, token_counts))
    print("Longest sentece = {} words".format(max_sentence_len))
    return sentences, token_counts, sent_count


def write_conllu(data, outfile):
    """Write list of sentences to '\n' separated sentences in a file."""

    with codecs.open(outfile, 'w', encoding="utf-8") as f:
        for block in data:
            for sent in block:
                for entry in sent:
                    f.write(entry)
            f.write('\n')

# metadata
in_name = os.path.basename(args.input)
file_string = in_name.split('.')[0]
tbid = file_string.split('-')[0]
file_type = file_string.split('-')[-1]

# gather sentences from file
sentences, token_counts, sent_count = conllu_reader(args.input)


# filter sentences by max length
if args.mode == "max-len":
    processed_sentences = []
    num_sents_below_cutoff = 0
    
    for sent in sentences:
        if len(sent) > args.cutoff:
            continue
        else:
            processed_sentences.append(sent)
            num_sents_below_cutoff += 1
    
    out_file_string = (f"{tbid}-ud-{file_type}.conllu")
    
    print("Out of {} sentences, {} remain after filtering by length".format(sent_count, num_sents_below_cutoff))

# keep only sentences with elided tokens
elif args.mode == "elided":
    # empty list to collect sentences with ellided tokens
    processed_sentences = []
    num_sents_with_ellided_tokens = 0
    
    for sent in sentences:
        for conllu_row in sent:
            # check for copy node (ellided token)
            if "CopyOf" in conllu_row:
                if sent not in processed_sentences:
                    processed_sentences.append(sent)
                    num_sents_with_ellided_tokens += 1
                    
    out_file_string = (f"{tbid}-ud-{file_type}-ellided_only.conllu")                
    print("Out of {} sentences, {} contain copy nodes".format(sent_count, num_sents_with_ellided_tokens))


out_file = os.path.join(args.outdir, out_file_string)
print("Writing output to {}".format(out_file))
conllu_out = write_conllu(processed_sentences, out_file)
