import argparse
import os
import codecs

"""
Reads a regular CoNLLU file and writes only those sentences which contain ellided tokens to an output file.
"""

parser = argparse.ArgumentParser(description='File utils')
parser.add_argument('--input', '-i', type=str, help='Input CoNLLU file.')
parser.add_argument('--outdir','-o', type=str, help='Directory to write out files to.')
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
    return sentences, token_counts, sent_count


def write_conllu(data, outfile):
    """Write list of sentences to '\n' separated sentences in a file."""

    with codecs.open(outfile, 'w', encoding="utf-8") as f:
        for block in data:
            for sent in block:
                for entry in sent:
                    f.write(entry)
            f.write('\n')



in_name = os.path.basename(args.input)
file_string = in_name.split('.')[0]
tbid = file_string.split('-')[0]
file_type = file_string.split('-')[-1]
out_file_string = (f"{tbid}-ud-{file_type}-ellided_only.conllu")
out_file = os.path.join(args.outdir, out_file_string) 

# gather sentences from file
sentences, token_counts, sent_count = conllu_reader(args.input)

# empty list to collect sentences with ellided tokens
sents_containing_ellided_tokens = []
num_sents_with_ellided_tokens = 0

for sent in sentences:
    for conllu_row in sent:
        # check for copy node (ellided token)
        if "CopyOf" in conllu_row:
            sents_containing_ellided_tokens.append(sent)
            num_sents_with_ellided_tokens += 1

print("Out of {} sentences, {} contain copy nodes".format(sent_count, num_sents_with_ellided_tokens))
print("Writing output to {}".format(out_file))

ellided_out = write_conllu(sents_containing_ellided_tokens, out_file)

