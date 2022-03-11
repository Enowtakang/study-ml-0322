import pandas as pd
from random import choice


"""
Load dataset
"""
path = "folder_path/splice.data"
names = ['a', 'b', 'dna_sequence']
original_splice_df = pd.read_csv(path, names=names)


"""
Add a DNA label column (value = 1)
'1' indicates that the DNA sequences 
are real
"""
original_splice_df['dna_label'] = pd.Series(
    [1 for x in range(len(original_splice_df.index))])


"""
Create a new dataframe which contains
only the 'dna_sequence' and the 
'dna_label' columns from the 
original_splice_df
"""
selected_columns = ['dna_sequence', 'dna_label']
original_splice_df_2 = original_splice_df[selected_columns]


"""
Create an empty dataframe.
This dataframe would hold the 
computer-generated DNA sequences.
"""
generated_data_df = pd.DataFrame()


"""
Create a function to generate a sequence.
Supply a default length of 10 nucleotides.
"""


def generate_sequence(length=10):
    global sequence
    # define the bases
    bases = ['A', 'T', 'G', 'C']
    # use list comprehension to generate the sequence
    sequence = [choice(bases) for i in range(length)]
    # convert the sequence to a string
    sequence = ''.join(sequence)
    # return the sequence
    return sequence


"""
Generating 3190 sequences of the same 
length (60 nucleotides) using the sequence 
generation function above.
"""
number_of_sequences = 3190
sequence_length = 60

sequences_0 = [
    generate_sequence(sequence_length) for i in range(
        number_of_sequences)]


"""
Store the list of sequences into a pandas series.
Then create a new column ('dna_sequence') in the 
empty database and store the data.
"""
a = pd.Series([i for i in sequences_0])
generated_data_df.insert(0, "dna_sequence", a)


"""
Add a DNA label column (value = 0) 
to the 'generated_data_df' dataframe.
'0' indicates that the DNA sequences  
are not real
"""
b = pd.Series([0 for c in range(len(generated_data_df.index))])
generated_data_df.insert(1, "dna_label", b)


"""
Concatenate both:
generated_data_df and original_splice_df_2
dataframes, then save them as a csv file.
"""
concatenated = pd.concat(
    [generated_data_df, original_splice_df_2])
concatenated.to_csv("project_data_final.csv")

"""
P.S. 
I later discovered that there was
white space before each of the 
real DNA sequences, so i reloaded 
the data and took care of the problem  
"""

path = "folder_path/project_data_final.csv"
research_data = pd.read_csv(path, skipinitialspace=True)
research_data.to_excel("research_data.xlsx")


# Thanks for going through this script.
# Thanks for going through this script.
# Thanks for going through this script.
