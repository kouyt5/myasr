import torch
class GreedyDecoder:
    def __init__(self, labels, blank_index=0):
        self.labels = []
        with open(labels,encoding='utf-8') as f:
            for c in f.readlines():
                self.labels.append(c.replace('\n',''))
        self.int_to_char = dict([(i, c) for (i, c) in enumerate(self.labels)])
        self.blank_index = blank_index
        space_index = len(labels)  # To prevent errors in decode, we add an out of bounds index for the space
        if ' ' in self.labels:
            space_index = self.labels.index(' ')
        self.space_index = space_index


    def convert_to_strings(self, sequences, sizes=None, remove_repetitions=False, return_offsets=False):
        """Given a list of numeric sequences, returns the corresponding strings"""
        strings = []
        offsets = [] if return_offsets else None
        for x in range(len(sequences)):
            seq_len = sizes[x] if sizes is not None else len(sequences[x])
            string, string_offsets = self.process_string(sequences[x], seq_len, remove_repetitions)
            strings.append([string])  # We only return one path
            if return_offsets:
                offsets.append([string_offsets])
        if return_offsets:
            return strings, offsets
        else:
            return strings

    def process_string(self, sequence, size, remove_repetitions=False):
        string = ''
        offsets = []
        for i in range(size):
            char = self.int_to_char[sequence[i].item()]
            if char != self.int_to_char[self.blank_index]:
                # if this char is a repetition and remove_repetitions=true, then skip
                if remove_repetitions and i != 0 and char == self.int_to_char[sequence[i - 1].item()]:
                    pass
                elif char == self.labels[self.space_index]:
                    string += ' '
                    offsets.append(i)
                else:
                    string = string + char
                    offsets.append(i)
        return string, torch.tensor(offsets, dtype=torch.int)

    def decode(self, probs, sizes=None):
        _, max_probs = torch.max(probs, 2)
        strings, offsets = self.convert_to_strings(max_probs.view(max_probs.size(0), max_probs.size(1)), sizes,
                                                   remove_repetitions=True, return_offsets=True)
        return strings, offsets
    def decoder_by_number(self,target):
        s = ''
        for i in target:
            temp = self.int_to_char.get(i)
            s+=str(temp)
        return s
if __name__ == "__main__":
    labels = "./data/labels.txt"
    decoder = GreedyDecoder(labels)
    s = decoder.decoder_by_number([1,2,3,4,5,5])
    x = torch.rand([8,32,29])
    s = decoder.decode(x)
    print(s)
