from dictionary_roll import onseteventstoword, wordtoonsetevents, wordtoint, inttoword
from torch import LongTensor

def midi_parser(json_data):
    words = onseteventstoword(json_data)
    index_list = wordtoint(words)
    return LongTensor(index_list), words

def list_parser(index_list=None, event_list=None):
    if not ((index_list is None) ^ (event_list is None)):
        raise ValueError("Input one of index_list or event_list, not both or neither")

    if event_list is not None:
        index_list = wordtoint(event_list)

    event_list = inttoword(index_list)
    json_data = wordtoonsetevents(event_list)
    return json_data
