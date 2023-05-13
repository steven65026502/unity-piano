import os
import torch
import time
import json
import server
from masking import *
from dictionary_roll import worddict, pad_token, wordtoint, inttoword, wordtoonsetevents
import torch.nn.functional as F


start_token = worddict.index('bar')
end_token = worddict.index('endbar')

def load_model(filepath):
    from model import MusicTransformer
    from model import hparams
    from model import dev

    file = torch.load(filepath, map_location=torch.device(dev))
    if "hparams" not in file:
        file["hparams"] = hparams

    model = MusicTransformer(**file["hparams"]).to(device)
    model.load_state_dict(file["state_dict"])
    model.eval()
    return model

def nucleus_sampling(logits, p=0.5):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    logits[sorted_indices_to_remove] = float('-inf')
    probs = F.softmax(logits, dim=-1)

    return torch.multinomial(probs, num_samples=1)

def greedy_decode(model, inp, mode="nucleus_sampling", temperature=0.7, k=5, min_length=1800, p=0.5):
    # convert input tokens to list of token ids
    inp = wordtoint(inp)
    # make sure inp starts with the start token
    if inp[0] != start_token:
        inp = [start_token] + inp

    # convert to torch tensor and convert to correct dimensions for masking
    inp = torch.tensor(inp, dtype=torch.int64, device=device)
    inp = inp.unsqueeze(0)
    n = inp.dim() + 2

    # parameters for decode sampling
    if not callable(temperature):
        temperature__ = temperature
        del temperature

        def temperature(x):
            return temperature__

    if k is not None and not callable(k):
        k__ = k
        del k

        def k(x):
            return k__

    # autoregressively generate output
    try:
        with torch.no_grad():
            while True:
                # get next predicted idx
                predictions = model(inp, mask=create_mask(inp, n))
                # divide logits by temperature as a function of current length of sequence
                predictions /= temperature(inp[-1].shape[-1])

                # sample the next predicted idx
                if mode == "argmax" or k is None:
                    prediction = torch.argmax(predictions[..., -1, :], dim=-1)
                elif k is not None:
                    # get top k predictions, where k is a function of current length of sequence
                    top_k_preds = torch.topk(predictions[..., -1, :], k(inp[-1].shape[-1]), dim=-1)
                    # sample top k predictions
                    predicted_idx = torch.distributions.Categorical(logits=top_k_preds.values[..., -1, :]).sample()
                    # get the predicted id
                    prediction = top_k_preds.indices[..., predicted_idx]

                elif mode == "categorical":
                    prediction = torch.distributions.Categorical(logits=predictions[..., -1, :]).sample()

                elif mode == "nucleus_sampling":
                    prediction = nucleus_sampling(predictions[..., -1, :], p=p)

                else:
                    raise ValueError("Invalid mode or top k passed in")

                # if we reached the end token and the generated sequence has reached the minimum length, break the loop and return the result
                if inp.shape[1] >= min_length:
                    break
                # else continue generating
                else:
                    inp = torch.cat(
                        [inp, prediction.view(1, 1)],
                        dim=-1
                    )

    except (KeyboardInterrupt, RuntimeError):
        # generation takes a long time, interrupt in between to save whatever has been generated until now
        # RuntimeError is in case the model generates more tokens that there are absolute positional encodings for
        pass

    # extra batch dimension needs to be gotten rid of, so squeeze
    return inp.squeeze()


def audiate(token_ids, save_path="output.json", tempo=850000, verbose=False):
    word_sequence = inttoword(token_ids.tolist())

    # Convert the word sequence to a dictionary of onset events
    raw_onset_events = wordtoonsetevents(word_sequence)

    # Reformat the onset events
    onset_events = []
    for event in raw_onset_events['onset_events']:
        onset_events.append([event[0], event[1], event[2], event[3]])

    # Save the dictionary to a JSON file
    print(f"Saving JSON file at {save_path}...") if verbose else None
    musicjson = json.dumps({'onset_events': onset_events, "pedal_events": [], "tempo": tempo})

    print("Done")
    return musicjson

def generate(model_, inp, save_path="output_test.json", mode="nucleus_sampling", temperature=0.7, min_length=1800, p=0.5,
             tempo= 850000, verbose=False):

    # greedy decode
    print("Greedy decoding...") if verbose else None
    start = time.time()
    token_ids = greedy_decode(model=model_, inp=inp, mode=mode, temperature=temperature, min_length=min_length, p=p)
    end = time.time()
    print(f"Generated {len(token_ids)} tokens.", end=" ") if verbose else None
    print(f"Time taken: {round(end - start, 2)} secs.") if verbose else None

    # generate audio
    music_json = audiate(token_ids=token_ids, save_path=None, tempo=tempo, verbose=verbose)
    return music_json


if __name__ == "__main__":
    from model import hparams
    import socket

    # Hardcoded parameters
    save_path = "output_test.json"
    mode = "nucleus_sampling"
    verbose = True
    tempo = 850000

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 1234))
        s.listen()
        c, addr = s.accept()
        with c:
            print(addr, "connected.")
            while True:
                print("Waiting for start signal...")
                received_data = server.recv_json(c)

                if  received_data["signal"] == "start":
                    received_temperature = received_data['temperature']
                    received_p = received_data['p']
                    received_min_length = received_data['minLength']
                    received_model_name = received_data['modelName']
                    # 根據模型名稱選擇模型文件路徑
                    if received_model_name == "Model1":
                        model_path = "final_model.pt"
                    elif received_model_name == "Model2":
                        model_path = "pop_music.pt"

                    # Load the model instance from the selected model path
                    model_instance = load_model(model_path)

                    if received_temperature is not None and received_p is not None and received_min_length is not None:
                        temperature = received_temperature
                        p = received_p
                        min_length = received_min_length
                        print(f"New parameters: temperature={temperature}, p={p}, min_length={min_length}")
                        print(model_path)
                        generated_music_json = generate(model_=model_instance, inp=["bar"], save_path=save_path,
                                                        temperature=temperature, p=p, mode=mode, min_length=min_length,
                                                        verbose=verbose)
                        server.send_json(c, generated_music_json)
