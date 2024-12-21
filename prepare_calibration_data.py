from datasets import load_dataset
import tiktoken
import random
import pickle
import torch

traindata =  load_dataset('json', data_files=
                            {'train': '/root/autodl-tmp/nanoGPT/data/c4_dataset/c4-train.00000-of-01024.json.gz'}, 
                            split='train',cache_dir='/root/autodl-tmp/nanoGPT/data/c4_dataset/cache/train')
enc = tiktoken.get_encoding("gpt2")
seed  = 42
seqlen = 1024

nsamples = 128
trainloader = []
for _ in range(nsamples):
    while True:
        i = random.randint(0, len(traindata) - 1)
        # transform list to tensor

        trainenc = enc.encode_ordinary(traindata[i]['text'])
        trainenc = torch.tensor(trainenc, dtype=torch.long).unsqueeze(0)
        if trainenc.shape[1] > seqlen:
            break
    i = random.randint(0, trainenc.shape[1]  - seqlen - 1)
    j = i + seqlen
    inp = trainenc[:, i:j]
    trainloader.append(inp)

# save the calibration dataset
with open('/root/autodl-tmp/nanoGPT/data/c4_dataset/calibration_dataset.pkl', 'wb') as f:
    pickle.dump(trainloader, f)
