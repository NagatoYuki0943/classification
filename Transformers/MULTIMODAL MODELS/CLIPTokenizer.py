import torch
from transformers import CLIPTokenizer


version = "openai/clip-vit-base-patch32"
sequence = "The quick brown fox jumps over the lazy dog."
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


tokenlizer: CLIPTokenizer = CLIPTokenizer.from_pretrained(version)
print(tokenlizer.all_special_ids)       # [49406, 49407, 49407]
print(tokenlizer.all_special_tokens)    # ['<|startoftext|>', '<|endoftext|>', '<|endoftext|>']

tokens = tokenlizer.tokenize(sequence)
print(tokens)
# ['the</w>', 'quick</w>', 'brown</w>', 'fox</w>', 'jumps</w>', 'over</w>', 'the</w>', 'lazy</w>', 'dog</w>', '.</w>']

print(tokenlizer.convert_tokens_to_ids(tokens))                 # [518, 3712, 2866, 3240, 18911, 962, 518, 10753, 1929, 269]

print(tokenlizer.encode(sequence, add_special_tokens=False))    # [518, 3712, 2866, 3240, 18911, 962, 518, 10753, 1929, 269]
ids = tokenlizer.encode(sequence, add_special_tokens=True)
print(ids)                                                      # [49406, 518, 3712, 2866, 3240, 18911, 962, 518, 10753, 1929, 269, 49407]

print(tokenlizer.convert_ids_to_tokens(ids))
# ['<|startoftext|>', 'the</w>', 'quick</w>', 'brown</w>', 'fox</w>', 'jumps</w>', 'over</w>', 'the</w>', 'lazy</w>', 'dog</w>', '.</w>', '<|endoftext|>']
tokens = tokenlizer.convert_ids_to_tokens(ids, skip_special_tokens=True)
print(tokens)
# ['the</w>', 'quick</w>', 'brown</w>', 'fox</w>', 'jumps</w>', 'over</w>', 'the</w>', 'lazy</w>', 'dog</w>', '.</w>']

print(tokenlizer.convert_tokens_to_string(tokens))              # the quick brown fox jumps over the lazy dog .

print(tokenlizer.decode(ids, skip_special_tokens=False))        # <|startoftext|>the quick brown fox jumps over the lazy dog. <|endoftext|>
print(tokenlizer.decode(ids, skip_special_tokens=True))         # the quick brown fox jumps over the lazy dog.
print(tokenlizer.batch_decode([ids], skip_special_tokens=True)) # ['the quick brown fox jumps over the lazy dog.']

print(
    tokenlizer(
        text=sequence,
        padding=True,
        add_special_tokens=True,
        return_length=True,
        return_tensors="pt"
    ).to(device)
)
# {'input_ids': tensor([[49406,   518,  3712,  2866,  3240, 18911,   962,   518, 10753,  1929, 269, 49407]], device='cuda:0'),
#  'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]], device='cuda:0'), 'length': tensor([12], device='cuda:0')}
