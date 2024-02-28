# position self attn

```
x = [B, P, C]
#                                    C = h*c
qkv = [B, P, C] -> [B, P, 3*C] -> [B, P, 3, h, c] -> [3, B, h, P, c] -> 3 * [B, h, P, c]

attn = q @ k.T = [B, h, P, c] @ [B, h, c, P] = [B, h, P, P]
attn = softmax([B, h, P, P], dim=-1)

y = attn @ v = [B, h, P, P] @ [B, h, P, c] = [B, h, P, c]
y = [B, h, P, c] -> [B, P, h, c] -> [B, P, C]
y = projection([B, P, C]) = [B, P, C]
```

# channel self attn

```
x = [B, P, C]
#                                    C = h*c
qkv = [B, P, C] -> [B, P, 3*C] -> [B, P, 3, h, c] -> [3, B, h, c, P] -> 3 * [B, h, c, P]

attn = q @ k.T = [B, h, c, P] @ [B, h, P, c] = [B, h, c, c]
attn = softmax([B, h, c, c], dim=-1)

y = attn @ v = [B, h, c, c] @ [B, h, c, P] = [B, h, c, P]
y = [B, h, c, P] -> [B, P, h, c] -> [B, P, C]
y = projection([B, P, C]) = [B, P, C]
```

