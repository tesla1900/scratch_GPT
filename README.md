# Introduction
This is a continuation of my previous series, [learning_makemore](https://github.com/tesla1900/learning_makemore), where I follow lectures of [Andrej Kaparthy](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) and build a Generative Pre-training Transformer (GPT) from scratch. In this project, we build a character-level language model using the Transformer architecture, inspired by the model powering systems like ChatGPT. We implement the transformer based model from the paper "Attention Is All You Need" paper (Vaswani et al., 2017).  While ChatGPT is a bigger model trained on vast internet data, this project uses the much smaller dataset - all the works of Shakespeare. The goal isn't to replicate ChatGPT, but to understand the Transformer neural network by building a simplified version, nanoGPT.



Biagram model output 

```
```

Output of 10M parameter model
```
10.788929 M parameters 
step 0: train loss 4.2846, val loss 4.2820 
step 500: train loss 1.8935, val loss 2.0082
step 1000: train loss 1.5323, val loss 1.7204
step 1500: train loss 1.3901, val loss 1.6002 
step 2000: train loss 1.3043, val loss 1.5434 
step 2500: train loss 1.2495, val loss 1.5220 
step 3000: train loss 1.2001, val loss 1.4974 
step 3500: train loss 1.1578, val loss 1.4798 
step 4000: train loss 1.1200, val loss 1.4778 
step 4500: train loss 1.0836, val loss 1.4777

Thy wrong davour sail, i' the commons:
Come that is no office I to be gentle climary, And am thurns there, I'll even to fall thee To lose you.
FLORIZEL:
Come Anrichmony that:
There is hot were sufformers banish comment,
That fear'd my well-heart.
I shall not be tempested town;
But that,
let honourable perpillars. For I,
And Messenger, thee-unting of opposition botterness
Shall stand bapest you city.
Clown:
Haply man! He told Hermingst the mords! applahs and yeb:
More are answer'd face withith
```


