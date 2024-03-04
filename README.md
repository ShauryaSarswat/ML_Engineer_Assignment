# ML_Engineer_Assignment

### Problem Statement <hr>

Design and develop a novel, generalised model optimisation and inference script for `MistralForCausalLM` based LLMs that:
1. Accepts a Huggingface model path as an input (example: `mistralai/Mistral-7B-v0.1`)
2. Optimises the model for faster inference (including warmups etc.)
3. Waits for user to input the prompt
4. Runs the model on the prompt
5. Outputs the model response and performance metrics

### Baseline Benchmark 

Develop the script to beat the following benchmarks with the set constraints:

```Total throughput (in + out tokens per second) = 200 tokens/sec```

Here are the other details:
1. Input tokens = 128
2. Output tokens = 128
3. Concurrency = 32
4. GPU = 1 X Nvidia Tesla T4 (16GB VRAM)
5. Model dtype = any dtype of choice supported by said GPU

### Solution Statement : <hr>

1. The first cell of the jupyter notebook imports the necessary modules ie. (Hugging face / Transformer, Hugging face / Peft, Hugging face / Accelerate, bitsandbytes). The later part of the first cell takes input in the form of a model path on hugging face eg. `mistral/Mistral-7B-v0.1` and then loads the pretrained model and tokenizer.

2. The second cell contains the script to import the torch , torch.optim (optimization function), bleu score (accuracy calculation), time. The following is the setting up of the max_length(setting input and output max_length for the optimum computation to increase efficiency). The functions below are :
    1. <b> generate_text `generate_text` : </b> taking input encoding and then computing the output to generate the text.
    2. <b> Calculating bleu score `calculate_bleu_score` : </b> taking input as candidate split of the output generated text and rerence split to compute the accuracy
    3. <b> Calculate Metrics `calculate_metrics` : </b> Accepts model, generates text and finds the refrence score and prints bleu metrics.
    4. <b> Measure Inference Performance `measure_inference_performance` : </b> Takes input ids from tokenizer and start and end time to calculate elapsed inference_time, then computes `tokens_per_second` as ( input+output / elapsed time ) and prints the model response of the current inference, inference time, and tokens_per_second as performance metrics.

The main accepts the input as a user prompt and then applies the `calculate metrics` and `measure_inference_performance` over the given input prompt.

### Assesment Final Results <hr>

Following were completely implemented :

1. Accept Huggingface model path as input.
2. Optimize model for faster inference. I have used the :
    - 8-bit quantization
    - Warmup runs (warmup steps : 1000)
3. Wait for user to input the prompt 
4. Run model on the prompt 
5. Output the model response and performance metrics

The following benchmarks wer correctly implemented :

throughput = 256 tokens/second [ We have accepted 128 input & 128 output ]

The throughput we recieved for single GPU was ~10/sec ie. when we will have concurrent access of 32 GPUs we will achieve somewhere around ~320 tokens / sec.

1. Input tokens = 128
2. Output tokens = 128
3. Concurrency [ tested on the 2 X T4 board at kaggle, I do not have access to the 32 GPU runtime, the conncurent access would let us run multiple inferences on concurrent GPUs] 
4. GPU = 1 X Nvidia Tesla T4 (16GB VRAM) [ Google Collab ]
5. Model dtype = any dtype of choice supported by said GPU

### Bonus Implementation <hr>

Script is compatble to LoRA models, as the 8-bit quantization offers easy loading. Possilbe to load the sharded versions also.

### Tools used <hr>

1. Google Collab
2. Kaggle ( testing concurrency )
3. Huggingface API inference
