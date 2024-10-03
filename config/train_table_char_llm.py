log_interval = 40  # don't print too often

always_save_checkpoint = True

wandb_log = False  # override via command line if you like
wandb_project = 'table-llm-char'
wandb_run_name = 'mini-gpt'

dataset = 'table_llm_char'
gradient_accumulation_steps = 1
batch_size = 256
block_size = 256  # context of up to 256 previous characters

# baby GPT model
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3  # with baby networks can afford to go a bit higher
max_iters = 250000
lr_decay_iters = 250000  # make equal to max_iters usually
min_lr = 1e-4  # learning_rate / 10 usually
beta2 = 0.99  # make a bit bigger because number of tokens per iter is small

warmup_iters = 100  # not super necessary potentially
