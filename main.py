import vllm  # needs to be imported first

import datasets
import math
import math_verify
import neptune
import numpy as np
import peft
import re
import time
import torch
import transformers


def accuracy_reward_fn(numbers, full_solution):
    try:
        parsed_solution = math_verify.parse(full_solution)
        correct_value = math_verify.verify(math_verify.parse('24'), parsed_solution)

        numbers_in_solution = [int(x) for x in re.findall(r'\d+', parsed_solution[1])]
        all_numbers_used = sorted(numbers) == sorted(numbers_in_solution)

        return 1. if all_numbers_used and correct_value else 0.
    except:
        return 0.


n_iterations = 1500
n_updates = 2
learning_rate = 1e-4
n_generations = 8
prompt_batch_size = 4
train_batch_size = 2
beta = .02
ema_decay = .99
eps = .1
device = 'cuda'
warmup_iterations = 2
weight_decay = .2
max_seq_len = 3072
temperature = .6
model_name = 'tiiuae/Falcon3-7B-Instruct'
base_prompt = open('base_prompt.txt').read()

model = transformers.AutoModelForCausalLM.from_pretrained(model_name, device_map=device)
model.gradient_checkpointing_enable()
lora_config = peft.LoraConfig(r=384, lora_alpha=384, lora_dropout=.0, bias='none', task_type='CAUSAL_LM')
model = peft.get_peft_model(model, lora_config, adapter_name='policy')
model.add_adapter('target', lora_config)
policy_parameters = [p for k, p in model.named_parameters() if k.count('policy') > 0]
target_parameters = [p for k, p in model.named_parameters() if k.count('target') > 0]

with torch.no_grad():
    for target_parameter, online_parameter in zip(target_parameters, policy_parameters):
        target_parameter.data = online_parameter.data

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(.9, .99), weight_decay=weight_decay)
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_iterations)
decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, n_iterations * n_updates, learning_rate * .01)
lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_iterations])

data_collator = transformers.DataCollatorWithPadding(tokenizer=tokenizer)

def format_prompt(ex):
    problem = 'Solve ' + ', '.join([str(a) for a in ex['numbers']])
    prompt = base_prompt + '\n' + problem

    formatted_prompt = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}], tokenize=False, add_generation_prompt=True)

    # ex['numbers'] = torch.tensor(ex['numbers'])
    ex['prompt'] = prompt
    ex['formatted_prompt'] = formatted_prompt
    return ex

dataset = datasets.load_dataset('nlile/24-game', split='train').map(format_prompt, remove_columns=['solutions']).train_test_split(test_size=50, seed=3000)
n_repeat = math.ceil(n_iterations / len(dataset['train']))
train_dataset = dataset['train'].repeat(n_repeat)
train_dataset.set_format("pt", columns=["numbers"], output_all_columns=True)
val_dataset = dataset['test']
val_dataset.set_format("pt", columns=["numbers"], output_all_columns=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=prompt_batch_size, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=len(val_dataset))

vllm_model = vllm.LLM(model=model_name, gpu_memory_utilization=.35)
sampling_params = vllm.SamplingParams(temperature=temperature, max_tokens=max_seq_len, n=n_generations, skip_special_tokens=False)
val_sampling_params = vllm.SamplingParams(temperature=0., max_tokens=max_seq_len, n=1, skip_special_tokens=False)

neptune_project = ''
neptune_api_token = ''
if neptune_project != '':
    run = neptune.init_run(
        project=neptune_project,
        api_token=neptune_api_token,
    )
else:
    run = None
successful_samples = dict()

for iteration, train_batch in enumerate(train_loader):
    outputs = vllm_model.generate(train_batch['formatted_prompt'], sampling_params)

    if iteration % 10 == 0:
        val_reward_list = []
        for val_iteration, val_batch in enumerate(val_loader):
            val_outputs = vllm_model.generate(val_batch['formatted_prompt'], val_sampling_params)
            val_reward_list.extend([accuracy_reward_fn(n, o.outputs[0].text) for n, o in zip(val_batch['numbers'], val_outputs)])

        val_reward = np.mean(val_reward_list)
        if run:
            run["val/rewards"].append(val_reward)
        print(f'Iteration {iteration}: Validation reward = {val_reward:.3f}')

    inputs = []
    rewards = []
    prompt_lengths = []
    full_lengths = []
    
    for numbers, output in zip(train_batch['numbers'], outputs):
        for completion in output.outputs:
            prompt_lengths.append(len(output.prompt_token_ids))
            full_lengths.append(len(output.prompt_token_ids) + len(completion.token_ids))
            inputs.append({'input_ids': list(output.prompt_token_ids) + list(completion.token_ids)})

            accuracy_reward = accuracy_reward_fn(numbers, completion.text)
            extracted = math_verify.parse(completion.text)
            with open('logs.txt', 'a') as f:
                f.write(str(numbers) + ' ' + str(extracted) + ' ' + f' -- {accuracy_reward}\n')
                f.write(completion.text + '\n\n')
            rewards.append(accuracy_reward)

    prompt_lengths = torch.tensor(prompt_lengths)
    full_lengths = torch.tensor(full_lengths)
    inputs = data_collator(inputs).to(device)
    seq_len = inputs['input_ids'].shape[1]

    prompt_mask = (prompt_lengths[:, None] > torch.arange(seq_len)[None, :]).float()
    completion_mask = torch.logical_and(prompt_lengths[:, None] <= torch.arange(seq_len), full_lengths[:, None] > torch.arange(seq_len)[None, :]).float()

    shifted_labels = inputs['input_ids'][:, 1:]
    rewards = torch.tensor(rewards).reshape((prompt_batch_size, n_generations)).to(device)
    baseline = rewards.mean(-1, keepdim=True)
    advantage = ((rewards - baseline) / (rewards - baseline).std(-1, keepdim=True).clamp(min=.001)).reshape(-1)
    if iteration % 10 == 0:
        print(numbers.numpy(), completion.text)
    print(rewards.detach().cpu().numpy(), rewards.mean().item())
    if run:
        run["train/rewards"].append(rewards.mean().item())

    model.train()
    old_logprob_list = []
    loss_list = []
    rl_loss_list = []
    kl_loss_list = []
    ratio_list = []
    for update_idx in range(n_updates):
        for idx in range(0, advantage.shape[0], train_batch_size):
            batch_input_ids = inputs['input_ids'][idx:idx + train_batch_size].to(device)
            batch_attention_mask = inputs['attention_mask'][idx:idx + train_batch_size].to(device)
            batch_completion_mask = completion_mask[idx:idx + train_batch_size].to(device)
            batch_shifted_labels = shifted_labels[idx:idx + train_batch_size].to(device)
            batch_advantage = advantage[idx:idx + train_batch_size][..., None].to(device)

            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                model.set_adapter('target')
                with torch.no_grad():
                    ref_output = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                    _, seq_len, n_vocab = ref_output.logits.shape
                    ref_selected_logprob = -torch.nn.functional.cross_entropy(input=ref_output.logits[:, :-1].transpose(1, 2) / temperature, target=batch_shifted_labels, reduction='none')

                    del ref_output
                    torch.cuda.empty_cache()

                model.set_adapter('policy')
                output = model(input_ids=batch_input_ids, attention_mask=batch_attention_mask)
                selected_logprob = -torch.nn.functional.cross_entropy(input=output.logits[:, :-1].transpose(1, 2) / temperature, target=batch_shifted_labels, reduction='none')
                if update_idx == 0:
                    old_logprob_list.append(selected_logprob.detach())
                old_logprob = old_logprob_list[idx // train_batch_size]

                n_valid_tokens = batch_completion_mask[:, 1:].sum(-1)
                log_ratio = selected_logprob - old_logprob
                ratio = log_ratio.exp()
                ratio_list.append(ratio.min().item())
                ratio_list.append(ratio.max().item())

                rl_loss = -(torch.sum(batch_completion_mask[:, 1:] * torch.min(ratio * batch_advantage, ratio.clamp(min=1 - eps, max=1 + eps) * batch_advantage), -1) / n_valid_tokens).mean()
                d_kl_loss = (torch.sum(batch_completion_mask[:, 1:] * ((ref_selected_logprob - selected_logprob).exp() - ref_selected_logprob + selected_logprob - 1), -1) / n_valid_tokens).mean()
                rl_loss_list.append(rl_loss.item())
                kl_loss_list.append(d_kl_loss.item())

                loss = rl_loss + beta * d_kl_loss
                loss_list.append(loss.item())
            loss.backward()
            torch.cuda.empty_cache()
        torch.nn.utils.clip_grad_norm_(model.parameters(), .1)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        with torch.no_grad():
            for target_parameter, online_parameter in zip(target_parameters, policy_parameters):
                target_parameter.data = ema_decay * target_parameter.data + (1. - ema_decay) * online_parameter.data
        torch.cuda.empty_cache()

    print(f'loss: {np.mean(loss_list):.4f} -- rl loss: {np.mean(rl_loss_list):.4f} -- kl loss: {np.mean(kl_loss_list):.4f} -- min ratio: {np.min(ratio_list):.4f} -- max ratio: {np.max(ratio_list):.4f}')
    print(f'seq len: {seq_len} -- prompt len: {prompt_lengths[0]} -- lr: {lr_scheduler.get_last_lr()[0]:.4e}')
    if run:
        run["train/loss"].append(np.mean(loss_list))
        run["train/rl_loss"].append(np.mean(rl_loss_list))
        run["train/kl_loss"].append(np.mean(kl_loss_list))
        run["train/response_length"].append(n_valid_tokens.float().mean().item())
        run["train/lr"].append(lr_scheduler.get_last_lr()[0])

    t1 = time.time()
    model.merge_adapter()
    state_dict = dict()
    for k, v in model.get_base_model().state_dict().items():
        if k.count('lora') == 0:
            state_dict[k.replace('base_layer.', '')] = v
    t_vllm_model = vllm_model.llm_engine.model_executor.driver_worker.model_runner.model
    t_vllm_model.load_weights(state_dict.items())
    model.unmerge_adapter()

    if iteration % 100 == 0:
        model.push_to_hub('Falcon3-7B-Instruct-r384-24-game', private=True)

    if iteration >= n_iterations:
        break
print('Finished training')
model.push_to_hub('Falcon3-7B-Instruct-r384-24-game', private=True)
