from __future__ import absolute_import, division, print_function

import glob
import logging
import os
import random
import json
import math

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
import random
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange #we should change here later
from tensorboardX import SummaryWriter

from transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer, BertModel, BertForMaskedLM,
                                  XLMConfig, XLMForSequenceClassification, XLMTokenizer, 
                                  XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
                                  RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                                  AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer, AlbertModel, AlbertForMaskedLM,
                                  AutoModel, AutoTokenizer)

from transformers import AdamW, get_linear_schedule_with_warmup

from utils_pretrain import (convert_examples_to_features,
                        output_modes, processors)

import torch.nn as nn

from addDiseaseNameList import addDiseaseNameList 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

args = {
    'data_dir': 'data/',
    'model_type':  'bert',#'albert'
    'model_name': 'bert-base-uncased',#'albert-xxlarge-v2'
    'task_name': 'binary',
    'output_dir': 'outputs_pretrain/',
    'cache_dir': 'cache/',
    'do_train': True,
    'do_eval': True,
    'fp16': False,# we have to set it as False
    'fp16_opt_level': 'O1',
    'max_seq_length': 128,
    'output_mode': 'classification',
    'train_batch_size': 12,
    'eval_batch_size': 12,

    'gradient_accumulation_steps': 1,
    'num_train_epochs': 20,
    'weight_decay': 0,
    'learning_rate': 1e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,

    'logging_steps': 50,
    'evaluate_during_training': False,
    'save_steps': 1219,
    'eval_all_checkpoints': True,

    'overwrite_output_dir': False,
    'reprocess_input_data': True,
    'notes': 'Using Yelp Reviews dataset'
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open('args.json', 'w') as f:
    json.dump(args, f)

if os.path.exists(args['output_dir']) and os.listdir(args['output_dir']) and args['do_train'] and not args['overwrite_output_dir']:
    raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args['output_dir']))

MODEL_CLASSES = {
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),#BertForSequenceClassification
    'xlnet': (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
    'xlm': (XLMConfig, XLMForSequenceClassification, XLMTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'albert': (AlbertConfig, AlbertForMaskedLM, AlbertTokenizer)#AlbertForSequenceClassification
}

config_class, model_class, tokenizer_class = MODEL_CLASSES[args['model_type']]

config = config_class.from_pretrained(args['model_name'], output_hidden_states=True)#args['task_name']
tokenizer = tokenizer_class.from_pretrained(args['model_name'])
### we take BERT as an example here, if you want infuse disease knowledge into other BERT-like models, you could change here like:
#tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')

model = model_class.from_pretrained(args['model_name'], config=config)
### we take BERT as an example here, if you want infuse disease knowledge into other BERT-like models, you could change here like:
#model = model_class.from_pretrained('allenai/scibert_scivocab_uncased', output_hidden_states=True)

# We use new token [blank] to replace the disease tokens that may appear in the passage. 
# Our core idea is to BERT infer the disease and aspect from a passage.
# If the ground-truth disease tokens appear in the passage, they will make it much easier 
# for BERT to infer the disease from the passage and lower the performance.
tokenizer.add_tokens('[blank]') 
model.resize_token_embeddings(len(tokenizer))

model.to(device)

task = args['task_name']

if task in processors.keys() and task in output_modes.keys():
    processor = processors[task]()
    label_list = processor.get_labels()
    num_labels = len(label_list)
else:
    raise KeyError(f'{task} not found in processors or in output_modes. Please check utils.py.')

def load_and_cache_examples(task, tokenizer, mode='train'):
	processor = processors[task]()
	output_mode = args['output_mode']
	#mode = 'dev' if evaluate else 'train'
	cached_features_file = os.path.join(args['data_dir'], f"cached_{mode}_{args['model_name']}_{args['max_seq_length']}_{task}")

	if os.path.exists(cached_features_file) and not args['reprocess_input_data']:
		logger.info("Loading features from cached file %s", cached_features_file)
		features = torch.load(cached_features_file)

	else:
		logger.info("Creating features from dataset file at %s", args['data_dir'])
		label_list = processor.get_labels()
		if mode=='train':
			examples = processor.get_train_examples(args['data_dir'])
		elif mode=='test':
			examples = processor.get_test_examples(args['data_dir'])
		elif mode=='dev':
			examples = processor.get_dev_examples(args['data_dir']) 
		elif mode=='pretrain':
			examples = processor.get_maskedLM_examples(args['data_dir']) 

		if __name__ == "__main__":
			features = convert_examples_to_features(examples, label_list, args['max_seq_length'], tokenizer, output_mode,
                cls_token_at_end=bool(args['model_type'] in ['xlnet']),            # xlnet has a cls token at the end
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if args['model_type'] in ['xlnet'] else 0,
                sep_token=tokenizer.sep_token,
                sep_token_extra=bool(args['model_type'] in ['roberta']),           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                pad_on_left=bool(args['model_type'] in ['xlnet']),                 # pad on the left for xlnet
                pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
                pad_token_segment_id=4 if args['model_type'] in ['xlnet'] else 0)
		logger.info("Saving features into cached file %s", cached_features_file)
		torch.save(features, cached_features_file)

	all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
	all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
	all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
	all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
	all_diseaseName_ids = torch.tensor([f.diseaseName_id for f in features], dtype=torch.long)

	# if output_mode == "classification":
	# 	all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
	# elif output_mode == "regression":
	# 	all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
	dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_diseaseName_ids)
	return dataset

def removeZeros(L):
    while L[-1] == 0:
        L.pop(-1)
    return L

def train(train_dataset, model, tokenizer):
    tb_writer = SummaryWriter()
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args['train_batch_size'])
    
    t_total = len(train_dataloader) // args['gradient_accumulation_steps'] * args['num_train_epochs']
    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args['weight_decay']},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    warmup_steps = math.ceil(t_total * args['warmup_ratio'])
    args['warmup_steps'] = warmup_steps if args['warmup_steps'] == 0 else args['warmup_steps']
    
    optimizer = AdamW(optimizer_grouped_parameters, lr=args['learning_rate'], eps=args['adam_epsilon'])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args['warmup_steps'], num_training_steps=t_total)

    if args['fp16']:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args['fp16_opt_level'])
        
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args['num_train_epochs'])
    logger.info("  Total train batch size  = %d", args['train_batch_size'])
    logger.info("  Gradient Accumulation steps = %d", args['gradient_accumulation_steps'])
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(int(args['num_train_epochs']), desc="Epoch")
    
    for _ in train_iterator:
        loss_epoch = 0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            maskTokenPositionLists = []
            realTokens = []
            for i in range(len(batch[0])):
                maskTokenPosition = []
                realToken = []
                for j in range(0, args['max_seq_length']):
                    if batch[0][i][j]!=0:
                        if batch[0][i][j]==tokenizer.convert_tokens_to_ids('[MASK]'):
                            maskTokenPosition.append(j)
                            realToken.append(batch[3][i][j])
                        # It is found to us that randomly (with 75% probability) recover [blank] token with true disease tokens will boost performance.
                        if batch[0][i][j]==tokenizer.convert_tokens_to_ids('[blank]'):
                            if random.random()<0.75:
                                batch[0][i][j] = batch[3][i][j]
                                batch[3][i][j] = -100
                            else:
                                batch[3][i][j] = -100
                maskTokenPositionLists.append(maskTokenPosition)
                realTokens.append(realToken)
            batch = tuple(t.to(device) for t in batch) 


            inputs = {'input_ids':      batch[0],
                      #'attention_mask': batch[1],
                      #'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet', 'albert'] else None,  # XLM don't use segment_ids
                      'masked_lm_labels':  batch[3]}#'labels':         batch[3]

            outputs = model(**inputs)

            loss_mlm = outputs[0]

            last_hidden_states = outputs[2][-1]
            #print(last_hidden_states.size())
            CLS_hidden_state = last_hidden_states[:, 0]
            #print('size of CLS_hidden_stat: ', CLS_hidden_state.size())

            # Section 3.3 in the paper: 
            # Note that the vocabulary size of BERT is around
            # 30,000 which means masked language modeling
            # task is a 30,000 multi-class problem. The logits
            # after the normalization of softmax (Equation 2) 
            # will be pretty small (the expectation of mean
            # should be around 1/30,000=3.3*e-5), which might
            # cause some obstacles for the learning. Therefore,
            # we also maximize the raw logits (like 𝑧𝑡) before
            # softmax normalization which might keep more useful information.
            prediction_scores = outputs[1]

            logits_score_total = torch.tensor([1], dtype=torch.float)
            logits_score_total = logits_score_total.to(device)
            for i in range(len(prediction_scores)):
                logits_score = torch.tensor([0], dtype=torch.float)
                logits_score = logits_score.to(device)
                prediction_score = prediction_scores[i]
                maskTokenPosition = maskTokenPositionLists[i]#[1:]
                tokenIds = removeZeros(batch[4][i].tolist())

                if len(maskTokenPosition)==len(tokenIds):
                    for j in range(len(maskTokenPosition)):
                        logits_score += prediction_score[maskTokenPosition[j]][tokenIds[j]]

                elif len(maskTokenPosition)==len(tokenIds)+1:
                    maskTokenPosition = maskTokenPosition[1:]
                    for j in range(len(maskTokenPosition)):
                        logits_score += prediction_score[maskTokenPosition[j]][tokenIds[j]]

                elif len(maskTokenPosition)==len(tokenIds)+2:
                    maskTokenPosition = maskTokenPosition[2:]
                    for j in range(len(maskTokenPosition)):
                        logits_score += prediction_score[maskTokenPosition[j]][tokenIds[j]]

                logits_score_total += logits_score
            logits_score_total = 1.0*len(prediction_scores)*10/logits_score_total
            logits_score_total = logits_score_total.to(device)

            loss = loss_mlm + logits_score_total

            loss_epoch += loss
           
            print("\r%f" % loss, end='')

            if args['gradient_accumulation_steps'] > 1:
                loss = loss / args['gradient_accumulation_steps']

            if args['fp16']:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args['max_grad_norm'])
                
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args['max_grad_norm'])

            tr_loss += loss.item()
            if (step + 1) % args['gradient_accumulation_steps'] == 0:
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args['logging_steps'] > 0 and global_step % args['logging_steps'] == 0:
                    # Log metrics
                    if args['evaluate_during_training']:  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args['logging_steps'], global_step)
                    logging_loss = tr_loss

                if args['save_steps'] > 0 and global_step % args['save_steps'] == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args['output_dir'], 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    # tower_PATH = output_dir + 'tower.dict'
                    # torch.save(towerModel.state_dict(), tower_PATH)
                    logger.info("Saving model checkpoint to %s", output_dir)
        print('average loss of this epoch: ', loss_epoch*1.0/step)


    return global_step, tr_loss / global_step

def diseaseKnowledgeInfusionTraining():
    train_dataset = load_and_cache_examples(task, tokenizer, mode='pretrain')
    global_step, tr_loss = train(train_dataset, model, tokenizer)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if not os.path.exists(args['output_dir']):
            os.makedirs(args['output_dir'])
    logger.info("Saving model checkpoint to %s", args['output_dir'])
    
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args['output_dir'])
    tokenizer.save_pretrained(args['output_dir'])
    torch.save(args, os.path.join(args['output_dir'], 'training_args.bin'))

if __name__ == "__main__":
    diseaseKnowledgeInfusionTraining()
	