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
#from tensorboardX import SummaryWriter
import addTokens


from transformers import (WEIGHTS_NAME, BertConfig, BertForSequenceClassification, BertTokenizer, BertModel, BertForMaskedLM,
                                  XLMConfig, XLMForSequenceClassification, XLMTokenizer, 
                                  XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer,
                                  RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                                  AlbertConfig, AlbertForSequenceClassification, AlbertTokenizer, AlbertModel, AlbertForMaskedLM)

from transformers import AdamW, get_linear_schedule_with_warmup

from utils import (convert_examples_to_features,
                        output_modes, processors)

import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

args = {
    'data_dir': 'data/',
    'model_type':  'bert',#'albert'
    'model_name': 'bert-base-uncased',#'albert-xxlarge-v2'
    'task_name': 'regression',
    'output_dir': 'outputs-test/',
    'cache_dir': 'cache/',
    'do_train': True,
    'do_eval': True,
    'fp16': False,# we have to set it as False
    'fp16_opt_level': 'O1',
    'max_seq_length': 128,
    'output_mode': 'regression',
    'train_batch_size': 8,
    'eval_batch_size': 8,

    'gradient_accumulation_steps': 2,
    'num_train_epochs': 20,
    'weight_decay': 0,
    'learning_rate': 1e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,

    'logging_steps': 50,
    'evaluate_during_training': False,
    'save_steps': 213,
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
#model = model_class.from_pretrained(args['model_name'], config=config)
#tokenizer = addTokens.addDiseaseTokens('data/extractedQuestionAnswers_total_masked100_maskedLM.txt')
#model = model_class.from_pretrained('/home/ubuntu/BERT-multiClass/outputs_pretrain/checkpoint-2438')

### change path here for your BERT model pretrained by disease_knowledge_infusion_training.py
model = model_class.from_pretrained('/home/infolab/env_py3_yunhe/BERT_LM_logits_aws/outputs_pretrain/checkpoint-2438')
model.to(device)

class regressionLayer(nn.Module):
    def __init__(self, size):
        super(regressionLayer, self).__init__() #the super class of Net is nn.Module, this "super" keywords can search methods in its super class
        self.f1 = nn.Linear(size, 1)

    def forward(self, x):
        x = self.f1(x)
        return x

regressionLayerMedIQA = regressionLayer(model.config.hidden_size)

regressionLayerMedIQA.to(device)
#model.to(device)

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

	if output_mode == "classification":
		all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
	elif output_mode == "regression":
		all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
	dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
	return dataset

def train(train_dataset, model, towerModel, tokenizer):
    #tb_writer = SummaryWriter()
    
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
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
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
            batch = tuple(t.to(device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      }#'labels':         batch[3]
            outputs = model(**inputs)
            last_hidden_states = outputs[1][-1]
            print(last_hidden_states.size())
            CLS_hidden_state = last_hidden_states[:, 0]
            #print(CLS_hidden_state.size())
            logits = towerModel(CLS_hidden_state)
            #logits = torch.squeeze(logits)
            logits = logits[:, 0]
            #print(logits.size())
            #print(output)
            #output = nn.Linear(model.config.hidden_size, 3)(CLS_hidden_state)
            labels = batch[3]
            #print(labels.size())
            loss = criterion(logits, labels)
            loss_epoch += loss
            #loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)
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
                    # if args['evaluate_during_training']:  # Only evaluate when single GPU otherwise metrics may not average well
                    #     results, _ = evaluate(model, tokenizer)
                    #     for key, value in results.items():
                    #         tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    # tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    # tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args['logging_steps'], global_step)
                    logging_loss = tr_loss

                if args['save_steps'] > 0 and global_step % args['save_steps'] == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args['output_dir'], 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tower_PATH = output_dir + 'tower.dict'
                    torch.save(towerModel.state_dict(), tower_PATH)
                    logger.info("Saving model checkpoint to %s", output_dir)
        print('average loss of this epoch: ', loss_epoch*1.0/step)

    return global_step, tr_loss / global_step

from sklearn.metrics import mean_squared_error, matthews_corrcoef, confusion_matrix
from scipy.stats import pearsonr

def get_mismatched(labels, preds):
    mismatched = labels != preds
    examples = processor.get_dev_examples(args['data_dir'])
    wrong = [i for (i, v) in zip(examples, mismatched) if v]
    
    return wrong

def get_eval_report(labels, preds):
    # mcc = matthews_corrcoef(labels, preds)
    # mismatched = labels != preds
    # count_right = 0
    # for item in mismatched:
    #     if item==0:
    #         count_right += 1
    # acc = count_right*1.0/len(labels)
    acc = (preds == labels).mean()
    #tn, fp, fn, tp = 0, 0, 0, 0#confusion_matrix(labels, preds).ravel()
    return {
        "acc": acc,
        # "tp": tp,
        # "tn": tn,
        # "fp": fp,
        # "fn": fn
    }#, get_mismatched(labels, preds)

def compute_metrics(task_name, preds, labels):
    print(preds[:100])
    assert len(preds) == len(labels)
    return get_eval_report(labels, preds)

def evaluate(eval_dataset, model, towerModel, tokenizer, mode, checkpoint, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args['output_dir']

    results = {}
    
    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)


    #eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args['eval_batch_size'], shuffle=False)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args['eval_batch_size'])
    print('checkpoint: ', checkpoint)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    #criterion = nn.CrossEntropyLoss()
    criterion = nn.MSELoss()
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args['model_type'] in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      }#'labels':         batch[3]
            outputs = model(**inputs)
            last_hidden_states = outputs[1][-1]
            #print(last_hidden_states.size())
            CLS_hidden_state = last_hidden_states[:, 0]
            #print(CLS_hidden_state.size())
            logits = towerModel(CLS_hidden_state)
            logits = logits[:, 0]
            #logits = torch.squeeze(logits)
            #print(output)
            #output = nn.Linear(model.config.hidden_size, 3)(CLS_hidden_state)
            labels = batch[3]
            tmp_eval_loss = criterion(logits, labels)
            #tmp_eval_loss, logits = outputs[:2]

            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            #out_label_ids = inputs['labels'].detach().cpu().numpy()
            out_label_ids = labels.detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            #out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    if args['output_mode'] == "classification":
        preds = np.argmax(preds, axis=1)
    elif args['output_mode'] == "regression":
        preds = np.squeeze(preds)

    # result = compute_metrics(EVAL_TASK, preds, out_label_ids)
    # results.update(result)

    # output_eval_file = os.path.join(eval_output_dir, "eval_results.txt")
    # with open(output_eval_file, "w") as writer:
    #     logger.info("***** Eval results {} *****".format(prefix))
    #     for key in sorted(result.keys()):
    #         logger.info("  %s = %s", key, str(result[key]))
    #         writer.write("%s = %s\n" % (key, str(result[key])))

    # return results
    return preds

def final_train():
    train_dataset = load_and_cache_examples(task, tokenizer)
    global_step, tr_loss = train(train_dataset, model, regressionLayerMedIQA, tokenizer)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    if not os.path.exists(args['output_dir']):
            os.makedirs(args['output_dir'])
    logger.info("Saving model checkpoint to %s", args['output_dir'])
    
    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args['output_dir'])
    tokenizer.save_pretrained(args['output_dir'])
    torch.save(args, os.path.join(args['output_dir'], 'training_args.bin'))
    tower_PATH = args['output_dir'] + 'checkpoint-outputstower.dict'
    torch.save(regressionLayerMedIQA.state_dict(), tower_PATH)


def final_test(mode):
    from eval import generateResult
    from mediqa2019_evaluator_allTasks_final import MediqaEvaluator
    results = {}
    output_eval_file = os.path.join(args['output_dir'], "eval_results_"+mode+".txt")
    writer = open(output_eval_file, "w")

    EVAL_TASK = args['task_name']

    eval_dataset = load_and_cache_examples(EVAL_TASK, tokenizer, mode)

    checkpoints = [args['output_dir']]
    if args['eval_all_checkpoints']:
        checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args['output_dir'] + '/**/' + WEIGHTS_NAME, recursive=True)))
        logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    logger.info("Evaluate the following checkpoints: %s", checkpoints)
    for checkpoint in checkpoints:
        #print('checkpoint: ', checkpoint)
        global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
        model = model_class.from_pretrained(checkpoint)
        tower_PATH = os.path.join(args['output_dir'], 'checkpoint-{}'.format(global_step)) + 'tower.dict'
        regressionLayerMedIQA.load_state_dict(torch.load(tower_PATH))
        model.to(device)
        regressionLayerMedIQA.to(device)
        preds = evaluate(eval_dataset, model, regressionLayerMedIQA, tokenizer, mode, checkpoint, prefix=global_step)
        np.save(os.path.join(args['output_dir'], 'checkpoint-{}'.format(global_step)) + 'preds', preds)
        print('saved predictions!')
        generateResult(preds, 'data/result_'+mode+'.txt', 
            os.path.join(args['output_dir'], 'checkpoint-{}'.format(global_step))+'ranked.txt')
        if mode=='test':
            answer_file_path = 'QA_testSet_ground_truth_round_2.txt'
        else:
            answer_file_path = 'QA_validationSet_ground_truth.txt'
        _client_payload = {}
        _client_payload["submission_file_path"] = os.path.join(args['output_dir'], 'checkpoint-{}'.format(global_step))+'ranked.txt'

        # Instaiate a dummy context
        _context = {}
        # Instantiate an evaluator
        aicrowd_evaluator = MediqaEvaluator(answer_file_path, task=3)
        # Evaluate
        result = aicrowd_evaluator._evaluate(_client_payload, _context)
        print(result)
        result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
        results.update(result)

        
        
        logger.info("***** Eval results {} *****".format(global_step))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
            writer.flush()

if __name__ == "__main__":
	final_train()
	final_test('dev')
	final_test('test')