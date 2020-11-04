# From the original file example_evaluator.py by Sharada Mohanty (https://github.com/AICrowd/aicrowd-example-evaluator)
# Adapted for MEDIQA 2019 by Asma Ben Abacha --Accuracy for Tasks 1 and 2 (NLI and RQE) & MRR, Accuracy, Precision, and Spearman's rank correlation coefficient for Task 3 (QA).
# Updated on May 6, 2019. 

import pandas as pd
import numpy as np
import scipy
import scipy.stats

class MediqaEvaluator:
    def __init__(self, answer_file_path, task=1, round=1):
        """
        `round` : Holds the round for which the evaluation is being done.
        can be 1, 2...upto the number of rounds the challenge has.
        Different rounds will mostly have different ground truth files.
        """
        self.answer_file_path = answer_file_path
        self.round = round
        self.task = task

    def _evaluate(self, client_payload, _context={}):
        if self.task == 1:
            return self._evaluate_task_1(client_payload, _context)
        elif self.task == 2:
            return self._evaluate_task_2(client_payload, _context)
        elif self.task == 3:
            return self._evaluate_task_3(client_payload, _context)


    def _evaluate_task_1(self, client_payload, _context={}):
        """
        `client_payload` will be a dict with (atleast) the following keys :
          - submission_file_path : local file path of the submitted file
          - aicrowd_submission_id : A unique id representing the submission
          - aicrowd_participant_id : A unique id for participant/team submitting (if enabled)
        """
        submission_file_path = client_payload["submission_file_path"]

        # Result file format: pair_id,label (csv file)

        col_names = ['pair_id', 'label']

        submission = pd.read_csv(submission_file_path, header=None, names=col_names)
        gold_truth = pd.read_csv(self.answer_file_path, header=None, names=col_names)

        # Drop duplicates except for the first occurrence.
        submission = submission.drop_duplicates(['pair_id'])

        submission.label = submission.label.astype(str)
        gold_truth.label = gold_truth.label.astype(str)

        submission['entry'] = submission.apply(lambda x: '_'.join(x), axis=1)
        gold_truth['entry'] = gold_truth.apply(lambda x: '_'.join(x), axis=1)

        s1 = submission[submission['entry'].isin(gold_truth['entry'])]

        accuracy = s1.size / gold_truth.size

        _result_object = {
            "score": accuracy,
            "score_secondary" : 0.0
        }
        return _result_object

    def _evaluate_task_2(self, client_payload, _context={}):
        """
        `client_payload` will be a dict with (atleast) the following keys :
          - submission_file_path : local file path of the submitted file
          - aicrowd_submission_id : A unique id representing the submission
          - aicrowd_participant_id : A unique id for participant/team submitting (if enabled)
        """
        submission_file_path = client_payload["submission_file_path"]

        # Result file format: pair_id,label (csv file)

        col_names = ['pair_id', 'label']

        submission = pd.read_csv(submission_file_path, header=None, names=col_names, dtype={'pair_id': str, "label": str})
        gold_truth = pd.read_csv(self.answer_file_path, header=None, names=col_names, dtype={'pair_id': str, "label": str})

        # Drop duplicates except for the first occurrence.
        submission = submission.drop_duplicates(['pair_id'])

        submission.label = submission.label.astype(str)
        gold_truth.label = gold_truth.label.astype(str)

        submission['entry'] = submission.apply(lambda x: '_'.join(x), axis=1)
        gold_truth['entry'] = gold_truth.apply(lambda x: '_'.join(x), axis=1)

        s1 = submission[submission['entry'].isin(gold_truth['entry'])]

        accuracy = s1.size / gold_truth.size

        _result_object = {
            "score": accuracy,
            "score_secondary" : 0.0
        }

        return _result_object

    def _evaluate_task_3(self, client_payload, _context={}):
        """
        `client_payload` will be a dict with (atleast) the following keys :
          - submission_file_path : local file path of the submitted file
          - aicrowd_submission_id : A unique id representing the submission
          - aicrowd_participant_id : A unique id for participant/team submitting (if enabled)
        """
        submission_file_path = client_payload["submission_file_path"]

        # Result file format: q_id,a_id,label{0/1}

        col_names = ['question_id','answer_id', 'label']

        submission = pd.read_csv(submission_file_path, header=None, names=col_names)
        gold_truth = pd.read_csv(self.answer_file_path, header=None, names=col_names)

        # Drop duplicates except for the first occurrence.
        submission = submission.drop_duplicates(['question_id', 'answer_id'])

        submission.label = submission.label.astype(str)
        gold_truth.label = gold_truth.label.astype(str)

        submission['entry'] = submission.apply(lambda x: '_'.join(map(str,x)), axis=1)
        gold_truth['entry'] = gold_truth.apply(lambda x: '_'.join(map(str,x)), axis=1)

        s1 = submission[submission['entry'].isin(gold_truth['entry'])]

        accuracy = s1.size / gold_truth.size

        question_ids = []
        correct_answers = {}
        for index, row in gold_truth.iterrows():
            qid = row['question_id']

            if qid not in question_ids:
                question_ids.append(qid)

            if row['label'] == '1':
                if qid not in correct_answers:
                    correct_answers[qid] = []

                correct_answers[qid].append(row['answer_id'])

        Pr = 0.
        spearman = 0.
        pv = 0.
        predictedPositive = 0.
        correctPredictedPositive = 0.
        mrr = 0.
        sp_nan_ignoredQs = 0

        for qid in question_ids:
            submitted_correct_answers = []
            index = 1
            first = True
            for _, row in submission[submission['question_id']==qid].iterrows():
                aid = row['answer_id']
                if row['label'] == '1':
                    predictedPositive += 1
                    if aid in correct_answers[qid]:
                        correctPredictedPositive += 1
                        submitted_correct_answers.append(aid)
                        if first:
                            mrr += 1. / index
                            first=False

                index += 1
            matched_gold_subset = []

            for x in correct_answers[qid]:
                if x in submitted_correct_answers:
                    matched_gold_subset.append(x)

            rho, p_value = scipy.stats.spearmanr(submitted_correct_answers, matched_gold_subset)
            if np.isnan(rho):
                rho = 0.0
                sp_nan_ignoredQs += 1
            spearman += rho
            pv += p_value

        question_nb = len(question_ids)
        q_nb_spearman = question_nb - sp_nan_ignoredQs
        spearman = spearman / q_nb_spearman
        Pr = correctPredictedPositive / predictedPositive
        mrr = mrr / question_nb

        if np.isnan(spearman):
            spearman = 0.0

        _result_object = {
            "accuracy": accuracy,
            "spearman": spearman,
            "meta" : {
                "MRR": mrr,
                "Precision": Pr
            }
        }
        return _result_object


if __name__ == "__main__":
    # Test Tasks 1,2,3
    for task in range(1, 4):
        print("Testing Task (Round-1) : {}".format(task))
        answer_file_path = "data/task{}/ground_truth.csv".format(task)
        _client_payload = {}
        _client_payload["submission_file_path"] = "data/task{}/sample_submission.csv".format(task)

        # Instaiate a dummy context
        _context = {}
        # Instantiate an evaluator
        aicrowd_evaluator = MediqaEvaluator(answer_file_path, task=task)
        # Evaluate
        result = aicrowd_evaluator._evaluate(_client_payload, _context)
        print(result)

    # Test Tasks 1,2,3 - Round -2
    for task in range(1, 4):
        print("Testing Task (Round-2) : {}".format(task))
        answer_file_path = "data/task{}/ground_truth_round_2.csv".format(task)
        _client_payload = {}
        _client_payload["submission_file_path"] = "data/task{}/sample_submission_round_2.csv".format(task)

        # Instaiate a dummy context
        _context = {}
        # Instantiate an evaluator
        aicrowd_evaluator = MediqaEvaluator(answer_file_path, task=task, round=2)
        # Evaluate
        result = aicrowd_evaluator._evaluate(_client_payload, _context)
        print(result)

