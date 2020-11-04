import numpy as np
import operator
from mediqa2019_evaluator_allTasks_final import MediqaEvaluator

def generateResult(preds, questionAnswerFile, resultSaveFile):
    # preds = np.load(predsFile)
    fp = open(questionAnswerFile)
    lines = fp.readlines()
    txt = ''
    resultDict = {}
    questionIdList = []
    for index in range(len(lines)):
        line = lines[index]
        lineStr = line.strip().split(',')
        questionId = lineStr[0]
        answerId = lineStr[1]
        if questionId not in resultDict:
            resultDict[questionId] = {}
            resultDict[questionId][answerId] = preds[index]
            questionIdList.append(questionId)
        else:
            resultDict[questionId][answerId] = preds[index]
    # questionIdMin = min(questionIdList)
    # questionIdMax = max(questionIdList)

    txt = ''
    for questionIndex in questionIdList:
        answerDict = resultDict[questionIndex]
        for answerId, score in sorted(answerDict.items(), key=lambda item: item[1], reverse=True):
            if score >= 0:
                label = '1'
            else:
                label = '0'
            txt += questionIndex+','+answerId+','+label+'\n'
    fp_w = open(resultSaveFile, 'w')
    fp_w.write(txt)
    return

if __name__ == "__main__":
    # for i in range(210, 2101, 210):
    #     generateResult('outputs/checkpoint-'+str(i)+'preds.npy',
    #'data/result.txt',
    #'outputs/checkpoint-'+str(i)+'ranked.txt')
    for i in range(210, 2101, 210):
        answer_file_path = 'QA_testSet_ground_truth_round_2.txt'
        _client_payload = {}
        _client_payload["submission_file_path"] = 'outputs/checkpoint-'+str(i)+'ranked.txt'

        # Instaiate a dummy context
        _context = {}
        # Instantiate an evaluator
        aicrowd_evaluator = MediqaEvaluator(answer_file_path, task=3)
        # Evaluate
        result = aicrowd_evaluator._evaluate(_client_payload, _context)
        print(result)