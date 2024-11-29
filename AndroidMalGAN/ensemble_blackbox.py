import torch


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ensemble_detector(model_type='', test_data=None):
    blackboxes = [{'name': 'rf', 'path': f'rf_{model_type}_model.pth'}, {'name': 'dt', 'path': f'dt_{model_type}_model.pth'},
     {'name': 'svm', 'path': f'svm_{model_type}_model.pth'}, {'name': 'knn', 'path': f'knn_{model_type}_model.pth'},
     {'name': 'gnb', 'path': f'gnb_{model_type}_model.pth'}, {'name': 'lr', 'path': f'lr_{model_type}_model.pth'},
     {'name': 'mlp', 'path': f'mlp_{model_type}_model.pth'}]

    bb_models = []
    test_data = test_data.to(DEVICE)

    for bb in blackboxes:
        blackbox = torch.load(bb['path'])
        blackbox = blackbox.to(DEVICE)
        bb_models.append(blackbox)

    combined = []
    for bb in range(len(bb_models)):
        results = bb_models[bb].predict_proba(test_data)
        # if svm
        if bb == 2:
            results = [[0.0, 1.0] if result == 1 else [1.0, 0.0] for result in results]
        combined.append(results)
    combined_results = []
    for results in combined:
        combined_results = [list(a) for a in zip(combined_results, results)]

    final = []
    for sample in combined_results:
        mal = 0
        ben = 0
        for result in sample:
            if result[0] < 0.5:
                ben += 1
            else:
                mal += 1
        if ben > mal:
            final.append([0.0, 1.0])
        else:
            final.append([1.0, 0.0])

    return final
