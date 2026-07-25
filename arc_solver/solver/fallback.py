def identity_predictions(task):
    return [x.copy() for x in task.test]
