from kfp import dsl


def validate_model():
    import json

    with open("reports/metrics.json") as f:
        metrics = json.load(f)

    if metrics["accuracy"] < 0.80:
        raise Exception("Model rejected due to low accuracy")

@dsl.pipeline(name="churn-training-pipeline")
def pipeline():

    prepare = dsl.ContainerOp(
        name="prepare-data",
        image="kubeflow.azurecr.io/prepare:latest"
    )

    train = dsl.ContainerOp(
        name="train-model",
        image="kubeflow.azurecr.io/train:latest"
    ).after(prepare)

    evaluate = dsl.ContainerOp(
        name="evaluate-model",
        image="kubeflow.azurecr.io/evaluate:latest"
    ).after(train)

    validate = dsl.ContainerOp(
        name="validate-model",
        image="kubeflow.azurecr.io/evaluate:latest",
        command=["python", "-c"],
        arguments=["from validate import validate_model; validate_model()"]
    ).after(evaluate)
    
    deploy = dsl.ContainerOp(
        name="deploy-model",
        image="bitnami/kubectl",
        command=["kubectl"],
        arguments=["apply", "-f", "/manifests/deployment.yaml"]
    ).after(validate)