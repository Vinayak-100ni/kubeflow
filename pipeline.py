from kfp import dsl

@dsl.pipeline(name="churn-training-pipeline")
def pipeline():

    prepare = dsl.ContainerOp(
        name="prepare-data",
        image="<registry>/prepare:latest"
    )

    train = dsl.ContainerOp(
        name="train-model",
        image="<registry>/train:latest"
    ).after(prepare)

    evaluate = dsl.ContainerOp(
        name="evaluate-model",
        image="<registry>/evaluate:latest"
    ).after(train)