from kfp import dsl
from kfp.dsl import container_component

# Step 1: Prepare
@container_component
def prepare_op():
    return dsl.ContainerSpec(
        image="kubeflow.azurecr.io/prepare:latest"
    )

# Step 2: Train
@container_component
def train_op():
    return dsl.ContainerSpec(
        image="kubeflow.azurecr.io/train:latest"
    )

# Step 3: Evaluate
@container_component
def evaluate_op():
    return dsl.ContainerSpec(
        image="kubeflow.azurecr.io/evaluate:latest"
    )

# Step 4: Validate
@container_component
def validate_op():
    return dsl.ContainerSpec(
        image="kubeflow.azurecr.io/evaluate:latest",
        command=["python", "-c"],
        args=["print('Validation step placeholder')"]
    )

# Step 5: Deploy
@container_component
def deploy_op():
    return dsl.ContainerSpec(
        image="bitnami/kubectl",
        command=["kubectl"],
        args=["apply", "-f", "/manifests/deployment.yaml"]
    )

# Pipeline definition
@dsl.pipeline(name="churn-training-pipeline")
def pipeline():
    p = prepare_op()
    t = train_op().after(p)
    e = evaluate_op().after(t)
    v = validate_op().after(e)
    deploy_op().after(v)


# Compile pipeline
if __name__ == "__main__":
    from kfp import compiler

    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="pipeline.yaml"
    )
