# Databricks notebook source
# DBTITLE 1,Log model
from main import UC_TOOL_NAMES,VECTOR_SEARCH_TOOL

import mlflow
from mlflow.models.resources import DatabricksFunction
from pkg_resources import get_distribution


resources = []

resources.extend(VECTOR_SEARCH_TOOL.resources)


for tool_name in UC_TOOL_NAMES:
    resources.append(DatabricksFunction(function_name=tool_name))


with mlflow.start_run():
    logged_agent_info = mlflow.pyfunc.log_model(
        name="foodly-ai-assistant",
        code_paths=["helpers.py"],
        python_model='main.py',
        pip_requirements=[
            f"databricks-connect=={get_distribution('databricks-connect').version}",
            f"mlflow=={get_distribution('mlflow').version}",
            f"databricks-langchain=={get_distribution('databricks-langchain').version}",
            f"langgraph=={get_distribution('langgraph').version}",
             f"langgraph-supervisor=={get_distribution('langgraph-supervisor').version}",
        ],
        resources=resources,
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### Pre-deployment agent validation
# MAGIC

# COMMAND ----------

mlflow.models.predict(
    model_uri=f"runs:/{logged_agent_info.run_id}/foodly-ai-assistant",
    input_data={"input": [{"role": "user", "content": "Tell me about foodly return policy"}]},
    env_manager="uv",
)

# COMMAND ----------

# DBTITLE 1,Register the model
mlflow.set_registry_uri("databricks-uc")

# TODO: define the catalog, schema, and model name for your UC model
catalog = "agents"
schema = "main"
model_name = "foodly-ai-assistant"
UC_MODEL_NAME = f"{catalog}.{schema}.{model_name}"

# register the model to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_agent_info.model_uri, name=UC_MODEL_NAME)

# COMMAND ----------

# DBTITLE 1,Deploy to model serving
from databricks import agents

agents.deploy(
    UC_MODEL_NAME,
    uc_registered_model_info.version,
    scale_to_zero=True,
    tags={"name": "foodly-ai-assistant-endpoint","creator" : "MohanRaj"},
)

# COMMAND ----------

# MAGIC %md
# MAGIC
