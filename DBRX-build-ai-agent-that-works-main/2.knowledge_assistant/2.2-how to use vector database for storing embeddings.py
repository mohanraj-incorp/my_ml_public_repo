# Databricks notebook source
# Path to the uploaded file
file_path = "foodly_company_documents.txt"

# Read the entire text
with open(file_path, "r") as f:
    policy_text = f.read()
    print(policy_text)


# COMMAND ----------

from langchain_community.document_loaders import TextLoader

file_path = "foodly_company_documents.txt"
# Load the document
loader = TextLoader(file_path, encoding="utf-8")
documents = loader.load()

print(f"Loaded {len(documents)} document(s).")

print(documents[0].metadata)
print(documents[0].page_content[:1000])


# COMMAND ----------

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,      # ~800 characters per chunk
    chunk_overlap=100,   # 100 characters overlap to preserve context
    separators=["\n\n", "\n", " ", ""]

)

docs = text_splitter.split_documents(documents)

print(f"Created {len(docs)} chunks.")

# "\n\n" → try to split by paragraphs first.

# "\n" → if still too big, split by lines.

# " " → if still too big, split by words.

# "" → as a last resort, split character by character.


print(f"Created {len(docs)} chunks.")



for _doc in docs:
    print(_doc)
    print("-"*100)

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC create catalog if not exists agents;
# MAGIC create schema if not exists agents.main;

# COMMAND ----------

# Prepare data for Delta
chunk_data = []
for i, d in enumerate(docs):
    chunk_data.append({
        "chunk_id": i + 1,
        "content": d.page_content,
        "metadata": str(d.metadata)  # store metadata as JSON-like string
    })

# Convert to dataframe

spark_df = spark.createDataFrame(chunk_data)

display(spark_df)

# Save as Delta
spark_df.write.format("delta").mode("overwrite").saveAsTable("agents.main.foodly_policy_chunks")

# COMMAND ----------

spark.sql("ALTER TABLE agents.main.foodly_policy_chunks SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Now you can create vector serach index using this table, Follow the below doc
# MAGIC
# MAGIC https://docs.databricks.com/aws/en/generative-ai/create-query-vector-search
