# Feature store

Using a feature store to connect the [DataOps](https://madewithml.com/courses/mlops/orchestration/#dataops) and [MLOps](https://madewithml.com/courses/mlops/orchestration/#mlops) workflows to enable collaborative teams to develop efficiently.

<div align="left">
    <a target="_blank" href="https://newsletter.madewithml.com"><img src="https://img.shields.io/badge/Subscribe-30K-brightgreen"></a>&nbsp;
    <a target="_blank" href="https://github.com/GokuMohandas/Made-With-ML"><img src="https://img.shields.io/github/stars/GokuMohandas/Made-With-ML.svg?style=social&label=Star"></a>&nbsp;
    <a target="_blank" href="https://www.linkedin.com/in/goku"><img src="https://img.shields.io/badge/style--5eba00.svg?label=LinkedIn&logo=linkedin&style=social"></a>&nbsp;
    <a target="_blank" href="https://twitter.com/GokuMohandas"><img src="https://img.shields.io/twitter/follow/GokuMohandas.svg?label=Follow&style=social"></a>
    <br>
</div>

<br>

👉 &nbsp;This repository contains the [interactive notebook](https://colab.research.google.com/github/GokuMohandas/feature-store/blob/main/feature_store.ipynb) that complements the [feature store lesson](https://madewithml.com/courses/mlops/feature-store/), which is a part of the [MLOps course](https://github.com/GokuMohandas/mlops-course). If you haven't already, be sure to check out the [lesson](https://madewithml.com/courses/mlops/feature-store/) because all the concepts are covered extensively and tied to software engineering best practices for building ML systems.

<div align="left">
<a target="_blank" href="https://madewithml.com/courses/mlops/feature-store/"><img src="https://img.shields.io/badge/📖 Read-lesson-9cf"></a>&nbsp;
<a href="https://github.com/GokuMohandas/feature-store/blob/main/feature_store.ipynb" role="button"><img src="https://img.shields.io/static/v1?label=&amp;message=View%20On%20GitHub&amp;color=586069&amp;logo=github&amp;labelColor=2f363d"></a>&nbsp;
<a href="https://colab.research.google.com/github/GokuMohandas/feature-store/blob/main/feature_store.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
</div>

<br>

- [Motivation](#motivation)
- [Feast](#feast)
    - [Data ingestion](#data-ingestion)
    - [Feature definitions](#feature-definitions)
    - [Historical features](#historical-features)
    - [Materialize](#materialize)
    - [Online features](#online-features)
- [Over-engineering](#over-engineering)

## Motivation

Let's motivate the need for a feature store by chronologically looking at what challenges developers face in their current workflows. Suppose we had a task where we needed to predict something for an entity (ex. user) using their features.

1. **Isolation**: feature development in isolation (for each unique ML application) can lead to duplication of efforts (setting up ingestion pipelines, feature engineering, etc.).
    - `Solution`: create a central feature repository where the entire team contributes maintained features that anyone can use for any application.
2. **Skew**: we may have different pipelines for generating features for training and serving which can introduce skew through the subtle differences.
    - `Solution`: create features using a unified pipeline and store them in a central location that the training and serving pipelines pull from.
3. **Values**: once we set up our data pipelines, we need to ensure that our input feature values are up-to-date so we aren't working with stale data, while maintaining point-in-time correctness so we don't introduce data leaks.
    - `Solution`: retrieve input features for the respective outcomes by pulling what's available when a prediction would be made.

Point-in-time correctness refers to mapping the appropriately up-to-date input feature values to an observed outcome at $t_{n+1}$. This involves knowing the time ($t_n$) that a prediction is needed so we can collect feature values ($X$) at that time.
<div class="ai-center-all">
    <img src="https://madewithml.com/static/images/mlops/feature_store/point_in_time.png" width="700" alt="point-in-time correctness">
</div>

When actually constructing our feature store, there are several core components we need to have to address these challenges:

- **data ingestion**: ability to ingest data from various sources (databases, data warehouse, etc.) and keep them updated.
- **feature definitions**: ability to define entities and corresponding features
- **historical features**: ability to retrieve historical features to use for training.
- **online features**: ability to retrieve features from a low latency origin for inference.

> **Over-engineering**: Not all machine learning tasks require a feature store. In fact, our use case is a perfect example of a test that *does not* benefit from a feature store. All of our data points are independent and stateless and there is no entity that has changing features over time. The real utility of a feature store shines when we need to have up-to-date features for an entity that we continually generate predictions for. For example, a user's behavior (clicks, purchases, etc.) on an e-commerce platform or the deliveries a food runner recently made today, etc.

## Feast

We're going to leverage [Feast](https://feast.dev/) as the feature store for our application for it's ease of local setup, SDK for training/serving, etc.

```bash
# Install Feast and dependencies
pip install feast==0.10.5 PyYAML==5.3.1 -q
```

We're going to create a feature repository at the root of our project. [Feast](https://feast.dev/) will create a configuration file for us and we're going to add an additional [features.py](https://github.com/GokuMohandas/mlops-course/blob/main/features/features.py) file to define our features.

> Traditionally, the feature repository would be it's own isolated repository that other services will use to read/write features from.

```bash
%%bash
mkdir -p stores/feature
mkdir -p data
feast init --minimal --template local features
cd features
touch features.py
```

```bash
features/
├── feature_store.yaml  - configuration
└── features.py         - feature definitions
```

We're going to configure the locations for our registry and online store (SQLite) in our `feature_store.yaml` file.

<div class="ai-center-all">
    <img src="https://madewithml.com/static/images/mlops/feature_store/batch.png" width="700" alt="batch processing">
</div>

<br>

- **registry**: contains information about our feature repository, such as data sources, feature views, etc. Since it's in a DB, instead of a Python file, it can very quickly be accessed in production.
- **online store**: DB (SQLite for local) that stores the (latest) features for defined entities to be used for online inference.

If all definitions look valid, Feast will sync the metadata about Feast objects to the registry. The registry is a tiny database storing most of the same information you have in the feature repository. This step is necessary because the production feature serving infrastructure won't be able to access Python files in the feature repository at run time, but it will be able to efficiently and securely read the feature definitions from the registry.

```yaml
# features/feature_store.yaml
project: features
registry: ../stores/feature/registry.db
provider: local
online_store:
    path: ../stores/feature/online_store.db
```

> When we run Feast locally, the offline store is effectively represented via Pandas point-in-time joins. Whereas, in production, the offline store can be something more robust like [Google BigQuery](https://cloud.google.com/bigquery), [Amazon RedShift](https://aws.amazon.com/redshift/), etc.

### Data ingestion

The first step is to establish connections with our data sources (databases, data warehouse, etc.). Feast requires it's [data sources](https://github.com/feast-dev/feast/blob/master/sdk/python/feast/data_source.py) to either come from a file ([Parquet](https://databricks.com/glossary/what-is-parquet)), data warehouse ([BigQuery](https://cloud.google.com/bigquery)) or data stream ([Kafka](https://kafka.apache.org/) / [Kinesis](https://aws.amazon.com/kinesis/)). We'll convert our generated features file from the DataOps pipeline (`features.json`) into a Parquet file, which is a column-major data format that allows fast feature retrieval and caching benefits (contrary to row-major data formats such as CSV where we have to traverse every single row to collect feature values).

```python
import os
import json
import pandas as pd
from pathlib import Path
from urllib.request import urlopen
```

```python
# Load projects
url = "https://raw.githubusercontent.com/GokuMohandas/Made-With-ML/main/datasets/projects.json"
projects = json.loads(urlopen(url).read())
df = pd.DataFrame(projects)
df["text"] = df.title + " " + df.description
df.drop(["title", "description"], axis=1, inplace=True)
df.head(5)
```

<div class="output_subarea output_html rendered_html output_result" dir="auto"><div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>created_on</th>
      <th>title</th>
      <th>description</th>
      <th>tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>2020-02-20 06:43:18</td>
      <td>Comparison between YOLO and RCNN on real world...</td>
      <td>Bringing theory to experiment is cool. We can ...</td>
      <td>computer-vision</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>2020-02-20 06:47:21</td>
      <td>Show, Infer &amp; Tell: Contextual Inference for C...</td>
      <td>The beauty of the work lies in the way it arch...</td>
      <td>computer-vision</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>2020-02-24 16:24:45</td>
      <td>Awesome Graph Classification</td>
      <td>A collection of important graph embedding, cla...</td>
      <td>graph-learning</td>
    </tr>
    <tr>
      <th>3</th>
      <td>15</td>
      <td>2020-02-28 23:55:26</td>
      <td>Awesome Monte Carlo Tree Search</td>
      <td>A curated list of Monte Carlo tree search papers...</td>
      <td>reinforcement-learning</td>
    </tr>
    <tr>
      <th>4</th>
      <td>19</td>
      <td>2020-03-03 13:54:31</td>
      <td>Diffusion to Vector</td>
      <td>Reference implementation of Diffusion2Vec (Com...</td>
      <td>graph-learning</td>
    </tr>
  </tbody>
</table>
</div></div>

```python
# Format timestamp
df.created_on = pd.to_datetime(df.created_on)
```

```python
# Convert to parquet
DATA_DIR = Path(os.getcwd(), "data")
df.to_parquet(
    Path(DATA_DIR, "features.parquet"),
    compression=None,
    allow_truncated_timestamps=True,
)
```

### Feature definitions

Now that we have our data source prepared, we can define our features for the feature store.

```python
from datetime import datetime
from pathlib import Path
from feast import Entity, Feature, FeatureView, ValueType
from feast.data_source import FileSource
from google.protobuf.duration_pb2 import Duration
```

The first step is to define the location of the features (FileSource in our case) and the timestamp column for each data point.

```python
# Read data
START_TIME = "2020-02-17"
project_details = FileSource(
    path=str(Path(DATA_DIR, "features.parquet")),
    event_timestamp_column="created_on",
)
```

Next, we need to define the main entity that each data point pertains to. In our case, each project has a unique ID with features such as text and tags.

```python
# Define an entity
project = Entity(
    name="id",
    value_type=ValueType.INT64,
    description="project id",
)
```

Finally, we're ready to create a [FeatureView](https://docs.feast.dev/concepts/feature-views) that loads specific features (`features`), of various [value types](https://api.docs.feast.dev/python/feast.html?highlight=valuetype#feast.value_type.ValueType), from a data source (`input`) for a specific period of time (`ttl`).

```python
# Define a Feature View for each project
project_details_view = FeatureView(
    name="project_details",
    entities=["id"],
    ttl=Duration(
        seconds=(datetime.today() - datetime.strptime(START_TIME, "%Y-%m-%d")).days * 24 * 60 * 60
    ),
    features=[
        Feature(name="text", dtype=ValueType.STRING),
        Feature(name="tag", dtype=ValueType.STRING),
    ],
    online=True,
    input=project_details,
    tags={},
)
```

We need to place all of this code into our `features.py` file we created earlier. Once we've defined our feature views, we can `apply` it to push a version controlled definition of our features to the registry for fast access. It will also configure our registry and online stores that we've defined in our `feature_store.yaml`.

```bash
cd features
feast apply
```

```bash
Registered entity id
Registered feature view project_details
Deploying infrastructure for project_details
```

### Historical features

Once we've registered our feature definition, along with the data source, entity definition, etc., we can use it to fetch historical features. This is done via joins using the provided timestamps using pandas for our local setup or BigQuery, Hive, etc. as an offline DB for production.

```python
import pandas as pd
from feast import FeatureStore
```

```python
# Identify entities
project_ids = df.id[0:3].to_list()
now = datetime.now()
timestamps = [datetime(now.year, now.month, now.day)]*len(project_ids)
entity_df = pd.DataFrame.from_dict({"id": project_ids, "event_timestamp": timestamps})
entity_df.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>event_timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6</td>
      <td>2022-06-23</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7</td>
      <td>2022-06-23</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9</td>
      <td>2022-06-23</td>
    </tr>
  </tbody>
</table>
</div>

```python
# Get historical features
store = FeatureStore(repo_path="features")
training_df = store.get_historical_features(
    entity_df=entity_df,
    feature_refs=["project_details:text", "project_details:tag"],
).to_df()
training_df.head()
```

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>event_timestamp</th>
      <th>id</th>
      <th>project_details__text</th>
      <th>project_details__tag</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2022-06-23 00:00:00+00:00</td>
      <td>6</td>
      <td>Comparison between YOLO and RCNN on real world...</td>
      <td>computer-vision</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2022-06-23 00:00:00+00:00</td>
      <td>7</td>
      <td>Show, Infer &amp; Tell: Contextual Inference for C...</td>
      <td>computer-vision</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2022-06-23 00:00:00+00:00</td>
      <td>9</td>
      <td>Awesome Graph Classification A collection of i...</td>
      <td>graph-learning</td>
    </tr>
  </tbody>
</table>
</div>

### Materialize

For online inference, we want to retrieve features very quickly via our online store, as opposed to fetching them from slow joins. However, the features are not in our online store just yet, so we'll need to [materialize](https://docs.feast.dev/quickstart#4-materializing-features-to-the-online-store) them first.

```bash
cd features
CURRENT_TIME=$(date -u +"%Y-%m-%dT%H:%M:%S")
feast materialize-incremental $CURRENT_TIME
```
```
Materializing 1 feature views to 2022-06-23 19:16:05+00:00 into the sqlite online store.
```

This has moved the features for all of our projects into the online store since this was first time materializing to the online store. When we subsequently run the [`materialize-incremental`](https://docs.feast.dev/how-to-guides/load-data-into-the-online-store#2-b-materialize-incremental-alternative) command, Feast keeps track of previous materializations and so we'll only materialize the new data since the last attempt.

### Online features

```python
# Get online features
store = FeatureStore(repo_path="features")
feature_vector = store.get_online_features(
    feature_refs=["project_details:text", "project_details:tag"],
    entity_rows=[{"id": 6}],
).to_dict()
feature_vector
```

```json
{"id": [6],
 "project_details__tag": ["computer-vision"],
 "project_details__text": ["Comparison between YOLO and RCNN on real world videos Bringing theory to experiment is cool. We can easily train models in colab and find the results in minutes."]}
```

## Over-engineering

Not all machine learning tasks require a feature store. In fact, our use case is a perfect example of a test that *does not* benefit from a feature store. All of our inputs are stateless and there is no entity that has changing features over time. The real utility of a feature store shines when we need to have up-to-date features for an entity that we continually generate predictions for. For example, a user's behavior (clicks, purchases, etc.) on an e-commerce platform or the deliveries a food runner recently made today, etc.