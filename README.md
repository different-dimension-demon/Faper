# Faper
Query optimizers rely on accurate cardinality estimation for optimal execution plans. Cardinality estimation's main challenge is modeling attribute joint distributions precisely. Despite much research, current learning-based approaches often lack accuracy, can't quantify node-level uncertainty in query plan trees, and can't adapt to workload changes, which are crucial for robust decision-making.

We propose Faper, a query-driven cardinality estimation method that is fast in probability computation, precise in estimation quality, and robust to workload drifts. Faper uses a join tree inference mechanism, focusing on node relationships rather than just the query plan structure. By combining Spiking Neural Networks and Simple Recurrent Units, it captures node intricacies, improving accuracy and training efficiency. Bayesian estimation techniques in Faper enable uncertainty quantification for a comprehensive model performance assessment.

This project presents the realization of Faper

# Fast Start
In our experiments, We conduct experimets on Python 3.8

## Model Training and Model Testing
Train Faper model and get the basic result: (IMDB and STATS are both available)

```shell
python Faper.py
```
When testing the existing model, set the model to 'Test' and get the tested result

# Full Process
This is the full process of training the Faper model on any datasets you use, here we take STATS as the example

## Deployment
Follow the instruction of the open-source benchmark [End-to-End CardEst Benchmark](https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark/tree/master) to build the database.

```bash
git clone git@github.com:Nathaniel-Han/End-to-End-CardEst-Benchmark.git
cd End-to-End-CardEst-Benchmark
bash benchmark_builder.sh
cd postgresql-13.1/
./configure --prefix=/usr/local/pgsql/13.1 --enable-depend --enable-cassert --enable-debug CFLAGS="-ggdb -O0"
make && sudo make install
echo 'export PATH=/usr/local/pgsql/13.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/pgsql/13.1/lib/:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
initdb -D $PSQL_DATA_DIRECTORY
postgres -D $PSQL_DATA_DIRECTORY
```
It is recommended to create a new user. If you do so, replace the last five commands with the following commands.
```bash
adduser postgresuser
su-postgresuser
echo 'export PATH=/usr/local/pgsql/13.1/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/pgsql/13.1/lib/:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
mkdir ~/my_postgres_data
initdb -D ~/my_postgres_data
postgres -D ~/my_postgres_data
```
## Data Import
Activate the postgresql and create the database.
```sql
create database stats;
\c stats;
\i datasets/stats_simplified/stats.sql;
\i scripts/sql/stats_load.sql;
\i scripts/sql/stats_index.sql;
```
## Workload Generation
```shell
cd workload_generator
python sql_gen.py
```
After the training data has been generated, the next step is to execute the queries within the PostgreSQL database. In this process, it is necessary to utilize the 'EXPLAIN (ANALYZE, FORMAT JSON)' command. This particular command plays a crucial role as it is used to generate the JSON format of the syntax analysis tree.

Once the JSON format of the syntax analysis tree has been successfully generated, the subsequent task is to load it into the 'output.json' file. This loading process is an important part of the overall workflow as it prepares the data for further processing.

After the data has been loaded into 'output.json', it is then time to run the following code. This code is designed to perform specific operations based on the data that has been prepared through the previous steps, and it is an integral part of the entire process related to handling the syntax analysis tree in JSON format.

```shell
python datadealer.py
python workload_gen.py
```

## Model Training and Model Testing
Modify the path-related information in 'Faper.py' and then train the corresponding model.
```shell
python Faper.py
```

# Acknowledgments
This work makes use of [End-to-End CardEst Benchmark](https://github.com/Nathaniel-Han/End-to-End-CardEst-Benchmark/tree/master) and [LPCE](https://github.com/Eilowangfang/LPCE). We are grateful for their contribution to the open-source community.


