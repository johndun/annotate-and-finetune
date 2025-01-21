# annotate-and-finetune

Package to perform LLM-based annotation and text classification model fine tuning.

```bash
git clone https://github.com/google-research-datasets/Taskmaster

python scripts/prepare_taskmaster2_dialog_dataset.py --help
python scripts/prepare_taskmaster2_turn_dataset.py --help
```


## Aider

```bash
aider --sonnet --no-analytics --read examples.py --read scripts/example_data.json scripts/prepare_taskmaster2_dialog_dataset.py
```

### Prompt 1

Write a command line script that prepares the `google-research-datasets/taskmaster2` dataset by creating a single jsonlines file containing dictionaries with "dialog" and "label" fields.

- Default input directory: `~/Work/Taskmaster/TM-2-2020/data` contains files: flights.json, food-ordering.json, hotels.json, movies.json, music.json, restaurant-search.json, sports.json
- Default output directory: `~/data/taskmaster2`
- Include a link to the repo in the script docstring

### Prompt 2

Update the script so that it converts the dialog into a string representation with USER and ASSISTANT turns.

### Prompt 3

Update the script so that the first message is deleted if it is an ASSISTANT message.

### Prompt 4

Update the script to replace output_dir with output_path

```bash
aider --sonnet --no-analytics --read examples.py --read scripts/example_data.json  --read scripts/prepare_taskmaster2_dialog_dataset.py scripts/prepare_taskmaster2_turn_dataset.py
```

### Prompt 1

Write a command line script that inputs a path to the output of scripts/prepare_taskmaster2_dialog_dataset.py and performs the following transformation:

- For each row of the dialog dataset, generate a new dataset with one row per turn with each row containing all previous dialog turns (cumulative).
- Default output directory: `~/data/taskmaster2/`

### Prompt 2

Update the script so that each row contains both a new USER and a new ASSISTANT message. Currently, each turn contains a single new message.

### Prompt 3

Update the script to replace output_dir with output_path

```bash
aider --sonnet --no-analytics --read examples.py --read LLMPIPE_VIGNETTE.md --read scripts/example_annotation_instructions.yaml scripts/annotate.py
```

Write a command line script that:

- loads a config from a yaml file and uses it to initialize a `llmpipe.PromptModule` instance
- loads a dataset
- runs the dataset through the prompt
- saves the dataset

Command line args:

- prompt-yaml-path (default to scripts/example_annotation_prompt.yaml)
- input-data-path (default to ~/data/taskmaster2/taskmaster2_dialogs.jsonl)
- output-data-path (default to ~/data/taskmaster2/taskmaster2_dialogs_annotated.jsonl)
- num-proc (default to 1)
- model (default to claude-3-5-sonnet-20241022)
- verbose (default to False)

```bash
aider --sonnet --no-analytics --read scripts/annotate.py  --read scripts/ex_anno_refinement_prompt.yaml scripts/refine_annotation_classes.py
```

Write a command line script that:

- loads a config from a yaml file and uses it to initialize a `llmpipe.PromptModule` instance
- loads a dataset with a "label" field
- creates a markdown table of the top 500 most frequently occurring labels
- passes this to the "labels" input of the prompt module instance.
- extracts the "refined_labels" key from the response and saves it to a jsonlines file

Command line args:

- prompt-yaml-path (default to scripts/ex_anno_refinement__prompt.yaml)
- input-data-path (default to ~/data/taskmaster2/taskmaster2_dialogs_annotated.jsonl)
- output-data-path (default to ~/data/taskmaster2/refined_labels.jsonl)
- num-proc (default to 1)
- model (default to claude-3-5-sonnet-20241022)
- verbose (default to False)

```bash
aider --sonnet --no-analytics scripts/annotate.py
```

Update the script so that it has an allowed_classes_path argument that defaults to None. If this field is present, the file should be loaded (it should be a jsonlines file), converted to a markdown list, and added to `data`.



```bash
python scripts/prepare_taskmaster2_dialog_dataset.py
python scripts/prepare_taskmaster2_turn_dataset.py
python scripts/annotate.py \
    --prompt-yaml-path scripts/ex_annotation_prompt.yaml \
    --input-data-path ~/data/taskmaster2/taskmaster2_dialogs.jsonl \
    --output-data-path ~/data/taskmaster2/taskmaster2_dialogs_annotated.jsonl \
    --n-samples 100 \
    --model claude-3-5-haiku-20241022
python scripts/refine_annotation_classes.py \
    --input-data-path ~/data/taskmaster2/taskmaster2_dialogs_annotated.jsonl \
    --output-data-path ~/data/taskmaster2/refined_labels.jsonl \
    --verbose
python scripts/annotate.py \
    --prompt-yaml-path scripts/ex_annotation_prompt2.yaml \
    --input-data-path ~/data/taskmaster2/taskmaster2_dialogs.jsonl \
    --output-data-path ~/data/taskmaster2/taskmaster2_dialogs_annotated_refined_classes.jsonl \
    --allowed-labels-path ~/data/taskmaster2/refined_labels.jsonl \
    --n-samples 100 \
    --model claude-3-5-haiku-20241022
```




```python
import polars as pl
from llmpipe import read_data
from annotate_and_finetune.summarize_list import summarize_list
df = read_data("~/data/taskmaster2/taskmaster2_dialogs_annotated_refined_classes.jsonl", as_df=True)
data = df.to_dict(as_series=False)

result = []
result.append("## Dialog Categories")

labels = read_data("~/data/taskmaster2/refined_labels.jsonl", as_df=False)
label_str = []
for x in labels:
    label_str.append(f"- {x['label']}: {x['description']}")

result.append("\n".join(label_str))

result.append(f"## Data Summary: dialog")
result.append(summarize_list(data["dialog"], n_examples=3))

result.append(f"## Data Summary: Annotation Labels")
result.append(summarize_list(data["label"], n_examples=100))

print("\n\n".join(result))
```