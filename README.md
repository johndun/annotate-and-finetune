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

- prompt-yaml-path (default to scripts/example_annotation_instructions.yaml)
- input-data-path (default to ~/data/taskmaster2/taskmaster2_dialogs.jsonl)
- output-data-path (default to ~/data/taskmaster2/taskmaster2_dialogs_annotated.jsonl)
- num-proc (default to 1)