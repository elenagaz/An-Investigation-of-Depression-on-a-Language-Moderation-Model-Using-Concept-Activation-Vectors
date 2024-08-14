import pyarrow as pa
import pyarrow.ipc as ipc


def arrow_to_jsonl(arrow_file, jsonl_file):
    with pa.memory_map(arrow_file, 'r') as source:
        reader = ipc.open_stream(source)
        batches = [batch for batch in reader]
        table = pa.Table.from_batches(batches)

    # use panda for the data because of error
    df = table.to_pandas()
    df.to_json(jsonl_file, orient='records', lines=True)


# arrow_file_path = 'train_data-00000-of-00001.arrow'
# jsonl_file_path = 'converted_train_data-00000-of-00001.jsonl'

arrow_file_path = 'KoalaAI_Text-Moderation-v2-small/valid_data-00000-of-00001.arrow'
jsonl_file_path = 'KoalaAI_Text-Moderation-v2-small/converted_valid_data-00000-of-00001.jsonl'

arrow_to_jsonl(arrow_file_path, jsonl_file_path)
